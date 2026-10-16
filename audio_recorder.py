import soundcard as sc
import soundfile as sf
import contextlib
import queue
import types
import threading
import time
import os
import lameenc
import numpy as np
import tempfile

class PortAudioDevice:
    """
    Fallback for microphones soundcard can't open. soundcard only handles
    devices whose Windows mix format is float, some USB mics report 16 bit
    PCM and fail with an empty AssertionError (#11, bastibe/SoundCard#93).
    Mirrors the small part of the soundcard device API that RawRecorder uses.
    """
    def __init__(self, name, index, channels):
        self.name = name
        self.index = index
        self.channels = channels

    @classmethod
    def find(cls, name):
        try:
            import sounddevice as sd
        except Exception:
            return None
        wasapi = [i for i, h in enumerate(sd.query_hostapis()) if "WASAPI" in h["name"]]
        for i, d in enumerate(sd.query_devices()):
            if d["hostapi"] in wasapi and d["max_input_channels"] > 0 and d["name"] == name:
                return cls(name, i, d["max_input_channels"])
        return None

    @contextlib.contextmanager
    def recorder(self, samplerate, channels):
        import sounddevice as sd
        # Callback mode: blocking reads of 2048 frames dropped audio over WASAPI
        blocks = queue.Queue()
        stream = sd.InputStream(device=self.index, samplerate=samplerate, channels=channels,
                                dtype="float32", extra_settings=sd.WasapiSettings(auto_convert=True),
                                callback=lambda data, frames, t, status: blocks.put(data.copy()))
        pending = np.zeros((0, channels), dtype="float32")

        def record(numframes):
            nonlocal pending
            while len(pending) < numframes:
                pending = np.concatenate((pending, blocks.get(timeout=2)))
            data, pending = pending[:numframes], pending[numframes:]
            return data

        stream.start()
        try:
            yield types.SimpleNamespace(record=record)
        finally:
            stream.stop()
            stream.close()

class RawRecorder(threading.Thread):
    """
    Helper thread to record a single device to a WAV file.
    """
    def __init__(self, device, filepath, samplerate=44100, channels=2):
        super().__init__()
        self.device = device
        self.filepath = filepath
        self.samplerate = samplerate
        self.channels = channels
        self.stop_event = threading.Event()
        self.error = None
        # Frames written so far: tells a device that died after a while from
        # one that fails to start at all, the AudioRecorder needs that.
        self.frames = 0

    def _probe_params(self):
        """Find working (samplerate, channels) for this device."""
        channels = self.channels
        if hasattr(self.device, 'channels'):
            channels = min(self.channels, self.device.channels)

        # requested values first, fallbacks after (dict keeps order, drops dupes)
        rates = dict.fromkeys([self.samplerate, 48000, 44100])
        channel_options = dict.fromkeys([channels, 1])

        failures = []
        for sr in rates:
            for ch in channel_options:
                try:
                    with self.device.recorder(samplerate=sr, channels=ch):
                        return sr, ch
                except Exception as e:
                    # soundcard raises a bare AssertionError for some mics, keep the type visible
                    failures.append(f"  {sr}Hz/{ch}ch: {str(e) or type(e).__name__}")

        if not isinstance(self.device, PortAudioDevice):
            fallback = PortAudioDevice.find(self.device.name)
            if fallback:
                print(f"soundcard can't open '{self.device.name}', trying PortAudio")
                self.device = fallback
                return self._probe_params()

        raise Exception(
            f"Device '{self.device.name}' rejected all parameter combinations:\n"
            + "\n".join(failures)
        )

    def probe(self):
        self.samplerate, self.channels = self._probe_params()

    def run(self):
        try:
            samplerate, channels = self.samplerate, self.channels
            with sf.SoundFile(self.filepath, mode='w', samplerate=samplerate, channels=channels) as f_wav:
                try:
                    with self.device.recorder(samplerate=samplerate, channels=channels) as mic:
                        while not self.stop_event.is_set():
                            data = mic.record(numframes=2048)
                            f_wav.write(data)
                            self.frames += len(data)
                except Exception as e:
                    # The device was probably unplugged. Keep the frames written
                    # so far instead of throwing the whole recording away, the
                    # AudioRecorder decides whether to reconnect or to save.
                    self.error = str(e) or type(e).__name__
        except Exception as e:
            self.error = str(e) or type(e).__name__

    def stop(self, timeout=5):
        self.stop_event.set()
        self.join(timeout=timeout)
        # False means the device hung and the thread is still stuck in it
        return not self.is_alive()

class AudioRecorder(threading.Thread):
    """
    Orchestrates recording from Microphone, Loopback, or Both.

    If a device disappears mid-recording:
      - reconnect=True  -> wait for the device, resume into a new segment,
        all segments are concatenated at the end;
      - reconnect=False -> finalize and save what was captured right away.
    on_notify_callback reports both events immediately.
    """
    def __init__(self, mic_id, source_mode, output_folder, output_format="mp3", 
                 normalize=False, mono=False, on_finish_callback=None, on_progress=None,
                 reconnect=True, on_notify_callback=None):
        super().__init__()
        self.mic_id = mic_id
        self.source_mode = source_mode # "mic", "loopback", "both"
        self.output_folder = output_folder
        self.output_format = output_format.lower()
        self.normalize = normalize
        self.mono = mono
        self.callback = on_finish_callback
        self.on_progress = on_progress  # called with 0..1 while saving
        self.reconnect = reconnect
        self.notify_callback = on_notify_callback

        # name of the mic in use: the id is gone after a replug
        self.mic_name = None
        self.recording = False
        self.stop_event = threading.Event()
        self.error_message = None
        self.final_filepath = None
        
        # Temp files
        self.temp_files = []
        self.sources = []

    def _get_device(self, is_loopback):
        if is_loopback:
            default_speaker = sc.default_speaker()
            mics = sc.all_microphones(include_loopback=True)
            loopback_mic = next((m for m in mics if m.name == default_speaker.name), None)
            if not loopback_mic:
                loopback_mic = next((m for m in mics if default_speaker.name in m.name), None)
            if not loopback_mic:
                available = [m.name for m in mics if m.isloopback] if any(hasattr(m, 'isloopback') for m in mics) else [m.name for m in mics]
                raise Exception(
                    f"No loopback device found for speaker '{default_speaker.name}'. "
                    f"Available loopback devices: {available}"
                )
            return loopback_mic
        else:
            return self._get_mic_device()

    def _get_mic_device(self):
        # The saved id is stable while the device stays on the same port,
        # after a replug it may change, so fall back to the name.
        first_error = None
        try:
            dev = sc.get_microphone(self.mic_id, include_loopback=False)
            if dev is not None:
                return dev
        except Exception as e:
            first_error = e

        if self.mic_name:
            try:
                mics = sc.all_microphones(include_loopback=False)
                dev = next((m for m in mics if m.name == self.mic_name), None)
                if dev is not None:
                    return dev
            except Exception:
                pass

        available = [m.name for m in sc.all_microphones()]
        detail = f" ({first_error})" if first_error else ""
        raise Exception(
            f"Microphone '{self.mic_id or self.mic_name}' not found{detail}. "
            f"Available: {available}"
        )

    def _notify(self, message):
        if self.notify_callback:
            try:
                self.notify_callback(message)
            except Exception:
                pass
        print(message)

    def _new_recorder(self, src, device):
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        self.temp_files.append(path)
        src["files"].append(path)
        if not src["is_loop"]:
            # remember the name: it is what a reconnect can match on
            self.mic_name = device.name
        rec = RawRecorder(device, path,
                          samplerate=src["samplerate"] or 44100,
                          channels=src["channels"] or 2)
        src["rec"] = rec
        return rec

    def _start_recorder(self, src, device):
        rec = self._new_recorder(src, device)
        rec.start()
        return rec

    def run(self):
        self.recording = True
        self.error_message = None
        self.temp_files = []
        self.sources = []
        recorder_errors = []

        try:
            # 1. Setup Recorders
            if self.source_mode == "both":
                modes = [False, True]  # is_loopback flags: mic + system audio
            elif self.source_mode == "loopback":
                modes = [True]
            else: # mic
                modes = [False]

            for is_loop in modes:
                dev = self._get_device(is_loop)
                src = {"is_loop": is_loop, "files": [], "rec": None,
                       "failed_starts": 0, "samplerate": None, "channels": None,
                       "lost": False}
                self.sources.append(src)
                rec = self._new_recorder(src, dev)

                # Probe before starting so both sources start close together.
                # The loopback gets asked for the mic's rate first, so mixing
                # rarely has to resample.
                if src is not self.sources[0]:
                    rec.samplerate = self.sources[0]["rec"].samplerate
                try:
                    rec.probe()
                except Exception as e:
                    raise Exception(f"Recording failed on '{rec.device.name}': {e}") from e
                # Keep the probed values: a reconnected segment has to match
                # them, otherwise the segments can't be concatenated.
                src["samplerate"], src["channels"] = rec.samplerate, rec.channels

            print(f"Starting recording mode: {self.source_mode}")

            # 2. Start Recording
            for src in self.sources:
                src["rec"].start()

            # 3. Wait for the stop signal, watching for lost devices
            while not self.stop_event.is_set():
                self.stop_event.wait(0.5)
                if self.stop_event.is_set():
                    break

                for src in self.sources:
                    rec = src["rec"]
                    if rec is None:
                        continue
                    if rec.error is None and rec.is_alive():
                        continue

                    # The device died (or the recorder crashed)
                    rec.stop()
                    if rec.error:
                        recorder_errors.append(rec.error)
                    src["rec"] = None

                    label = "System audio" if src["is_loop"] else "Microphone"

                    # Count instant failures (device "present" but dies right
                    # away) to avoid an endless reconnect loop.
                    if rec.frames == 0:
                        src["failed_starts"] += 1
                    else:
                        src["failed_starts"] = 0

                    if src["failed_starts"] >= 5:
                        self._notify(f"{label} keeps failing - saving recording...")
                        src["lost"] = True
                        self.stop_event.set()
                        break

                    if not self.reconnect:
                        # Finalize immediately: keep what was recorded.
                        self._notify(f"{label} disconnected - saving recording...")
                        src["lost"] = True
                        self.stop_event.set()
                        break

                    self._notify(f"{label} disconnected - waiting for reconnection...")

                    while not self.stop_event.is_set():
                        self.stop_event.wait(1.0)
                        if self.stop_event.is_set():
                            break
                        try:
                            dev = self._get_device(src["is_loop"])
                            self._start_recorder(src, dev)
                            self._notify(f"{label} reconnected - recording resumed.")
                            break
                        except Exception:
                            continue

                    if self.stop_event.is_set():
                        # stopped by the user, or the device never came back
                        if src["rec"] is None:
                            src["lost"] = True
                        break

            # 4. Stop Recording
            for src in self.sources:
                rec = src["rec"]
                if rec is None:
                    continue
                if not rec.stop() and rec.error is None:
                    rec.error = "Recorder thread did not stop (device hung)."
                if rec.error:
                    recorder_errors.append(rec.error)
                src["rec"] = None

            # Keep whatever was captured instead of dropping the whole file.
            has_audio = any(
                os.path.exists(t) and os.path.getsize(t) > 44  # > WAV header
                for t in self.temp_files
            )
            if not has_audio:
                reason = recorder_errors[0] if recorder_errors else "no audio captured"
                raise Exception(f"Recording failed: {reason}")

            # 5. A reconnected device leaves several segments, glue them back
            source_wavs = []
            for src in self.sources:
                wavs = [f for f in src["files"]
                        if os.path.exists(f) and os.path.getsize(f) > 44]
                if not wavs:
                    continue
                if len(wavs) == 1:
                    source_wavs.append(wavs[0])
                else:
                    combined = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                    self._concat_segments(wavs, combined, src["samplerate"], src["channels"])
                    self.temp_files.append(combined)
                    source_wavs.append(combined)

            if not source_wavs:
                raise Exception("Recording failed: no audio captured")

            # 6. Mix, normalize and encode block by block. Loading a long
            # recording at once needs several GB of RAM (see #7).
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"Recording_{timestamp}.{self.output_format}"
            self.final_filepath = os.path.join(self.output_folder, filename)

            sources = [sf.SoundFile(t) for t in source_wavs]
            try:
                self._write_output(sources, self.final_filepath)
            finally:
                for s in sources:
                    s.close()

            # Saved successfully. Only complain if a device was actually
            # lost for good, a reconnect means the recording is complete.
            if any(src["lost"] for src in self.sources):
                reason = recorder_errors[0] if recorder_errors else "device lost"
                self.error_message = f"Device disconnected, partial recording saved: {reason}"

        except Exception as e:
            self.error_message = str(e)
            print(f"Error during recording process: {e}")
        finally:
            self.recording = False
            # Clean up all temp files
            for t in self.temp_files:
                if os.path.exists(t):
                    try:
                        os.remove(t)
                    except: pass
                
            if self.callback:
                self.callback(self.final_filepath, self.error_message)

    def stop(self):
        self.stop_event.set()

    @staticmethod
    def _concat_segments(files, out_file, samplerate, channels):
        """Glue the segments of one source back into a single WAV file,
        block by block so a long recording doesn't have to fit in RAM."""
        with sf.SoundFile(out_file, 'w', samplerate=samplerate, channels=channels) as f_out:
            for path in files:
                try:
                    with sf.SoundFile(path) as f_in:
                        for block in f_in.blocks(65536, dtype='float32', always_2d=True):
                            f_out.write(block)
                except Exception as e:
                    print(f"Skipping unreadable segment {path}: {e}")

    def _plan_mix(self, sources):
        """Output rate, channels and length, plus a read ratio per source.

        The first source (the mic in "both" mode) sets the rate. A source
        that is off by less than 0.5% gets stretched to the full length,
        which fixes clock drift between devices on long recordings instead
        of letting mic and system audio slowly drift apart.
        """
        samplerate = sources[0].samplerate
        channels = max(s.channels for s in sources)
        lengths = [s.frames * samplerate / s.samplerate for s in sources]
        length = int(round(max(lengths)))

        ratios = []
        for s, n in zip(sources, lengths):
            if length and abs(n - length) / length < 0.005:
                ratios.append(s.frames / length)
            else:
                ratios.append(s.samplerate / samplerate)
        return samplerate, channels, length, ratios

    @staticmethod
    def _read_block(source, start, count, ratio):
        """Reads `count` output frames starting at output frame `start`."""
        if ratio == 1:
            source.seek(min(start, source.frames))
            data = source.read(count, dtype='float32', always_2d=True)
        else:
            pos = (start + np.arange(count)) * ratio
            lo = min(int(pos[0]), source.frames)
            hi = min(int(pos[-1]) + 2, source.frames)
            source.seek(lo)
            chunk = source.read(hi - lo, dtype='float32', always_2d=True)
            data = np.zeros((count, source.channels), dtype='float32')
            inside = pos <= hi - 1
            if len(chunk) and inside.any():
                idx = np.arange(lo, hi)
                for c in range(source.channels):
                    data[inside, c] = np.interp(pos[inside], idx, chunk[:, c])
            return data

        if len(data) < count:
            data = np.concatenate((data, np.zeros((count - len(data), data.shape[1]), dtype='float32')))
        return data

    def _mixed_blocks(self, sources, gains=None, block=65536):
        samplerate, channels, length, ratios = self._plan_mix(sources)
        gains = gains or [1.0] * len(sources)
        for start in range(0, length, block):
            count = min(block, length - start)
            mixed = np.zeros((count, channels), dtype='float32')
            for s, ratio, g in zip(sources, ratios, gains):
                # mono into stereo: same signal on both sides
                mixed += g * self._read_block(s, start, count, ratio)
            yield np.clip(mixed, -1.0, 1.0)

    def _write_output(self, sources, path):
        samplerate, channels, length, _ = self._plan_mix(sources)
        per_source = self.normalize and len(sources) > 1

        # progress over all passes: per-source peaks, mix peak, final write.
        # Encoding is far slower than scanning for peaks, so it counts more
        # (measured on a 30 min recording: mp3 ~25x, wav/flac ~3x a scan)
        write_cost = 25 if self.output_format == "mp3" else 3
        total = length * write_cost
        if self.normalize:
            total += length
        if per_source:
            total += sum(s.frames for s in sources)
        done = 0

        def tick(b, cost=1):
            nonlocal done
            done += len(b) * cost
            if self.on_progress and total:
                self.on_progress(min(done / total, 1.0))
            return b

        source_gains = None
        if per_source:
            # Level mic and system audio first, otherwise loud system audio
            # sets the peak and a quiet mic stays quiet in the mix
            source_gains = []
            for s in sources:
                s.seek(0)
                peak = max((np.abs(tick(b)).max() for b in s.blocks(65536, dtype='float32')), default=0)
                source_gains.append(1.0 / peak if peak > 0 else 1.0)

        blocks = lambda: self._mixed_blocks(sources, source_gains)
        if self.mono and channels > 1:
            channels = 1
            blocks = lambda: (b.mean(axis=1, keepdims=True) for b in self._mixed_blocks(sources, source_gains))

        gain = 1.0
        if self.normalize:
            peak = max((np.abs(tick(b)).max() for b in blocks()), default=0)
            if peak > 0:
                gain = 0.99 / peak

        if self.output_format == "mp3":
            encoder = lameenc.Encoder()
            encoder.set_bit_rate(192)
            encoder.set_in_sample_rate(samplerate)
            encoder.set_channels(channels)
            encoder.set_quality(2)
            with open(path, "wb") as f_mp3:
                for b in blocks():
                    tick(b, write_cost)
                    pcm = (b * gain * 32767).clip(-32768, 32767).astype(np.int16)
                    f_mp3.write(encoder.encode(pcm.tobytes()))
                f_mp3.write(encoder.flush())
        else:
            # WAV or FLAC, soundfile picks the container from the extension
            with sf.SoundFile(path, 'w', samplerate=samplerate, channels=channels) as f_out:
                for b in blocks():
                    f_out.write(tick(b, write_cost) * gain)

def get_devices(include_loopback=False):
    try:
        devices = sc.all_microphones(include_loopback=include_loopback)
        return [{"id": d.id, "name": d.name} for d in devices]
    except Exception as e:
        print(f"Error fetching devices: {e}")
        return []
