import soundcard as sc
import soundfile as sf
import threading
import time
import os
import lameenc
import numpy as np
import tempfile
import shutil

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
        self.frames = 0

    def run(self):
        try:
            with sf.SoundFile(self.filepath, mode='w', samplerate=self.samplerate, channels=self.channels) as f_wav:
                try:
                    with self.device.recorder(samplerate=self.samplerate, channels=self.channels) as mic:
                        while not self.stop_event.is_set():
                            data = mic.record(numframes=2048)
                            f_wav.write(data)
                            self.frames += len(data)
                except Exception as e:
                    # Device was probably unplugged. Keep the frames written
                    # so far instead of failing the whole recording.
                    self.error = str(e)
        except Exception as e:
            self.error = str(e)

    def stop(self, timeout=5):
        self.stop_event.set()
        self.join(timeout=timeout)
        return not self.is_alive()

class AudioRecorder(threading.Thread):
    """
    Orchestrates recording from Microphone, Loopback, or Both.

    If a device disappears mid-recording:
      - reconnect=True  -> wait for the device, resume into a new segment,
        all segments are concatenated at the end;
      - reconnect=False -> finalize and save what was captured right away.
    A notification callback reports both events immediately.
    """
    def __init__(self, mic_id, source_mode, output_folder, output_format="mp3",
                 normalize=False, on_finish_callback=None,
                 reconnect=True, on_notify_callback=None):
        super().__init__()
        self.mic_id = mic_id
        self.source_mode = source_mode # "mic", "loopback", "both"
        self.output_folder = output_folder
        self.output_format = output_format.lower()
        self.normalize = normalize
        self.callback = on_finish_callback
        self.reconnect = reconnect
        self.notify_callback = on_notify_callback

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
            # For loopback, we try to find the default speaker's loopback
            default_speaker = sc.default_speaker()
            mics = sc.all_microphones(include_loopback=True)
            # Try exact name match
            loopback_mic = next((m for m in mics if m.name == default_speaker.name), None)
            # Try fuzzy match
            if not loopback_mic:
                loopback_mic = next((m for m in mics if default_speaker.name in m.name), None)

            if not loopback_mic:
                raise Exception("Could not detect System Audio loopback device.")
            return loopback_mic
        else:
            return self._get_mic_device()

    def _get_mic_device(self):
        # First try the saved id (stable while the device stays on the same
        # port); if the device was replugged with a new id, match by name.
        try:
            dev = sc.get_microphone(self.mic_id, include_loopback=False)
            if dev is not None:
                return dev
        except Exception:
            pass
        if self.mic_name:
            try:
                mics = sc.all_microphones(include_loopback=False)
                dev = next((m for m in mics if m.name == self.mic_name), None)
                if dev is not None:
                    return dev
            except Exception:
                pass
        raise Exception("Microphone not found.")

    def _notify(self, message):
        if self.notify_callback:
            try:
                self.notify_callback(message)
            except Exception:
                pass
        print(message)

    def _start_recorder(self, src, device):
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        self.temp_files.append(path)
        src["files"].append(path)
        if not src["is_loop"]:
            self.mic_name = device.name
        rec = RawRecorder(device, path)
        src["rec"] = rec
        rec.start()

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
                src = {"is_loop": is_loop, "files": [], "rec": None, "failed_starts": 0}
                self._start_recorder(src, dev)
                self.sources.append(src)

            print(f"Starting recording mode: {self.source_mode}")

            # 2. Wait for the stop signal, watching for device losses
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

                    # Count instant failures (device "present" but dies
                    # right away) to avoid an endless reconnect loop.
                    if rec.frames == 0:
                        src["failed_starts"] += 1
                    else:
                        src["failed_starts"] = 0

                    if src["failed_starts"] >= 5:
                        self._notify(f"{label} keeps failing - saving recording...")
                        self.stop_event.set()
                        break

                    if not self.reconnect:
                        # Finalize immediately: keep what was recorded.
                        self._notify(f"{label} disconnected - saving recording...")
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
                        break

            # 3. Stop Recording
            for src in self.sources:
                rec = src["rec"]
                if rec is None:
                    continue
                stopped = rec.stop()
                if not stopped and rec.error is None:
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
                raise Exception(f"Recorder error: {reason}")

            # 4. Concatenate segments per source (reconnect creates segments)
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
                    self._concat_audio(wavs, combined)
                    self.temp_files.append(combined)
                    source_wavs.append(combined)

            if not source_wavs:
                raise Exception("Recorder error: no audio captured")

            # 5. Mix (if more than one source produced audio)
            if len(source_wavs) == 2:
                mixed_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                self._mix_audio(source_wavs[0], source_wavs[1], mixed_wav)
                # Use mixed file as source for next steps
                source_wav = mixed_wav
                self.temp_files.append(mixed_wav) # Mark for cleanup
            else:
                source_wav = source_wavs[0]

            # 6. Normalization
            if self.normalize:
                self._normalize_audio(source_wav)

            # 7. Finalize
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"Recording_{timestamp}.{self.output_format}"
            self.final_filepath = os.path.join(self.output_folder, filename)

            if self.output_format == "mp3":
                self._convert_to_mp3(source_wav, self.final_filepath)
            else:
                shutil.copy2(source_wav, self.final_filepath)

            # Saved successfully, but note the device loss for the UI.
            if recorder_errors:
                self.error_message = f"Device disconnected, partial recording saved: {recorder_errors[0]}"

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

    def _concat_audio(self, files, out_file):
        combined = None
        sr = 44100
        for fp in files:
            try:
                data, s = sf.read(fp)
            except Exception as e:
                print(f"Skipping unreadable segment {fp}: {e}")
                continue
            sr = s
            if combined is None:
                combined = data
            elif data.ndim == combined.ndim and data.shape[1:] == combined.shape[1:]:
                combined = np.concatenate((combined, data))
        if combined is not None:
            sf.write(out_file, combined, sr)

    def _mix_audio(self, file1, file2, out_file):
        d1, sr1 = sf.read(file1)
        d2, sr2 = sf.read(file2)

        # Ensure same length
        max_len = max(len(d1), len(d2))

        # Pad d1
        if len(d1) < max_len:
            pad_width = max_len - len(d1)
            # handle mono/stereo padding
            shape = (pad_width, d1.shape[1]) if d1.ndim > 1 else (pad_width,)
            d1 = np.concatenate((d1, np.zeros(shape, dtype=d1.dtype)))

        # Pad d2
        if len(d2) < max_len:
            pad_width = max_len - len(d2)
            shape = (pad_width, d2.shape[1]) if d2.ndim > 1 else (pad_width,)
            d2 = np.concatenate((d2, np.zeros(shape, dtype=d2.dtype)))

        # Mix (Sum)
        mixed = d1 + d2
        # Clip
        mixed = np.clip(mixed, -1.0, 1.0)

        sf.write(out_file, mixed, sr1) # Assume sr1 == sr2 = 44100

    def _normalize_audio(self, filepath):
        try:
            data, sr = sf.read(filepath)
            max_val = np.max(np.abs(data))
            if max_val > 0:
                target_peak = 0.99
                factor = target_peak / max_val
                data = data * factor
                sf.write(filepath, data, sr)
        except Exception as e:
            print(f"Normalization failed: {e}")

    def _convert_to_mp3(self, src_wav, dst_mp3):
        data, sr = sf.read(src_wav)
        channels = data.shape[1] if data.ndim > 1 else 1

        pcm_data = (data * 32767).clip(-32768, 32767).astype(np.int16)

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(192)
        encoder.set_in_sample_rate(sr)
        encoder.set_channels(channels)
        encoder.set_quality(2)

        mp3_data = encoder.encode(pcm_data.tobytes())
        mp3_data += encoder.flush()

        with open(dst_mp3, "wb") as f_mp3:
            f_mp3.write(mp3_data)

def get_devices(include_loopback=False):
    try:
        devices = sc.all_microphones(include_loopback=include_loopback)
        return [{"id": d.id, "name": d.name} for d in devices]
    except Exception as e:
        print(f"Error fetching devices: {e}")
        return []
