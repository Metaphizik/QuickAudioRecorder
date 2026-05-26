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

    def _probe_params(self):
        """Find working (samplerate, channels) for this device."""
        channels = self.channels
        if hasattr(self.device, 'channels'):
            channels = min(self.channels, self.device.channels)

        for sr in sorted({self.samplerate, 48000, 44100}):
            for ch in sorted({channels, 1}):
                try:
                    with self.device.recorder(samplerate=sr, channels=ch):
                        return sr, ch
                except Exception:
                    pass

        raise Exception(f"No working audio params found for device: {self.device.name}")

    def run(self):
        try:
            samplerate, channels = self._probe_params()
            with sf.SoundFile(self.filepath, mode='w', samplerate=samplerate, channels=channels) as f_wav:
                with self.device.recorder(samplerate=samplerate, channels=channels) as mic:
                    while not self.stop_event.is_set():
                        data = mic.record(numframes=2048)
                        f_wav.write(data)
        except Exception as e:
            self.error = str(e)

    def stop(self):
        self.stop_event.set()
        self.join()

class AudioRecorder(threading.Thread):
    """
    Orchestrates recording from Microphone, Loopback, or Both.
    """
    def __init__(self, mic_id, source_mode, output_folder, output_format="mp3", 
                 normalize=False, on_finish_callback=None):
        super().__init__()
        self.mic_id = mic_id
        self.source_mode = source_mode # "mic", "loopback", "both"
        self.output_folder = output_folder
        self.output_format = output_format.lower()
        self.normalize = normalize
        self.callback = on_finish_callback
        
        self.recording = False
        self.stop_event = threading.Event()
        self.error_message = None
        self.final_filepath = None
        
        # Temp files
        self.temp_files = []
        self.recorders = []

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
            return sc.get_microphone(self.mic_id, include_loopback=False)

    def run(self):
        self.recording = True
        self.error_message = None
        self.temp_files = []
        self.recorders = []
        
        try:
            # 1. Setup Recorders
            if self.source_mode == "both":
                # Need two recorders
                dev_mic = self._get_device(is_loopback=False)
                dev_loop = self._get_device(is_loopback=True)
                
                t1 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                t2 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                self.temp_files = [t1, t2]
                
                self.recorders.append(RawRecorder(dev_mic, t1))
                self.recorders.append(RawRecorder(dev_loop, t2))
                
            elif self.source_mode == "loopback":
                dev = self._get_device(is_loopback=True)
                t1 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                self.temp_files = [t1]
                self.recorders.append(RawRecorder(dev, t1))
                
            else: # mic
                dev = self._get_device(is_loopback=False)
                t1 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                self.temp_files = [t1]
                self.recorders.append(RawRecorder(dev, t1))

            print(f"Starting recording mode: {self.source_mode}")

            # 2. Start Recording
            for r in self.recorders:
                r.start()
            
            # Wait for stop signal
            self.stop_event.wait()
            
            # 3. Stop Recording
            for r in self.recorders:
                r.stop()
                if r.error:
                    raise Exception(f"Recorder error: {r.error}")

            # 4. Mix/Process
            if len(self.temp_files) == 2:
                mixed_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                self._mix_audio(self.temp_files[0], self.temp_files[1], mixed_wav)
                # Use mixed file as source for next steps
                source_wav = mixed_wav
                self.temp_files.append(mixed_wav) # Mark for cleanup
            else:
                source_wav = self.temp_files[0]

            # 5. Normalization
            if self.normalize:
                self._normalize_audio(source_wav)
            
            # 6. Finalize
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"Recording_{timestamp}.{self.output_format}"
            self.final_filepath = os.path.join(self.output_folder, filename)
            
            if self.output_format == "mp3":
                self._convert_to_mp3(source_wav, self.final_filepath)
            else:
                shutil.copy2(source_wav, self.final_filepath)

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
