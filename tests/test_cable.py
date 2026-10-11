"""
Real-device tests over VB-Audio Virtual Cable (https://vb-audio.com/Cable/).

A tone is played into "CABLE Input" and recorded back through "CABLE Output"
(as a microphone) or the loopback of "CABLE Input" (as system audio). Your
default speakers and microphone are never touched. Skipped without the cable.
"""
import threading
import time

import numpy as np
import pytest
import soundcard as sc
import soundfile as sf

import audio_recorder
from conftest import dominant_freq, record


def _find(devices, name):
    return next((d for d in devices if name in d.name), None)


try:
    cable_in = _find(sc.all_speakers(), "CABLE Input")
    cable_out = _find(sc.all_microphones(), "CABLE Output")
except Exception:  # no audio stack at all, e.g. CI runners
    cable_in = cable_out = None

pytestmark = pytest.mark.skipif(not (cable_in and cable_out), reason="VB-Audio Virtual Cable not installed")


@pytest.fixture
def cable(monkeypatch):
    """Points the recorder at the cable and plays a 440 Hz tone into it."""
    real_sc = audio_recorder.sc
    monkeypatch.setattr(real_sc, "default_speaker", lambda: cable_in)
    monkeypatch.setattr(real_sc, "get_microphone",
                        lambda id, include_loopback=False: cable_out)

    stop = threading.Event()

    def play():
        sr = 48000
        t = np.arange(sr // 10) / sr
        block = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        with cable_in.player(samplerate=sr, channels=1) as p:
            while not stop.is_set():
                p.play(block)

    player = threading.Thread(target=play, daemon=True)
    player.start()
    time.sleep(0.3)
    yield
    stop.set()
    player.join(5)


@pytest.mark.parametrize("mode", ["mic", "loopback", "both"])
def test_records_tone_over_cable(cable, tmp_path, mode):
    path, error = record(tmp_path, mode, seconds=2, output_format="wav")
    assert not error
    data, sr = sf.read(path)
    assert len(data) > sr  # at least a second of audio
    assert abs(dominant_freq(data, sr) - 440) < 5


def test_falls_back_to_portaudio_for_mics_soundcard_rejects(cable, tmp_path, monkeypatch):
    # what soundcard does for mics with a non-float mix format (#11)
    def broken_recorder(*args, **kwargs):
        raise AssertionError()
    monkeypatch.setattr(cable_out, "recorder", broken_recorder)

    path, error = record(tmp_path, "mic", seconds=2, output_format="wav")
    assert not error
    data, sr = sf.read(path)
    assert abs(dominant_freq(data, sr) - 440) < 5
