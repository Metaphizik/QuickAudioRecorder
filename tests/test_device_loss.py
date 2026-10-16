"""A device that disappears in the middle of a recording."""

import os

import numpy as np
import soundfile as sf

from conftest import dominant_freq, record


def test_unplug_keeps_the_partial_recording(fake_sc, tmp_path):
    fake_sc.mic.die_after = 44100  # dies after a fraction of a second
    path, error = record(tmp_path, "mic", seconds=2.0, output_format="wav",
                         reconnect=False)

    assert os.path.exists(path), "the audio captured before the unplug is gone"
    data, sr = sf.read(path)
    assert len(data) > 0
    assert abs(dominant_freq(data, sr) - 440) < 5
    assert "disconnected" in error.lower()
    assert "partial recording saved" in error.lower()


def test_reconnect_resumes_and_glues_segments(fake_sc, tmp_path):
    fake_sc.mic.die_after = 44100
    notes = []
    path, error = record(tmp_path, "mic", seconds=3.0, output_format="wav",
                         reconnect=True, on_notify_callback=notes.append)

    assert not error
    assert any("waiting for reconnection" in n for n in notes), notes
    assert any("reconnected" in n for n in notes), notes
    data, sr = sf.read(path)
    # longer than the first segment alone, so the resumed part is in there
    assert len(data) > 200000
    assert abs(dominant_freq(data, sr) - 440) < 5


def test_reconnect_finds_the_mic_by_name_after_replug(fake_sc, tmp_path, monkeypatch):
    # the saved device id is gone after a replug, only the name matches
    real_get = fake_sc.get_microphone
    calls = {"n": 0}

    def get_by_changing_id(device_id, include_loopback=False):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("no such device")
        return real_get(device_id, include_loopback=include_loopback)

    fake_sc.mic.die_after = 44100
    monkeypatch.setattr(fake_sc, "get_microphone", get_by_changing_id)

    notes = []
    path, error = record(tmp_path, "mic", seconds=3.0, output_format="wav",
                         on_notify_callback=notes.append)
    assert any("reconnected" in n for n in notes), notes
    assert not error


def test_device_that_never_comes_back_saves_what_it_has(fake_sc, tmp_path):
    fake_sc.mic.die_after = 44100

    def unplug_for_good(message):
        if "waiting for reconnection" in message:
            fake_sc.mic.gone = True  # nothing to reconnect to

    path, error = record(tmp_path, "mic", seconds=2.5, output_format="wav",
                         reconnect=True, on_notify_callback=unplug_for_good)

    assert os.path.exists(path)
    assert "partial recording saved" in error.lower()
    data, sr = sf.read(path)
    assert len(data) > 0


def test_unplug_while_recording_both_keeps_the_other_track(fake_sc, tmp_path):
    fake_sc.mic.die_after = 44100

    def unplug_mic_for_good(message):
        if message.startswith("Microphone") and "waiting for reconnection" in message:
            fake_sc.mic.gone = True

    path, error = record(tmp_path, "both", seconds=2.5, output_format="wav",
                         reconnect=False, on_notify_callback=unplug_mic_for_good)

    assert os.path.exists(path)
    data, sr = sf.read(path, always_2d=True)
    # the system audio kept running, so its 1000 Hz tone is still there
    spectrum = abs(np.fft.rfft(data.mean(axis=1)))
    freqs = np.fft.rfftfreq(len(data), 1 / sr)
    assert spectrum[abs(freqs - 1000) < 3].max() > spectrum.max() * 0.3
