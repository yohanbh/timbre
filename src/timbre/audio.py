"""Decode mp3 to 48 kHz mono float32 and slice into fixed-length windows."""
import subprocess

import numpy as np

from .embed import HOP_SECONDS, SR, WINDOW_SAMPLES, WINDOWS_PER_TRACK


class DecodeError(Exception):
    pass


def decode(path):
    """mp3 -> 48 kHz mono float32. Raises DecodeError on corrupt input."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(SR), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise DecodeError(proc.stderr.decode("utf-8", "replace")[:200])
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise DecodeError("decoded to zero samples")
    return audio


def windows(audio):
    """Slice into up to 21 windows of exactly WINDOW_SAMPLES at a 1 s hop.

    Short tracks yield fewer than 21 windows; the caller zero-fills the rest of
    the reserved block. A partial trailing window is dropped rather than padded --
    padding would put an artifact vector into the index.
    """
    hop = SR * HOP_SECONDS
    out = []
    for i in range(WINDOWS_PER_TRACK):
        start = i * hop
        end = start + WINDOW_SAMPLES
        if end > audio.size:
            break
        out.append(np.ascontiguousarray(audio[start:end]))
    return out
