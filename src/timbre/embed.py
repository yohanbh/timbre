"""CLAP embedding with bit-reproducible output.

Byte-identical restarts require fixing *batch composition*, not just batch size:
the same window embedded inside a differently-sized batch yields different low
bits (measured maxdiff ~5e-08), because batched GEMM kernels pick different tile
decompositions. So one track's 21 windows are always embedded as one batch.
"""
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

MODEL_ID = "laion/larger_clap_general"
MODEL_REVISION = "ada0c23a36c4e8582805bb38fec3905903f18b41"
SR = 48000
WINDOW_SAMPLES = 480000  # exactly 10 s @ 48 kHz -- see assert_window_len
HOP_SECONDS = 1
WINDOWS_PER_TRACK = 21
DIM = 512


def assert_window_len(window):
    """Guard the rand_trunc landmine.

    preprocessor_config.json ships "truncation": "rand_trunc", and the random-crop
    branch in feature_extraction_clap.py fires on len > 480000 (strictly greater),
    calling an unseeded np.random.randint. Exactly 480000 stays deterministic.
    """
    if len(window) != WINDOW_SAMPLES:
        raise ValueError(
            f"window must be exactly {WINDOW_SAMPLES} samples, got {len(window)}; "
            "off-by-one here silently triggers unseeded rand_trunc cropping"
        )


def set_determinism():
    """Pin every knob that changes output bits. Call before loading the model."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)


class Embedder:
    def __init__(self, device="cuda"):
        set_determinism()
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing to fall back to CPU (would change bits)")
        self.device = device
        # Load the weights at the pinned revision, not an automatic safetensors PR.
        self.model = ClapModel.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, use_safetensors=False
        ).eval().to(device)
        self.processor = ClapProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    def features(self, windows):
        """Extract mel features on CPU. Runs in worker processes."""
        for w in windows:
            assert_window_len(w)
        return self.processor(audios=list(windows), sampling_rate=SR, return_tensors="pt")

    def embed_features(self, feats):
        """Embed pre-extracted features on GPU. One call per track."""
        feats = {k: v.to(self.device) for k, v in feats.items()}
        with torch.inference_mode():
            out = self.model.get_audio_features(**feats)
        return out.cpu().numpy().astype(np.float32)

    def embed(self, windows):
        return self.embed_features(self.features(windows))

    def embed_text(self, texts):
        feats = self.processor(text=list(texts), padding=True, truncation=True,
                               return_tensors="pt")
        with torch.inference_mode():
            out = self.model.get_text_features(
                **{k: v.to(self.device) for k, v in feats.items()}
            )
        return out.cpu().numpy().astype(np.float32)

    def check_text_separation(self):
        """Catch the collapsed music checkpoint before starting a long ingest.

        This is a collapse smoke check, not a perceptual-quality benchmark.
        The old checkpoint scored >0.999 for every unrelated pair below;
        the validated general checkpoint scores between 0.069 and 0.441.
        """
        vecs = self.embed_text([
            "a heavy metal guitar solo", "a slow sad piano ballad",
            "a female opera singer", "fast electronic dance music",
        ])
        sims = vecs @ vecs.T
        off = sims[~np.eye(len(sims), dtype=bool)]
        if not np.isfinite(off).all() or off.mean() > 0.99:
            raise RuntimeError("CLAP text embeddings collapsed; refusing to ingest")
        return float(off.mean())

    def check_dim(self):
        """Verify embedding width with one forward pass before sizing the store."""
        probe = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        out = self.embed([probe])
        if out.shape != (1, DIM):
            raise RuntimeError(f"expected (1, {DIM}), got {out.shape}")
        return out.shape


def versions():
    import transformers
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
