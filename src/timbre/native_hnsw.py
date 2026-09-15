"""Hand-written C++ construction and immutable HNSW search snapshots.

Build the extension with ``python3 -m pip install --no-deps -e .``.
The vector matrix is shared when C-contiguous; callers must keep it immutable.
Graph links and IDs are copied, so later Python insertion cannot stale them.
"""
import json
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .hnsw import HNSW


DIRECT_FORMAT = 1


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_direct(path, metadata, arrays):
    """Atomically write mmap-friendly arrays and their completion manifest."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"direct graph directory already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        manifest = {}
        for name, array in arrays.items():
            target = temporary / f"{name}.npy"
            with target.open("wb") as output:
                np.save(output, array, allow_pickle=False)
                output.flush()
                os.fsync(output.fileno())
            manifest[target.name] = {
                "dtype": np.dtype(array.dtype).str,
                "shape": list(array.shape),
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        metadata = {**metadata, "direct_format": DIRECT_FORMAT, "arrays": manifest}
        metadata_path = temporary / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        with metadata_path.open("rb") as source:
            os.fsync(source.fileno())
        complete = {"metadata_sha256": _sha256(metadata_path)}
        complete_path = temporary / "COMPLETE"
        complete_path.write_text(json.dumps(complete, sort_keys=True) + "\n")
        with complete_path.open("rb") as source:
            os.fsync(source.fileno())
        os.replace(temporary, path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_array(path, name, manifest, verify):
    filename = f"{name}.npy"
    expected = manifest.get(filename)
    if expected is None:
        raise ValueError(f"direct graph is missing {filename} metadata")
    target = path / filename
    if not target.is_file() or target.stat().st_size != expected["bytes"]:
        raise ValueError(f"direct graph has an incomplete {filename}")
    if verify and _sha256(target) != expected["sha256"]:
        raise ValueError(f"direct graph checksum failed for {filename}")
    array = np.load(target, mmap_mode="r", allow_pickle=False)
    if np.dtype(array.dtype).str != expected["dtype"] or list(array.shape) != expected["shape"]:
        raise ValueError(f"direct graph manifest does not match {filename}")
    return array


class NativeHNSW:
    def __init__(self, index):
        index.validate()
        layers = []
        for layer in range(index.max_level + 1):
            nodes = np.flatnonzero(index.levels >= layer).astype(np.int32)
            rows = [index._neighbors(int(node), layer) for node in nodes]
            offsets = np.r_[0, np.cumsum([len(row) for row in rows], dtype=np.int64)]
            links = np.array([n for row in rows for n in row], dtype=np.int32)
            layers.append((nodes, offsets, links))
        self._initialize(index, layers)

    def _initialize(self, index, layers):
        from ._hnsw_native import SearchIndex

        self.vectors = np.ascontiguousarray(index.vectors)
        self.ids = index.ids.copy()
        self.ids.flags.writeable = False
        self.levels = np.asarray(index.levels, dtype=np.int16).copy()
        self.levels.flags.writeable = False
        self.M, self.ef_construction, self.size = index.M, index.ef_construction, index.size
        self.entrypoint, self.max_level = index.entrypoint, index.max_level
        self.construction_backend = getattr(index, "BACKEND", None)
        self._layers = layers
        self._index = SearchIndex(self.vectors, self.ids, index.entrypoint, layers)
        self.distance_evaluations = 0

    @classmethod
    def load(cls, path, vectors):
        # Reuse Phase 2's fingerprint and graph validation before conversion.
        return cls(HNSW.load(path, vectors))

    @classmethod
    def load_directory(cls, path, verify=False):
        """Map a completed direct graph without expanding Python adjacency."""
        from ._hnsw_native import SearchIndex

        path = Path(path)
        metadata_path, complete_path = path / "metadata.json", path / "COMPLETE"
        if not metadata_path.is_file() or not complete_path.is_file():
            raise ValueError("direct graph directory is incomplete")
        metadata = json.loads(metadata_path.read_text())
        complete = json.loads(complete_path.read_text())
        if (metadata.get("direct_format") != DIRECT_FORMAT or
                complete.get("metadata_sha256") != _sha256(metadata_path)):
            raise ValueError("invalid direct graph metadata or completion marker")
        manifest = metadata.get("arrays", {})
        vectors = _load_array(path, "vectors", manifest, verify)
        ids = _load_array(path, "ids", manifest, verify)
        levels = _load_array(path, "levels", manifest, verify)
        if vectors.dtype != np.float32 or ids.dtype != np.int64 or levels.dtype != np.int16:
            raise ValueError("invalid direct graph array dtypes")
        if ids.shape != (len(vectors),) or levels.shape != (len(vectors),):
            raise ValueError("invalid direct graph vector, ID or level shapes")
        size = int(metadata["size"])
        max_level = int(metadata["max_level"])
        if (np.count_nonzero(levels >= 0) != size or
                (int(levels.max()) if size else -1) != max_level):
            raise ValueError("direct graph levels do not match metadata")
        layers = []
        for layer in range(max_level + 1):
            nodes = _load_array(path, f"nodes_{layer}", manifest, verify)
            offsets = _load_array(path, f"offsets_{layer}", manifest, verify)
            links = _load_array(path, f"links_{layer}", manifest, verify)
            if nodes.dtype != np.int32 or offsets.dtype != np.int64 or links.dtype != np.int32:
                raise ValueError("invalid direct graph layer dtypes")
            expected_nodes = np.flatnonzero(levels >= layer)
            if not np.array_equal(nodes, expected_nodes):
                raise ValueError("direct graph layer membership does not match levels")
            capacity = 2 * int(metadata["M"]) if layer == 0 else int(metadata["M"])
            if len(offsets) != len(nodes) + 1 or np.any(np.diff(offsets) > capacity):
                raise ValueError("direct graph degree limit exceeded")
            layers.append((nodes, offsets, links))
        result = cls.__new__(cls)
        result.vectors, result.ids, result.levels = vectors, ids, levels
        result.M = int(metadata["M"])
        result.ef_construction = int(metadata["ef_construction"])
        result.size, result.max_level = size, max_level
        result.entrypoint = int(metadata["entrypoint"])
        result.construction_backend = metadata.get("construction_backend")
        result._layers = layers
        result._index = SearchIndex(vectors, ids, result.entrypoint, layers)
        result.distance_evaluations = 0
        return result

    def save_directory(self, path):
        """Persist an immutable snapshot as independently mappable .npy files."""
        metadata = {
            "M": self.M,
            "ef_construction": self.ef_construction,
            "entrypoint": self.entrypoint,
            "max_level": self.max_level,
            "size": self.size,
            "vector_hash": hashlib.sha256(self.vectors.view(np.uint8)).hexdigest(),
            "construction_backend": self.construction_backend,
        }
        arrays = {"vectors": self.vectors, "ids": self.ids, "levels": self.levels}
        for layer, (nodes, offsets, links) in enumerate(self._layers):
            arrays[f"nodes_{layer}"] = nodes
            arrays[f"offsets_{layer}"] = offsets
            arrays[f"links_{layer}"] = links
        _write_direct(path, metadata, arrays)

    @classmethod
    def convert(cls, checkpoint, vectors, path):
        """Convert a legacy NPZ graph without constructing Python adjacency."""
        with np.load(checkpoint, allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays["metadata"]))
            if metadata.get("format") != 1:
                raise ValueError("unsupported HNSW file format")
            vector_hash = hashlib.sha256(np.ascontiguousarray(vectors).view(np.uint8)).hexdigest()
            if vector_hash != metadata["vector_hash"]:
                raise ValueError("graph does not match the supplied vectors")
            levels = arrays["levels"].astype(np.int16, copy=True)
            layers = [
                (arrays[f"nodes_{layer}"].astype(np.int32, copy=True),
                 arrays[f"offsets_{layer}"].astype(np.int64, copy=True),
                 arrays[f"links_{layer}"].astype(np.int32, copy=True))
                for layer in range(int(metadata["max_level"]) + 1)
            ]
            result = cls.__new__(cls)
            result.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            result.ids = arrays["ids"].astype(np.int64, copy=True)
            result.levels = levels
            result.M, result.ef_construction = metadata["M"], metadata["ef_construction"]
            result.entrypoint, result.max_level = metadata["entrypoint"], metadata["max_level"]
            result.size = metadata["size"]
            result.construction_backend = metadata.get("construction_backend")
            result._layers = layers
            from ._hnsw_native import SearchIndex
            result._index = SearchIndex(result.vectors, result.ids, result.entrypoint, layers)
            result.distance_evaluations = 0
            result.save_directory(path)
        return cls.load_directory(path)

    def search(self, query, k=10, ef_search=50):
        """Return external IDs and cosine scores; normalize as in Python HNSW."""
        if k < 1 or ef_search < k:
            raise ValueError("require ef_search >= k >= 1")
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.vectors.shape[1],) or not np.isfinite(query).all():
            raise ValueError("query must be finite and match the vector dimension")
        norm = np.linalg.norm(query)
        if norm == 0:
            raise ValueError("query must be nonzero")
        query = query / norm
        ids, scores, self.distance_evaluations = self._index.search(query, k, ef_search)
        return ids, scores


class NativeHNSWBuilder:
    """Serial C++ insertion; NumPy supplies reproducible exponential levels.

    ``build`` reports progress after batches of up to 1,000 insertions. Save from
    that callback for resumable construction; use ``freeze`` for query snapshots.
    Resume must keep the remaining insertion order and the same native kernel.
    """
    BACKEND = "cpp-eight-lane-v1"

    def __init__(self, vectors, ids=None, M=16, ef_construction=100, seed=0):
        from ._hnsw_native import Builder

        # Preserve HNSW's validation without allocating Python adjacency.
        self.vectors = np.ascontiguousarray(HNSW._validated_vectors(vectors))
        self.ids = HNSW._validated_ids(ids, len(self.vectors)).copy()
        self.ids.flags.writeable = False
        self.M, self.ef_construction = HNSW._validated_configuration(M, ef_construction)
        self.rng = np.random.default_rng(seed)
        self._vector_sha256 = None
        self._builder = Builder(self.vectors, self.M, self.ef_construction)

    @property
    def levels(self):
        return self._builder.levels

    @property
    def size(self):
        return self._builder.size

    @property
    def entrypoint(self):
        return self._builder.entrypoint

    @property
    def max_level(self):
        return self._builder.max_level

    def insert(self, node):
        self.build([node])

    def build(self, order=None, progress=None):
        levels = self.levels
        order = np.flatnonzero(levels < 0) if order is None else np.asarray(order, dtype=np.int64)
        if (order.ndim != 1 or np.any(order < 0) or np.any(order >= len(levels)) or
                np.unique(order).size != len(order) or np.any(levels[order] >= 0)):
            raise ValueError("order must contain unique uninserted vector rows")
        for start in range(0, len(order), 1000):
            batch = np.ascontiguousarray(order[start:start + 1000], dtype=np.int32)
            assigned = (-np.log(1.0 - self.rng.random(len(batch))) / np.log(self.M)).astype(np.int16)
            self._builder.build(batch, assigned)
            if progress is not None:
                progress(self.size)
        return self

    def validate(self):
        self._builder.validate()

    def freeze(self):
        """Copy adjacency into an immutable native search index."""
        self.validate()
        arrays = self._builder.export_graph()
        layers = [(arrays[f"nodes_{layer}"].astype(np.int32), arrays[f"offsets_{layer}"],
                   arrays[f"links_{layer}"]) for layer in range(self.max_level + 1)]
        snapshot = NativeHNSW.__new__(NativeHNSW)
        snapshot._initialize(self, layers)
        return snapshot

    def save_directory(self, path):
        self.freeze().save_directory(path)

    def save(self, path):
        """Atomic Phase 2-compatible checkpoint, including RNG/backend identity."""
        self.validate()
        if self._vector_sha256 is None:
            self._vector_sha256 = HNSW._vector_hash(self)
        metadata = {"format": 1, "M": self.M, "ef_construction": self.ef_construction,
                    "entrypoint": self.entrypoint, "max_level": self.max_level, "size": self.size,
                    "rng_state": self.rng.bit_generator.state, "vector_hash": self._vector_sha256,
                    "construction_backend": self.BACKEND}
        arrays = {"metadata": json.dumps(metadata), "ids": self.ids, **self._builder.export_graph()}
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as output:
            np.savez(output, **arrays)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)

    @classmethod
    def load(cls, path, vectors):
        with np.load(path, allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays["metadata"]))
        if metadata.get("construction_backend") != cls.BACKEND:
            raise ValueError("checkpoint construction backend differs; cannot resume natively")
        restored = HNSW.load(path, vectors)
        builder = cls(restored.vectors, restored.ids, restored.M, restored.ef_construction)
        graph = [[restored._neighbors(node, layer) for layer in range(int(level) + 1)]
                 for node, level in enumerate(restored.levels)]
        builder._builder.restore(graph, restored.entrypoint)
        builder._vector_sha256 = metadata["vector_hash"]
        builder.rng.bit_generator.state = restored.rng.bit_generator.state
        return builder
