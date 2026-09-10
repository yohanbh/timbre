"""HNSW for immutable, unit-normalized vectors, written in Python and NumPy.

Implements insertion, beam search and neighbor diversification from Malkov &
Yashunin, https://arxiv.org/abs/1603.09320 (algorithms 1, 2, 4 and 5).
New nodes establish M reciprocal links; overflow pruning caps outgoing degree
at 2*M on layer zero and M above it. Pruning can make links asymmetric.
"""
import hashlib
import heapq
import json
import os
from pathlib import Path

import numpy as np


class HNSW:
    def __init__(self, vectors, ids=None, M=16, ef_construction=100, seed=0):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors.ndim != 2 or self.vectors.shape[1] == 0:
            raise ValueError("vectors must be a matrix with a nonzero dimension")
        if not np.isfinite(self.vectors).all() or not np.allclose(
            np.linalg.norm(self.vectors, axis=1), 1, atol=1e-4
        ):
            raise ValueError("vectors must be finite and L2-normalized")
        n = len(self.vectors)
        self.ids = np.arange(n, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        if self.ids.shape != (n,) or np.unique(self.ids).size != n:
            raise ValueError("ids must contain one unique integer per vector")
        if M < 2 or ef_construction < M:
            raise ValueError("require M >= 2 and ef_construction >= M")
        self.M, self.ef_construction = int(M), int(ef_construction)
        self.rng = np.random.default_rng(seed)
        self.levels = np.full(n, -1, dtype=np.int16)
        self.base = [[] for _ in range(n)]
        self.upper = []  # layer l lives at upper[l - 1]: node -> neighbors
        self.entrypoint, self.max_level, self.size = -1, -1, 0
        self.distance_evaluations = 0

    def _neighbors(self, node, layer):
        return self.base[node] if layer == 0 else self.upper[layer - 1][node]

    def _set_neighbors(self, node, layer, neighbors):
        if layer == 0:
            self.base[node] = neighbors
        else:
            self.upper[layer - 1][node] = neighbors

    def _distances(self, query, nodes):
        self.distance_evaluations += len(nodes)
        return 1.0 - self.vectors[nodes] @ query

    def _greedy(self, query, entry, layer):
        distance = float(self._distances(query, [entry])[0])
        while self._neighbors(entry, layer):
            neighbors = self._neighbors(entry, layer)
            distances = self._distances(query, neighbors)
            best = int(np.argmin(distances))
            if distances[best] >= distance:
                break
            entry, distance = neighbors[best], float(distances[best])
        return entry

    def _search_layer(self, query, entries, ef, layer):
        """Nearest-first exploration plus a bounded farthest-first result heap."""
        visited = set(entries)
        initial = sorted(zip(self._distances(query, entries).tolist(), entries))[:ef]
        candidates = initial.copy()
        heapq.heapify(candidates)
        best = [(-distance, -node) for distance, node in initial]
        heapq.heapify(best)
        while candidates:
            distance, node = heapq.heappop(candidates)
            if len(best) >= ef and distance > -best[0][0]:
                break
            unseen = [n for n in self._neighbors(node, layer) if n not in visited]
            if not unseen:
                continue
            visited.update(unseen)
            for d, neighbor in zip(self._distances(query, unseen).tolist(), unseen):
                if len(best) < ef or (d, neighbor) < (-best[0][0], -best[0][1]):
                    heapq.heappush(candidates, (d, neighbor))
                    if len(best) < ef:
                        heapq.heappush(best, (-d, -neighbor))
                    else:
                        heapq.heapreplace(best, (-d, -neighbor))
        return sorted((-d, -n) for d, n in best)

    def _select_neighbors(self, candidates, limit):
        """Keep a candidate unless it is closer to an already selected neighbor.

        Rejecting redundant directions preserves routes between clusters. The
        comparisons against each accepted neighbor are vectorized over the
        remaining candidates. No candidate extension or pruned-link refill.
        """
        candidates = sorted(candidates)
        if len(candidates) <= limit:
            return [node for _, node in candidates]
        distances = np.array([d for d, _ in candidates], dtype=np.float32)
        nodes = [n for _, n in candidates]
        vectors = self.vectors[nodes]
        rejected = np.zeros(len(nodes), dtype=bool)
        selected = []
        for i, node in enumerate(nodes):
            if rejected[i]:
                continue
            selected.append(node)
            if len(selected) == limit:
                break
            between = 1.0 - vectors[i + 1:] @ vectors[i]
            rejected[i + 1:] |= between < distances[i + 1:]
        return selected

    def insert(self, node):
        """Insert one internal row; vectors and external IDs stay immutable."""
        node = int(node)
        if node < 0 or node >= len(self.vectors) or self.levels[node] >= 0:
            raise ValueError("node must be an uninserted vector row")
        level = int(-np.log(1.0 - self.rng.random()) / np.log(self.M))
        self.levels[node] = level
        while len(self.upper) < level:
            self.upper.append({})
        for layer in range(1, level + 1):
            self.upper[layer - 1][node] = []
        self.size += 1
        if self.entrypoint == -1:
            self.entrypoint, self.max_level = node, level
            return

        query = self.vectors[node]
        entry = self.entrypoint
        for layer in range(self.max_level, level, -1):
            entry = self._greedy(query, entry, layer)
        entries = [entry]
        for layer in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(query, entries, self.ef_construction, layer)
            selected = self._select_neighbors(candidates, self.M)
            self._set_neighbors(node, layer, selected)
            capacity = 2 * self.M if layer == 0 else self.M
            for neighbor in selected:
                links = self._neighbors(neighbor, layer)
                links.append(node)
                if len(links) > capacity:
                    distances = self._distances(self.vectors[neighbor], links)
                    kept = self._select_neighbors(zip(distances.tolist(), links), capacity)
                    self._set_neighbors(neighbor, layer, kept)
            entries = [n for _, n in candidates]
        if level > self.max_level:
            self.entrypoint, self.max_level = node, level

    def build(self, order=None, progress=None):
        if order is None:
            order = np.flatnonzero(self.levels < 0)
        for node in order:
            self.insert(node)
            if progress is not None:
                progress(self.size)
        return self

    def search(self, query, k=10, ef_search=50):
        """Return external IDs and cosine scores. ef_search must be at least k."""
        if k < 1 or ef_search < k:
            raise ValueError("require ef_search >= k >= 1")
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.vectors.shape[1],) or not np.isfinite(query).all():
            raise ValueError("query must be finite and match the vector dimension")
        norm = np.linalg.norm(query)
        if norm == 0:
            raise ValueError("query must be nonzero")
        query = query / norm
        self.distance_evaluations = 0
        if not self.size:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        entry = self.entrypoint
        for layer in range(self.max_level, 0, -1):
            entry = self._greedy(query, entry, layer)
        found = self._search_layer(query, [entry], ef_search, 0)
        found.sort(key=lambda pair: (pair[0], self.ids[pair[1]]))
        nodes = [node for _, node in found[:k]]
        scores = np.array([1.0 - distance for distance, _ in found[:k]], dtype=np.float32)
        return self.ids[nodes], scores

    def validate(self):
        """Check layer membership, degree bounds, links and the entry point."""
        inserted = np.flatnonzero(self.levels >= 0)
        if len(inserted) != self.size:
            raise ValueError("size does not match inserted nodes")
        expected_max = int(self.levels.max()) if self.size else -1
        if self.max_level != expected_max or len(self.upper) != max(0, expected_max):
            raise ValueError("invalid layer hierarchy")
        if self.size and (self.entrypoint not in inserted or
                          self.levels[self.entrypoint] != self.max_level):
            raise ValueError("invalid entry point")
        for layer in range(self.max_level + 1):
            nodes = np.flatnonzero(self.levels >= layer)
            if layer and set(self.upper[layer - 1]) != set(nodes):
                raise ValueError("invalid upper-layer membership")
            for node in nodes:
                links = self._neighbors(node, layer)
                if len(links) > (2 * self.M if layer == 0 else self.M):
                    raise ValueError("degree limit exceeded")
                if len(set(links)) != len(links) or node in links:
                    raise ValueError("duplicate or self link")
                if any(n < 0 or n >= len(self.levels) or self.levels[n] < layer for n in links):
                    raise ValueError("link to a missing node or layer")

    def _vector_hash(self):
        return hashlib.sha256(np.ascontiguousarray(self.vectors).view(np.uint8)).hexdigest()

    def save(self, path):
        """Save graph and IDs without pickle; vectors remain in the caller's store."""
        self.validate()
        metadata = {"format": 1, "M": self.M, "ef_construction": self.ef_construction,
                    "entrypoint": self.entrypoint, "max_level": self.max_level,
                    "size": self.size, "rng_state": self.rng.bit_generator.state,
                    "vector_hash": self._vector_hash()}
        arrays = {"metadata": json.dumps(metadata), "ids": self.ids, "levels": self.levels}
        for layer in range(self.max_level + 1):
            nodes = np.flatnonzero(self.levels >= layer)
            links = [self._neighbors(int(node), layer) for node in nodes]
            arrays[f"nodes_{layer}"] = nodes
            arrays[f"offsets_{layer}"] = np.r_[0, np.cumsum([len(row) for row in links])]
            arrays[f"links_{layer}"] = np.array([n for row in links for n in row], dtype=np.int32)
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as output:
            np.savez(output, **arrays)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)

    @classmethod
    def load(cls, path, vectors):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata["format"] != 1:
                raise ValueError("unsupported HNSW file format")
            index = cls(vectors, data["ids"], metadata["M"], metadata["ef_construction"])
            if index._vector_hash() != metadata["vector_hash"]:
                raise ValueError("graph does not match the supplied vectors")
            index.levels = data["levels"].copy()
            index.entrypoint, index.max_level = metadata["entrypoint"], metadata["max_level"]
            index.size = metadata["size"]
            index.rng.bit_generator.state = metadata["rng_state"]
            index.upper = [{} for _ in range(max(0, index.max_level))]
            for layer in range(index.max_level + 1):
                nodes, offsets, links = (data[f"{key}_{layer}"] for key in ("nodes", "offsets", "links"))
                for i, node in enumerate(nodes):
                    index._set_neighbors(int(node), layer, links[offsets[i]:offsets[i + 1]].tolist())
        index.validate()
        return index
