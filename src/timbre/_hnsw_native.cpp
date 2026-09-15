// Hand-written HNSW construction and traversal, sharing one distance/search core.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_set>
#include <vector>

namespace py = pybind11;
template <typename T> using Array = py::array_t<T, py::array::c_style>;
using Hit = std::pair<float, int32_t>;

struct Links {
    const int32_t* data;
    size_t size;
    const int32_t* begin() const { return data; }
    const int32_t* end() const { return size ? data + size : data; }
};

class VectorSpace {
protected:
    Array<float> vectors_;  // Retain the NumPy owner, including mmap-backed data.
    size_t dimension_;

public:
    explicit VectorSpace(Array<float> vectors) : vectors_(std::move(vectors)) {
        if (vectors_.ndim() != 2 || vectors_.shape(1) == 0 ||
            vectors_.shape(0) > std::numeric_limits<int32_t>::max())
            throw std::invalid_argument("invalid vectors shape");
        dimension_ = size_t(vectors_.shape(1));
    }

    float distance(const float* query, int32_t node, size_t& evaluations) const {
        ++evaluations;
        const float* vector = vectors_.data() + size_t(node) * dimension_;
        // Independent accumulators permit SIMD without fast-math reassociation.
        float sums[8] = {};
        size_t i = 0;
        for (; i + 8 <= dimension_; i += 8)
            for (size_t lane = 0; lane < 8; ++lane)
                sums[lane] += vector[i + lane] * query[i + lane];
        float dot = ((sums[0] + sums[1]) + (sums[2] + sums[3])) +
                    ((sums[4] + sums[5]) + (sums[6] + sums[7]));
        for (; i < dimension_; ++i) dot += vector[i] * query[i];
        return 1.0f - dot;
    }
};

template <typename Graph>
int32_t greedy(const Graph& graph, const float* query, int32_t entry, int layer,
               size_t& evaluations) {
    float current = graph.distance(query, entry, evaluations);
    while (true) {
        int32_t best = entry;
        float nearest = current;
        for (auto node : graph.neighbors(entry, layer)) {
            const float d = graph.distance(query, node, evaluations);
            if (d < nearest) { nearest = d; best = node; }
        }
        if (best == entry) return entry;
        entry = best;
        current = nearest;
    }
}

template <typename Graph>
std::vector<Hit> search_layer(const Graph& graph, const float* query,
                             const std::vector<int32_t>& entries, size_t ef,
                             int layer, size_t& evaluations) {
    std::priority_queue<Hit, std::vector<Hit>, std::greater<Hit>> candidates;
    std::priority_queue<Hit> best;
    std::unordered_set<int32_t> visited(entries.begin(), entries.end());
    std::vector<Hit> initial;
    for (auto node : entries) initial.emplace_back(graph.distance(query, node, evaluations), node);
    std::sort(initial.begin(), initial.end());
    if (initial.size() > ef) initial.resize(ef);
    for (auto hit : initial) { candidates.push(hit); best.push(hit); }
    while (!candidates.empty()) {
        const auto [d, node] = candidates.top();
        candidates.pop();
        if (best.size() >= ef && d > best.top().first) break;
        for (auto neighbor : graph.neighbors(node, layer)) {
            if (!visited.insert(neighbor).second) continue;
            const Hit hit{graph.distance(query, neighbor, evaluations), neighbor};
            if (best.size() < ef || hit < best.top()) {
                candidates.push(hit);
                if (best.size() >= ef) best.pop();
                best.push(hit);
            }
        }
    }
    // A max-heap yields the reverse of the sorted beam needed for insertion.
    std::vector<Hit> result(best.size());
    for (size_t i = result.size(); i > 0; --i) { result[i - 1] = best.top(); best.pop(); }
    return result;
}

struct Layer {
    Array<int32_t> nodes, links;
    Array<int64_t> offsets;

    size_t row(int32_t node) const {
        // The completed base layer is dense; upper layers use sorted node IDs.
        const auto begin = nodes.data(), end = begin + nodes.size();
        if (size_t(node) < size_t(nodes.size()) && begin[node] == node) return node;
        const auto found = std::lower_bound(begin, end, node);
        if (found == end || *found != node)
            throw std::invalid_argument("node missing from graph layer");
        return size_t(found - begin);
    }
};

class SearchIndex : public VectorSpace {
    Array<int64_t> ids_;
    int32_t entry_;
    std::vector<Layer> layers_;

public:
    Links neighbors(int32_t node, int layer) const {
        const auto& graph = layers_[layer];
        const size_t row = graph.row(node);
        const auto start = graph.offsets.data()[row];
        const auto count = graph.offsets.data()[row + 1] - start;
        return {graph.links.size() ? graph.links.data() + start : nullptr, size_t(count)};
    }

    SearchIndex(Array<float> vectors, Array<int64_t> ids, int32_t entry,
                const std::vector<std::tuple<Array<int32_t>, Array<int64_t>, Array<int32_t>>>& layers)
        : VectorSpace(std::move(vectors)), ids_(std::move(ids)), entry_(entry) {
        if (ids_.ndim() != 1 || ids_.size() != vectors_.shape(0))
            throw std::invalid_argument("invalid vectors/IDs shape");
        dimension_ = size_t(vectors_.shape(1));
        for (const auto& [nodes, offsets, links] : layers) {
            if (nodes.ndim() != 1 || offsets.ndim() != 1 || links.ndim() != 1 ||
                offsets.size() != nodes.size() + 1 || nodes.size() == 0)
                throw std::invalid_argument("invalid graph array shapes");
            Layer layer{nodes, links, offsets};
            const auto offset_begin = offsets.data(), offset_end = offset_begin + offsets.size();
            const auto node_begin = nodes.data(), node_end = node_begin + nodes.size();
            if (offset_begin[0] != 0 || offset_begin[offsets.size() - 1] != links.size() ||
                !std::is_sorted(offset_begin, offset_end) ||
                !std::is_sorted(node_begin, node_end) ||
                std::adjacent_find(node_begin, node_end) != node_end ||
                node_begin[0] < 0 || node_begin[nodes.size() - 1] >= vectors_.shape(0))
                throw std::invalid_argument("invalid graph nodes/offsets");
            for (py::ssize_t i = 0; i < links.size(); ++i) layer.row(links.data()[i]);
            if (!layers_.empty())
                for (py::ssize_t i = 0; i < nodes.size(); ++i)
                    layers_.back().row(nodes.data()[i]);
            layers_.push_back(std::move(layer));
        }
        if (layers_.empty()) {
            if (entry_ != -1) throw std::invalid_argument("invalid empty graph entry point");
        } else {
            layers_.back().row(entry_);
        }
    }

    py::tuple search(Array<float> query, int k, int ef) const {
        if (k < 1 || ef < k) throw std::invalid_argument("require ef_search >= k >= 1");
        if (query.ndim() != 1 || size_t(query.size()) != dimension_)
            throw std::invalid_argument("query dimension mismatch");
        bool nonzero = false;
        for (size_t i = 0; i < dimension_; ++i) {
            if (!std::isfinite(query.data()[i])) throw std::invalid_argument("query must be finite");
            nonzero |= query.data()[i] != 0;
        }
        if (!nonzero) throw std::invalid_argument("query must be nonzero");
        size_t evaluations = 0;
        std::vector<Hit> found;
        {
            py::gil_scoped_release release;
            if (!layers_.empty()) {
                int32_t entry = entry_;
                for (size_t layer = layers_.size() - 1; layer > 0; --layer)
                    entry = greedy(*this, query.data(), entry, int(layer), evaluations);
                found = search_layer(*this, query.data(), {entry}, size_t(ef), 0, evaluations);
                std::sort(found.begin(), found.end(), [this](const Hit& a, const Hit& b) {
                    return a.first != b.first ? a.first < b.first : ids_.data()[a.second] < ids_.data()[b.second];
                });
            }
        }
        const size_t count = std::min(size_t(k), found.size());
        py::array_t<int64_t> ids(count);
        py::array_t<float> scores(count);
        for (size_t i = 0; i < count; ++i) {
            ids.mutable_data()[i] = ids_.data()[found[i].second];
            scores.mutable_data()[i] = 1.0f - found[i].first;
        }
        return py::make_tuple(ids, scores, evaluations);
    }
};

template <typename T>
py::array_t<T> to_array(const std::vector<T>& values) {
    py::array_t<T> result(values.size());
    std::copy(values.begin(), values.end(), result.mutable_data());
    return result;
}

class Builder : public VectorSpace {
    using Adjacency = std::vector<std::vector<std::vector<int32_t>>>;
    Adjacency graph_;  // node -> layer -> neighbors; empty means uninserted.
    int M_, ef_;
    int32_t entry_ = -1;
    int max_level_ = -1;
    size_t size_ = 0;

    std::vector<int32_t> select_neighbors(std::vector<Hit> candidates, size_t limit) const {
        std::sort(candidates.begin(), candidates.end());
        std::vector<int32_t> selected;
        size_t evaluations = 0;
        for (const auto [d, node] : candidates) {
            bool redundant = false;
            if (candidates.size() > limit) {
                for (auto neighbor : selected) {
                    if (distance(vectors_.data() + size_t(neighbor) * dimension_, node, evaluations) < d) {
                        redundant = true;
                        break;
                    }
                }
            }
            if (!redundant) selected.push_back(node);
            if (selected.size() == limit) break;
        }
        return selected;
    }

    void insert(int32_t node, int level) {
        graph_[node].resize(size_t(level) + 1);
        ++size_;
        if (entry_ == -1) { entry_ = node; max_level_ = level; return; }
        const float* query = vectors_.data() + size_t(node) * dimension_;
        size_t evaluations = 0;
        int32_t entry = entry_;
        for (int layer = max_level_; layer > level; --layer)
            entry = greedy(*this, query, entry, layer, evaluations);
        std::vector<int32_t> entries{entry};
        for (int layer = std::min(level, max_level_); layer >= 0; --layer) {
            auto candidates = search_layer(*this, query, entries, size_t(ef_), layer, evaluations);
            auto selected = select_neighbors(candidates, size_t(M_));
            graph_[node][layer] = selected;
            const size_t capacity = size_t(layer == 0 ? 2 * M_ : M_);
            for (auto neighbor : selected) {
                auto& links = graph_[neighbor][layer];
                links.push_back(node);
                if (links.size() > capacity) {
                    std::vector<Hit> distances;
                    for (auto link : links)
                        distances.emplace_back(distance(vectors_.data() + size_t(neighbor) * dimension_,
                                                         link, evaluations), link);
                    links = select_neighbors(std::move(distances), capacity);
                }
            }
            entries.clear();
            for (auto hit : candidates) entries.push_back(hit.second);
        }
        if (level > max_level_) { entry_ = node; max_level_ = level; }
    }

    void check_graph(const Adjacency& graph, int32_t entry) const {
        if (graph.size() != size_t(vectors_.shape(0)))
            throw std::invalid_argument("graph size does not match vectors");
        size_t max_layers = 0;
        for (size_t node = 0; node < graph.size(); ++node) {
            max_layers = std::max(max_layers, graph[node].size());
            if (graph[node].size() > 32768) throw std::invalid_argument("invalid node level");
            for (size_t layer = 0; layer < graph[node].size(); ++layer) {
                const auto& links = graph[node][layer];
                if (links.size() > size_t(layer == 0 ? 2 * M_ : M_))
                    throw std::invalid_argument("degree limit exceeded");
                std::unordered_set<int32_t> seen;
                for (auto link : links) {
                    if (link < 0 || size_t(link) >= graph.size() || size_t(link) == node ||
                        graph[link].size() <= layer || !seen.insert(link).second)
                        throw std::invalid_argument("invalid graph link");
                }
            }
        }
        if (max_layers == 0 ? entry != -1 :
            (entry < 0 || size_t(entry) >= graph.size() || graph[entry].size() != max_layers))
            throw std::invalid_argument("invalid entry point");
    }

public:
    Builder(Array<float> vectors, int M, int ef) : VectorSpace(std::move(vectors)), M_(M), ef_(ef) {
        if (M < 2 || M > std::numeric_limits<int>::max() / 2 || ef < M)
            throw std::invalid_argument("require M >= 2 and ef_construction >= M");
        graph_.resize(size_t(vectors_.shape(0)));
    }

    Links neighbors(int32_t node, int layer) const {
        const auto& links = graph_[node][layer];
        return {links.data(), links.size()};
    }

    void build(Array<int32_t> order, Array<int16_t> levels) {
        if (order.ndim() != 1 || levels.ndim() != 1 || order.size() != levels.size())
            throw std::invalid_argument("order and levels must be equal-length arrays");
        std::unordered_set<int32_t> seen;
        for (py::ssize_t i = 0; i < order.size(); ++i) {
            const auto node = order.data()[i];
            if (node < 0 || size_t(node) >= graph_.size() || !graph_[node].empty() ||
                !seen.insert(node).second || levels.data()[i] < 0)
                throw std::invalid_argument("require unique uninserted nodes and nonnegative levels");
        }
        // Keep the GIL during each bounded batch: this mutable builder is serial.
        for (py::ssize_t i = 0; i < order.size(); ++i) insert(order.data()[i], levels.data()[i]);
    }

    void restore(Adjacency graph, int32_t entry) {
        check_graph(graph, entry);
        graph_ = std::move(graph);
        entry_ = entry;
        size_ = 0;
        max_level_ = -1;
        for (const auto& node : graph_) {
            size_ += !node.empty();
            max_level_ = std::max(max_level_, int(node.size()) - 1);
        }
    }

    void validate() const { check_graph(graph_, entry_); }
    size_t size() const { return size_; }
    int32_t entrypoint() const { return entry_; }
    int max_level() const { return max_level_; }

    py::array_t<int16_t> levels() const {
        py::array_t<int16_t> result(graph_.size());
        for (size_t i = 0; i < graph_.size(); ++i) result.mutable_data()[i] = int(graph_[i].size()) - 1;
        return result;
    }

    py::dict export_graph() const {
        py::dict arrays;
        arrays["levels"] = levels();
        for (int layer = 0; layer <= max_level_; ++layer) {
            std::vector<int64_t> nodes, offsets{0};
            std::vector<int32_t> links;
            for (size_t node = 0; node < graph_.size(); ++node) {
                if (graph_[node].size() <= size_t(layer)) continue;
                nodes.push_back(int64_t(node));
                const auto& row = graph_[node][layer];
                links.insert(links.end(), row.begin(), row.end());
                offsets.push_back(int64_t(links.size()));
            }
            arrays[py::str("nodes_" + std::to_string(layer))] = to_array(nodes);
            arrays[py::str("offsets_" + std::to_string(layer))] = to_array(offsets);
            arrays[py::str("links_" + std::to_string(layer))] = to_array(links);
        }
        return arrays;
    }
};

PYBIND11_MODULE(_hnsw_native, m) {
    m.attr("compiler") = __VERSION__;
    m.attr("pybind11_version") = PYBIND11_TOSTRING(PYBIND11_VERSION_MAJOR) "."
        PYBIND11_TOSTRING(PYBIND11_VERSION_MINOR) "." PYBIND11_TOSTRING(PYBIND11_VERSION_PATCH);
    py::class_<SearchIndex>(m, "SearchIndex")
        .def(py::init<Array<float>, Array<int64_t>, int32_t,
             const std::vector<std::tuple<Array<int32_t>, Array<int64_t>, Array<int32_t>>>&>(),
             py::arg("vectors").noconvert(), py::arg("ids").noconvert(),
             py::arg("entrypoint"), py::arg("layers"))
        .def("search", &SearchIndex::search, py::arg("query").noconvert(),
             py::arg("k"), py::arg("ef_search"));
    py::class_<Builder>(m, "Builder")
        .def(py::init<Array<float>, int, int>(), py::arg("vectors").noconvert(),
             py::arg("M"), py::arg("ef_construction"))
        .def("build", &Builder::build, py::arg("order").noconvert(), py::arg("levels").noconvert())
        .def("restore", &Builder::restore)
        .def("validate", &Builder::validate)
        .def("export_graph", &Builder::export_graph)
        .def_property_readonly("levels", &Builder::levels)
        .def_property_readonly("size", &Builder::size)
        .def_property_readonly("entrypoint", &Builder::entrypoint)
        .def_property_readonly("max_level", &Builder::max_level);
}
