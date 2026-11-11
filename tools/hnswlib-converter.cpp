#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <endian.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "exchange-format.h"

namespace {

constexpr uint32_t kInvalidId = std::numeric_limits<uint32_t>::max();
constexpr uint32_t kFirstVecfsId = 1;
constexpr uint16_t kElementFp32 = 1;
constexpr uint16_t kDistanceL2 = 1;
constexpr uint16_t kDistanceIp = 2;

struct Cursor {
    const uint8_t *data = nullptr;
    size_t size = 0;
    size_t offset = 0;

    void require(size_t bytes) const {
        if (bytes > size || offset > size - bytes)
            throw std::runtime_error("unexpected end of file");
    }

    template<typename T>
    T read_native() {
        T value;

        require(sizeof(value));
        std::memcpy(&value, data + offset, sizeof(value));
        offset += sizeof(value);

        return value;
    }

    const uint8_t *read_bytes(size_t bytes) {
        const uint8_t *result;

        require(bytes);
        result = data + offset;
        offset += bytes;

        return result;
    }
};

struct Crc32 {
    uint32_t table[256]{};

    Crc32() {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t value = i;

            for (unsigned int bit = 0; bit < 8; ++bit) {
                value = (value & 1)
                    ? 0xedb88320U ^ (value >> 1)
                    : value >> 1;
            }

            table[i] = value;
        }
    }

    uint32_t update(uint32_t crc,
                    const void *buffer,
                    size_t length) const {
        const auto *bytes =
            static_cast<const uint8_t *>(buffer);

        for (size_t i = 0; i < length; ++i)
            crc = table[(crc ^ bytes[i]) & 0xffU] ^
                  (crc >> 8);

        return crc;
    }

    uint32_t calculate(const void *buffer,
                       size_t length) const {
        return update(
            std::numeric_limits<uint32_t>::max(),
            buffer,
            length) ^
            std::numeric_limits<uint32_t>::max();
    }
};

const Crc32 g_crc32;

template<typename T>
void append_native(std::vector<uint8_t> &output,
                   const T &value) {
    const size_t old_size = output.size();

    output.resize(old_size + sizeof(value));
    std::memcpy(
        output.data() + old_size,
        &value,
        sizeof(value));
}

void append_bytes(std::vector<uint8_t> &output,
                  const void *data,
                  size_t bytes) {
    const size_t old_size = output.size();

    if (bytes > std::numeric_limits<size_t>::max() -
                    old_size)
        throw std::overflow_error("output size overflow");

    output.resize(old_size + bytes);

    if (bytes)
        std::memcpy(
            output.data() + old_size,
            data,
            bytes);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

template<typename T>
void store_le(std::vector<uint8_t> &buffer,
              size_t offset,
              T value);

template<>
void store_le<uint16_t>(std::vector<uint8_t> &buffer,
                        size_t offset,
                        uint16_t value) {
    const uint16_t disk = htole16(value);

    if (offset > buffer.size() ||
        sizeof(disk) > buffer.size() - offset)
        throw std::out_of_range("store_le16");

    std::memcpy(buffer.data() + offset,
                &disk,
                sizeof(disk));
}

template<>
void store_le<uint32_t>(std::vector<uint8_t> &buffer,
                        size_t offset,
                        uint32_t value) {
    const uint32_t disk = htole32(value);

    if (offset > buffer.size() ||
        sizeof(disk) > buffer.size() - offset)
        throw std::out_of_range("store_le32");

    std::memcpy(buffer.data() + offset,
                &disk,
                sizeof(disk));
}

template<>
void store_le<uint64_t>(std::vector<uint8_t> &buffer,
                        size_t offset,
                        uint64_t value) {
    const uint64_t disk = htole64(value);

    if (offset > buffer.size() ||
        sizeof(disk) > buffer.size() - offset)
        throw std::out_of_range("store_le64");

    std::memcpy(buffer.data() + offset,
                &disk,
                sizeof(disk));
}

uint16_t load_le16(const void *pointer) {
    uint16_t value;

    std::memcpy(&value, pointer, sizeof(value));
    return le16toh(value);
}

uint32_t load_le32(const void *pointer) {
    uint32_t value;

    std::memcpy(&value, pointer, sizeof(value));
    return le32toh(value);
}

uint64_t load_le64(const void *pointer) {
    uint64_t value;

    std::memcpy(&value, pointer, sizeof(value));
    return le64toh(value);
}

#pragma GCC diagnostic pop

std::vector<uint8_t> read_file(const std::string &path) {
    std::ifstream input(path, std::ios::binary);

    if (!input)
        throw std::runtime_error(
            "cannot open input file: " + path);

    input.seekg(0, std::ios::end);
    const std::streamoff length = input.tellg();

    if (length < 0)
        throw std::runtime_error(
            "cannot determine input file size");

    input.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(
        static_cast<size_t>(length));

    if (!data.empty()) {
        input.read(
            reinterpret_cast<char *>(data.data()),
            static_cast<std::streamsize>(data.size()));

        if (!input)
            throw std::runtime_error(
                "failed to read input file");
    }

    return data;
}

void write_file(const std::string &path,
                const std::vector<uint8_t> &data) {
    std::ofstream output(
        path,
        std::ios::binary |
        std::ios::trunc);

    if (!output)
        throw std::runtime_error(
            "cannot create output file: " + path);

    if (!data.empty()) {
        output.write(
            reinterpret_cast<const char *>(data.data()),
            static_cast<std::streamsize>(data.size()));

        if (!output)
            throw std::runtime_error(
                "failed to write output file");
    }

    output.flush();

    if (!output)
        throw std::runtime_error(
            "failed to flush output file");
}

uint16_t parse_distance(const std::string &value) {
    if (value == "l2")
        return kDistanceL2;

    if (value == "ip")
        return kDistanceIp;

    throw std::runtime_error(
        "distance must be l2 or ip");
}

const char *distance_name(uint16_t value) {
    switch (value) {
    case kDistanceL2:
        return "l2";
    case kDistanceIp:
        return "ip";
    default:
        return "unknown";
    }
}

struct GraphNode {
    uint32_t id = 0;
    uint64_t label = 0;
    uint16_t level = 0;

    std::vector<uint8_t> vector;
    std::vector<uint32_t> level0;
    std::vector<std::vector<uint32_t>> upper;
};

struct Graph {
    uint16_t element_type = kElementFp32;
    uint16_t distance_type = 0;

    uint32_t dimension = 0;

    uint16_t m = 0;
    uint16_t m0 = 0;
    uint16_t max_levels = 0;
    uint16_t current_max_level = 0;

    uint32_t ef_construction = 0;
    uint32_t default_ef_search = 64;

    uint32_t capacity = 0;
    uint32_t entry_node = kInvalidId;

    uint64_t random_seed = 100;

    std::vector<GraphNode> nodes;
};

void validate_graph(const Graph &graph) {
    if (graph.element_type != kElementFp32)
        throw std::runtime_error(
            "only FP32 graphs are supported");

    if (graph.distance_type != kDistanceL2 &&
        graph.distance_type != kDistanceIp)
        throw std::runtime_error(
            "invalid distance type");

    if (!graph.dimension)
        throw std::runtime_error(
            "zero vector dimension");

    if (graph.m < 2)
        throw std::runtime_error(
            "M must be at least 2");

    if (graph.m0 < graph.m)
        throw std::runtime_error(
            "M0 must not be smaller than M");

    if (!graph.max_levels ||
        graph.current_max_level >=
            graph.max_levels)
        throw std::runtime_error(
            "invalid max-level configuration");

    if (graph.capacity < 2)
        throw std::runtime_error(
            "invalid graph capacity");

    std::unordered_map<uint32_t, const GraphNode *> ids;
    std::unordered_map<uint64_t, uint32_t> labels;

    ids.reserve(graph.nodes.size());
    labels.reserve(graph.nodes.size());

    for (const GraphNode &node : graph.nodes) {
        if (node.id < kFirstVecfsId ||
            node.id >= graph.capacity)
            throw std::runtime_error(
                "node ID is outside graph capacity");

        if (!ids.emplace(node.id, &node).second)
            throw std::runtime_error(
                "duplicate node ID");

        if (!labels.emplace(
                node.label,
                node.id).second)
            throw std::runtime_error(
                "duplicate external label");

        if (node.level >= graph.max_levels)
            throw std::runtime_error(
                "node level exceeds maximum");

        if (node.vector.size() !=
            static_cast<size_t>(graph.dimension) *
                sizeof(float))
            throw std::runtime_error(
                "invalid vector length");

        if (node.level0.size() > graph.m0)
            throw std::runtime_error(
                "level-0 degree exceeds M0");

        if (node.upper.size() != node.level)
            throw std::runtime_error(
                "upper-level count mismatch");

        for (const auto &links : node.upper) {
            if (links.size() > graph.m)
                throw std::runtime_error(
                    "upper degree exceeds M");
        }
    }

    if (graph.nodes.empty()) {
        if (graph.entry_node != kInvalidId)
            throw std::runtime_error(
                "empty graph has an entry node");

        return;
    }

    const auto entry = ids.find(
        graph.entry_node);

    if (entry == ids.end())
        throw std::runtime_error(
            "entry node does not exist");

    if (entry->second->level !=
        graph.current_max_level)
        throw std::runtime_error(
            "entry node level mismatch");

    for (const GraphNode &node : graph.nodes) {
        auto check_link =
            [&](uint32_t neighbor,
                uint16_t level) {
                if (neighbor == node.id)
                    throw std::runtime_error(
                        "self-loop in graph");

                const auto found =
                    ids.find(neighbor);

                if (found == ids.end())
                    throw std::runtime_error(
                        "edge references missing node");

                if (found->second->level < level)
                    throw std::runtime_error(
                        "upper edge references a node "
                        "without that level");
            };

        for (uint32_t neighbor : node.level0)
            check_link(neighbor, 0);

        for (uint16_t level = 1;
             level <= node.level;
             ++level) {
            for (uint32_t neighbor :
                 node.upper[level - 1])
                check_link(neighbor, level);
        }
    }
}

/*
 * hnswlib native saveIndex header.
 *
 * hnswlib writes every field separately, so this code intentionally parses
 * each value rather than mapping a packed C++ structure over the input.
 */
Graph read_hnsw_native(
    const std::string &path,
    uint16_t distance_type,
    uint16_t configured_max_levels,
    uint32_t default_ef_search) {
    static_assert(sizeof(size_t) == 8,
                  "converter requires 64-bit size_t");
    static_assert(sizeof(double) == 8,
                  "converter requires 64-bit double");

#if __BYTE_ORDER != __LITTLE_ENDIAN
#error "hnswlib native converter requires little-endian host"
#endif

    const std::vector<uint8_t> file =
        read_file(path);

    Cursor cursor{
        file.data(),
        file.size(),
        0
    };

    const size_t offset_level0 =
        cursor.read_native<size_t>();

    const size_t max_elements =
        cursor.read_native<size_t>();

    const size_t current_count =
        cursor.read_native<size_t>();

    const size_t size_data_per_element =
        cursor.read_native<size_t>();

    const size_t label_offset =
        cursor.read_native<size_t>();

    const size_t data_offset =
        cursor.read_native<size_t>();

    const int32_t native_max_level =
        cursor.read_native<int32_t>();

    const uint32_t native_entry =
        cursor.read_native<uint32_t>();

    const size_t max_m =
        cursor.read_native<size_t>();

    const size_t max_m0 =
        cursor.read_native<size_t>();

    const size_t m =
        cursor.read_native<size_t>();

    const double multiplier =
        cursor.read_native<double>();

    const size_t ef_construction =
        cursor.read_native<size_t>();

    (void)max_elements;
    (void)multiplier;

    if (current_count >
        std::numeric_limits<uint32_t>::max() - 1)
        throw std::runtime_error(
            "hnswlib index has too many nodes");

    if (max_m > std::numeric_limits<uint16_t>::max() ||
        max_m0 > std::numeric_limits<uint16_t>::max() ||
        m > std::numeric_limits<uint16_t>::max())
        throw std::runtime_error(
            "hnswlib M values are too large");

    if (ef_construction >
        std::numeric_limits<uint32_t>::max())
        throw std::runtime_error(
            "ef_construction is too large");

    const size_t level0_links_bytes =
        max_m0 * sizeof(uint32_t) +
        sizeof(uint32_t);

    const size_t upper_stride =
        max_m * sizeof(uint32_t) +
        sizeof(uint32_t);

    if (offset_level0 != 0)
        throw std::runtime_error(
            "unsupported non-zero level-0 offset");

    if (data_offset < level0_links_bytes)
        throw std::runtime_error(
            "invalid hnswlib data offset");

    if (label_offset < data_offset)
        throw std::runtime_error(
            "invalid hnswlib label offset");

    const size_t vector_bytes =
        label_offset - data_offset;

    if (!vector_bytes ||
        vector_bytes % sizeof(float))
        throw std::runtime_error(
            "index does not contain an FP32 vector");

    if (label_offset >
            size_data_per_element ||
        sizeof(uint64_t) >
            size_data_per_element -
                label_offset)
        throw std::runtime_error(
            "invalid hnswlib label layout");

    if (current_count &&
        size_data_per_element >
            std::numeric_limits<size_t>::max() /
                current_count)
        throw std::overflow_error(
            "hnswlib level-0 size overflow");

    const size_t level0_total =
        current_count * size_data_per_element;

    const uint8_t *level0_memory =
        cursor.read_bytes(level0_total);

    if (native_max_level < -1)
        throw std::runtime_error(
            "invalid native maximum level");

    const uint16_t required_levels =
        native_max_level < 0
            ? 1
            : static_cast<uint16_t>(
                native_max_level + 1);

    if (configured_max_levels <
        required_levels)
        throw std::runtime_error(
            "--max-levels is smaller than the "
            "native graph level count");

    Graph graph;

    graph.distance_type =
        distance_type;

    graph.dimension =
        static_cast<uint32_t>(
            vector_bytes / sizeof(float));

    graph.m =
        static_cast<uint16_t>(m);

    graph.m0 =
        static_cast<uint16_t>(max_m0);

    graph.max_levels =
        configured_max_levels;

    graph.current_max_level =
        native_max_level < 0
            ? 0
            : static_cast<uint16_t>(
                native_max_level);

    graph.ef_construction =
        static_cast<uint32_t>(
            ef_construction);

    graph.default_ef_search =
        default_ef_search;

    graph.capacity =
        static_cast<uint32_t>(
            current_count + 1);

    graph.entry_node =
        current_count
            ? native_entry + 1
            : kInvalidId;

    graph.nodes.reserve(current_count);

    for (uint32_t native_id = 0;
         native_id < current_count;
         ++native_id) {
        const uint8_t *element =
            level0_memory +
            static_cast<size_t>(native_id) *
                size_data_per_element;

        GraphNode node;

        node.id = native_id + 1;

        node.label = 0;
        std::memcpy(
            &node.label,
            element + label_offset,
            sizeof(node.label));

        node.vector.assign(
            element + data_offset,
            element + data_offset +
                vector_bytes);

        const uint16_t level0_count =
            load_le16(
                element + offset_level0);

        if (level0_count > max_m0)
            throw std::runtime_error(
                "native level-0 degree exceeds M0");

        node.level0.reserve(level0_count);

        for (uint16_t i = 0;
             i < level0_count;
             ++i) {
            uint32_t neighbor;

            std::memcpy(
                &neighbor,
                element + offset_level0 +
                    sizeof(uint32_t) +
                    static_cast<size_t>(i) *
                        sizeof(uint32_t),
                sizeof(neighbor));

            if (neighbor >= current_count)
                throw std::runtime_error(
                    "native level-0 edge is invalid");

            node.level0.push_back(
                neighbor + 1);
        }

        graph.nodes.push_back(
            std::move(node));
    }

    for (uint32_t native_id = 0;
         native_id < current_count;
         ++native_id) {
        const uint32_t link_list_size =
            cursor.read_native<uint32_t>();

        if (!link_list_size)
            continue;

        if (!upper_stride ||
            link_list_size % upper_stride)
            throw std::runtime_error(
                "invalid native upper link-list size");

        const size_t levels =
            link_list_size / upper_stride;

        if (levels >= configured_max_levels)
            throw std::runtime_error(
                "native node level exceeds configured "
                "maximum");

        const uint8_t *upper_memory =
            cursor.read_bytes(
                link_list_size);

        GraphNode &node =
            graph.nodes[native_id];

        node.level =
            static_cast<uint16_t>(levels);

        node.upper.resize(levels);

        for (size_t level_index = 0;
             level_index < levels;
             ++level_index) {
            const uint8_t *block =
                upper_memory +
                level_index * upper_stride;

            const uint16_t count =
                load_le16(block);

            if (count > max_m)
                throw std::runtime_error(
                    "native upper degree exceeds M");

            auto &links =
                node.upper[level_index];

            links.reserve(count);

            for (uint16_t i = 0;
                 i < count;
                 ++i) {
                uint32_t neighbor;

                std::memcpy(
                    &neighbor,
                    block +
                        sizeof(uint32_t) +
                        static_cast<size_t>(i) *
                            sizeof(uint32_t),
                    sizeof(neighbor));

                if (neighbor >= current_count)
                    throw std::runtime_error(
                        "native upper edge is invalid");

                links.push_back(
                    neighbor + 1);
            }
        }
    }

    if (cursor.offset != cursor.size)
        throw std::runtime_error(
            "unexpected trailing bytes in native index");

    validate_graph(graph);

    return graph;
}

void update_exchange_header_crc(
    vecfs_exchange_file_header &header) {
    header.header_crc = htole32(0);

    header.header_crc = htole32(
        g_crc32.calculate(
            &header,
            sizeof(header)));
}

std::vector<uint8_t> build_exchange_record(
    const Graph &graph,
    const GraphNode &node) {
    const size_t vector_bytes =
        node.vector.size();

    const size_t level0_bytes =
        node.level0.size() *
        sizeof(uint32_t);

    const size_t upper_counts_bytes =
        node.level *
        sizeof(uint16_t);

    const size_t upper_slots =
        static_cast<size_t>(node.level) *
        graph.m;

    const size_t upper_links_bytes =
        upper_slots * sizeof(uint32_t);

    size_t record_size =
        sizeof(vecfs_exchange_node_header);

    auto checked_add =
        [&](size_t value) {
            if (value >
                std::numeric_limits<size_t>::max() -
                    record_size)
                throw std::overflow_error(
                    "exchange record size overflow");

            record_size += value;
        };

    checked_add(vector_bytes);
    checked_add(level0_bytes);
    checked_add(upper_counts_bytes);
    checked_add(upper_links_bytes);

    if (record_size >
        VECFS_EXCHANGE_MAX_RECORD_BYTES)
        throw std::runtime_error(
            "exchange node record is too large");

    std::vector<uint8_t> record(
        record_size,
        0);

    auto *header =
        reinterpret_cast<
            vecfs_exchange_node_header *>(
                record.data());

    header->node_id =
        htole32(node.id);

    header->level =
        htole16(node.level);

    header->flags =
        htole16(VECFS_EXCHANGE_NODE_F_NONE);

    header->external_id =
        htole64(node.label);

    header->vector_bytes =
        htole32(
            static_cast<uint32_t>(
                vector_bytes));

    header->level0_count =
        htole16(
            static_cast<uint16_t>(
                node.level0.size()));

    header->upper_level_count =
        htole16(node.level);

    header->record_bytes =
        htole64(record_size);

    header->record_crc =
        htole32(0);

    uint8_t *cursor =
        record.data() + sizeof(*header);

    std::memcpy(
        cursor,
        node.vector.data(),
        vector_bytes);

    cursor += vector_bytes;

    for (uint32_t neighbor :
         node.level0) {
        const uint32_t disk =
            htole32(neighbor);

        std::memcpy(
            cursor,
            &disk,
            sizeof(disk));

        cursor += sizeof(disk);
    }

    for (uint16_t level = 0;
         level < node.level;
         ++level) {
        const uint16_t disk =
            htole16(
                static_cast<uint16_t>(
                    node.upper[level].size()));

        std::memcpy(
            cursor,
            &disk,
            sizeof(disk));

        cursor += sizeof(disk);
    }

    for (uint16_t level = 0;
         level < node.level;
         ++level) {
        const auto &links =
            node.upper[level];

        for (uint16_t slot = 0;
             slot < graph.m;
             ++slot) {
            const uint32_t value =
                slot < links.size()
                    ? links[slot]
                    : 0;

            const uint32_t disk =
                htole32(value);

            std::memcpy(
                cursor,
                &disk,
                sizeof(disk));

            cursor += sizeof(disk);
        }
    }

    if (cursor !=
        record.data() + record.size())
        throw std::logic_error(
            "exchange record layout mismatch");

    header->record_crc = htole32(0);

    header->record_crc = htole32(
        g_crc32.calculate(
            record.data(),
            record.size()));

    return record;
}

void write_exchange(const std::string &path,
                    const Graph &graph) {
    validate_graph(graph);

    std::vector<uint8_t> records;

    for (const GraphNode &node :
         graph.nodes) {
        const std::vector<uint8_t> record =
            build_exchange_record(
                graph,
                node);

        append_bytes(
            records,
            record.data(),
            record.size());
    }

    vecfs_exchange_file_header header{};

    header.magic =
        htole64(VECFS_EXCHANGE_FILE_MAGIC);

    header.major =
        htole16(VECFS_EXCHANGE_FILE_MAJOR);

    header.minor =
        htole16(VECFS_EXCHANGE_FILE_MINOR);

    header.element_type =
        htole16(graph.element_type);

    header.distance_type =
        htole16(graph.distance_type);

    header.dimension =
        htole32(graph.dimension);

    header.m =
        htole16(graph.m);

    header.m0 =
        htole16(graph.m0);

    header.max_levels =
        htole16(graph.max_levels);

    header.current_max_level =
        htole16(graph.current_max_level);

    header.ef_construction =
        htole32(graph.ef_construction);

    header.default_ef_search =
        htole32(graph.default_ef_search);

    header.node_count =
        htole64(graph.nodes.size());

    header.node_capacity =
        htole32(graph.capacity);

    header.entry_node_id =
        htole32(graph.entry_node);

    header.random_seed =
        htole64(graph.random_seed);

    header.node_records_offset =
        htole64(sizeof(header));

    header.node_records_bytes =
        htole64(records.size());

    header.records_crc =
        htole32(
            g_crc32.calculate(
                records.data(),
                records.size()));

    update_exchange_header_crc(header);

    std::vector<uint8_t> output;

    append_bytes(
        output,
        &header,
        sizeof(header));

    append_bytes(
        output,
        records.data(),
        records.size());

    write_file(path, output);
}

Graph read_exchange(const std::string &path) {
    const std::vector<uint8_t> file =
        read_file(path);

    if (file.size() <
        sizeof(vecfs_exchange_file_header))
        throw std::runtime_error(
            "exchange file is too small");

    vecfs_exchange_file_header header;

    std::memcpy(
        &header,
        file.data(),
        sizeof(header));

    if (le64toh(header.magic) !=
        VECFS_EXCHANGE_FILE_MAGIC)
        throw std::runtime_error(
            "invalid exchange file magic");

    if (le16toh(header.major) !=
            VECFS_EXCHANGE_FILE_MAJOR ||
        le16toh(header.minor) >
            VECFS_EXCHANGE_FILE_MINOR)
        throw std::runtime_error(
            "unsupported exchange version");

    {
        const uint32_t expected =
            le32toh(header.header_crc);

        header.header_crc =
            htole32(0);

        const uint32_t actual =
            g_crc32.calculate(
                &header,
                sizeof(header));

        if (expected != actual)
            throw std::runtime_error(
                "exchange header CRC mismatch");
    }

    const uint64_t records_offset =
        le64toh(header.node_records_offset);

    const uint64_t records_bytes =
        le64toh(header.node_records_bytes);

    if (records_offset <
            sizeof(header) ||
        records_offset > file.size() ||
        records_bytes >
            file.size() - records_offset ||
        records_offset + records_bytes !=
            file.size())
        throw std::runtime_error(
            "invalid exchange record region");

    const uint8_t *record_data =
        file.data() + records_offset;

    if (g_crc32.calculate(
            record_data,
            records_bytes) !=
        le32toh(header.records_crc))
        throw std::runtime_error(
            "exchange records CRC mismatch");

    Graph graph;

    graph.element_type =
        le16toh(header.element_type);

    graph.distance_type =
        le16toh(header.distance_type);

    graph.dimension =
        le32toh(header.dimension);

    graph.m =
        le16toh(header.m);

    graph.m0 =
        le16toh(header.m0);

    graph.max_levels =
        le16toh(header.max_levels);

    graph.current_max_level =
        le16toh(header.current_max_level);

    graph.ef_construction =
        le32toh(header.ef_construction);

    graph.default_ef_search =
        le32toh(header.default_ef_search);

    graph.capacity =
        le32toh(header.node_capacity);

    graph.entry_node =
        le32toh(header.entry_node_id);

    graph.random_seed =
        le64toh(header.random_seed);

    const uint64_t expected_nodes =
        le64toh(header.node_count);

    Cursor cursor{
        record_data,
        static_cast<size_t>(records_bytes),
        0
    };

    graph.nodes.reserve(
        static_cast<size_t>(expected_nodes));

    while (cursor.offset < cursor.size) {
        const size_t record_start =
            cursor.offset;

        const auto *disk_header =
            reinterpret_cast<
                const vecfs_exchange_node_header *>(
                    cursor.read_bytes(
                        sizeof(
                            vecfs_exchange_node_header)));

        const uint64_t record_bytes =
            le64toh(
                disk_header->record_bytes);

        if (record_bytes <
                sizeof(*disk_header) ||
            record_bytes >
                VECFS_EXCHANGE_MAX_RECORD_BYTES)
            throw std::runtime_error(
                "invalid exchange node record size");

        const size_t payload_bytes =
            static_cast<size_t>(
                record_bytes -
                sizeof(*disk_header));

        cursor.read_bytes(payload_bytes);

        std::vector<uint8_t> record(
            record_bytes);

        std::memcpy(
            record.data(),
            record_data + record_start,
            record_bytes);

        auto *mutable_header =
            reinterpret_cast<
                vecfs_exchange_node_header *>(
                    record.data());

        const uint32_t expected_crc =
            le32toh(
                mutable_header->record_crc);

        mutable_header->record_crc =
            htole32(0);

        if (g_crc32.calculate(
                record.data(),
                record.size()) !=
            expected_crc)
            throw std::runtime_error(
                "exchange node CRC mismatch");

        const uint16_t level =
            le16toh(disk_header->level);

        const uint16_t level0_count =
            le16toh(
                disk_header->level0_count);

        const uint16_t upper_level_count =
            le16toh(
                disk_header->upper_level_count);

        const uint32_t vector_bytes =
            le32toh(
                disk_header->vector_bytes);

        if (level != upper_level_count)
            throw std::runtime_error(
                "upper level count mismatch");

        if (level >= graph.max_levels ||
            level0_count > graph.m0)
            throw std::runtime_error(
                "node degree or level is invalid");

        if (vector_bytes !=
            graph.dimension * sizeof(float))
            throw std::runtime_error(
                "invalid vector byte count");

        size_t required =
            sizeof(*disk_header) +
            vector_bytes +
            static_cast<size_t>(
                level0_count) *
                sizeof(uint32_t) +
            static_cast<size_t>(
                level) *
                sizeof(uint16_t) +
            static_cast<size_t>(
                level) *
                graph.m *
                sizeof(uint32_t);

        if (required != record_bytes)
            throw std::runtime_error(
                "node record layout mismatch");

        const uint8_t *payload =
            record_data +
            record_start +
            sizeof(*disk_header);

        GraphNode node;

        node.id =
            le32toh(disk_header->node_id);

        node.level = level;

        node.label =
            le64toh(
                disk_header->external_id);

        node.vector.assign(
            payload,
            payload + vector_bytes);

        payload += vector_bytes;

        node.level0.reserve(
            level0_count);

        for (uint16_t i = 0;
             i < level0_count;
             ++i) {
            node.level0.push_back(
                load_le32(
                    payload +
                    static_cast<size_t>(i) *
                        sizeof(uint32_t)));
        }

        payload +=
            static_cast<size_t>(
                level0_count) *
            sizeof(uint32_t);

        std::vector<uint16_t> counts(
            level);

        for (uint16_t i = 0;
             i < level;
             ++i) {
            counts[i] =
                load_le16(
                    payload +
                    static_cast<size_t>(i) *
                        sizeof(uint16_t));

            if (counts[i] > graph.m)
                throw std::runtime_error(
                    "upper degree exceeds M");
        }

        payload +=
            static_cast<size_t>(level) *
            sizeof(uint16_t);

        node.upper.resize(level);

        for (uint16_t level_index = 0;
             level_index < level;
             ++level_index) {
            auto &links =
                node.upper[level_index];

            links.reserve(
                counts[level_index]);

            for (uint16_t i = 0;
                 i < counts[level_index];
                 ++i) {
                links.push_back(
                    load_le32(
                        payload +
                        (static_cast<size_t>(
                             level_index) *
                             graph.m +
                         i) *
                            sizeof(uint32_t)));
            }
        }

        graph.nodes.push_back(
            std::move(node));
    }

    if (graph.nodes.size() !=
        expected_nodes)
        throw std::runtime_error(
            "exchange node count mismatch");

    validate_graph(graph);

    return graph;
}

Graph compact_for_hnsw(const Graph &source) {
    validate_graph(source);

    Graph result = source;

    result.nodes.clear();
    result.nodes.reserve(
        source.nodes.size());

    std::vector<const GraphNode *> sorted;

    sorted.reserve(source.nodes.size());

    for (const GraphNode &node :
         source.nodes)
        sorted.push_back(&node);

    std::sort(
        sorted.begin(),
        sorted.end(),
        [](const GraphNode *left,
           const GraphNode *right) {
            return left->id < right->id;
        });

    std::unordered_map<uint32_t, uint32_t> remap;

    remap.reserve(sorted.size());

    for (uint32_t native_id = 0;
         native_id < sorted.size();
         ++native_id) {
        remap.emplace(
            sorted[native_id]->id,
            native_id);
    }

    result.capacity =
        static_cast<uint32_t>(
            sorted.size() + 1);

    result.entry_node =
        source.nodes.empty()
            ? kInvalidId
            : remap.at(
                source.entry_node) + 1;

    for (uint32_t native_id = 0;
         native_id < sorted.size();
         ++native_id) {
        GraphNode node =
            *sorted[native_id];

        node.id = native_id + 1;

        for (uint32_t &neighbor :
             node.level0)
            neighbor =
                remap.at(neighbor) + 1;

        for (auto &level : node.upper) {
            for (uint32_t &neighbor :
                 level)
                neighbor =
                    remap.at(neighbor) + 1;
        }

        result.nodes.push_back(
            std::move(node));
    }

    validate_graph(result);

    return result;
}

void write_hnsw_native(const std::string &path,
                       const Graph &input) {
    const Graph graph =
        compact_for_hnsw(input);

    const size_t current_count =
        graph.nodes.size();

    const size_t max_elements =
        current_count;

    const size_t size_links_level0 =
        static_cast<size_t>(graph.m0) *
            sizeof(uint32_t) +
        sizeof(uint32_t);

    const size_t vector_bytes =
        static_cast<size_t>(graph.dimension) *
        sizeof(float);

    const size_t offset_level0 = 0;
    const size_t offset_data =
        size_links_level0;

    const size_t label_offset =
        offset_data + vector_bytes;

    const size_t size_data_per_element =
        label_offset + sizeof(uint64_t);

    const int32_t max_level =
        current_count
            ? static_cast<int32_t>(
                graph.current_max_level)
            : -1;

    const uint32_t entry =
        current_count
            ? graph.entry_node - 1
            : kInvalidId;

    const size_t max_m = graph.m;
    const size_t max_m0 = graph.m0;
    const size_t m = graph.m;

    const double multiplier =
        1.0 / std::log(
            static_cast<double>(graph.m));

    const size_t ef_construction =
        graph.ef_construction;

    std::vector<uint8_t> output;

    append_native(output, offset_level0);
    append_native(output, max_elements);
    append_native(output, current_count);
    append_native(output, size_data_per_element);
    append_native(output, label_offset);
    append_native(output, offset_data);
    append_native(output, max_level);
    append_native(output, entry);
    append_native(output, max_m);
    append_native(output, max_m0);
    append_native(output, m);
    append_native(output, multiplier);
    append_native(output, ef_construction);

    for (const GraphNode &node :
         graph.nodes) {
        std::vector<uint8_t> element(
            size_data_per_element,
            0);

        const uint16_t count =
            static_cast<uint16_t>(
                node.level0.size());

        std::memcpy(
            element.data(),
            &count,
            sizeof(count));

        for (size_t i = 0;
             i < node.level0.size();
             ++i) {
            const uint32_t native_neighbor =
                node.level0[i] - 1;

            std::memcpy(
                element.data() +
                    sizeof(uint32_t) +
                    i * sizeof(uint32_t),
                &native_neighbor,
                sizeof(native_neighbor));
        }

        std::memcpy(
            element.data() + offset_data,
            node.vector.data(),
            vector_bytes);

        std::memcpy(
            element.data() + label_offset,
            &node.label,
            sizeof(node.label));

        append_bytes(
            output,
            element.data(),
            element.size());
    }

    const size_t upper_stride =
        static_cast<size_t>(graph.m) *
            sizeof(uint32_t) +
        sizeof(uint32_t);

    for (const GraphNode &node :
         graph.nodes) {
        if (!node.level) {
            const uint32_t zero = 0;
            append_native(output, zero);
            continue;
        }

        const size_t upper_bytes =
            static_cast<size_t>(node.level) *
            upper_stride;

        if (upper_bytes >
            std::numeric_limits<uint32_t>::max())
            throw std::overflow_error(
                "native upper links are too large");

        const uint32_t disk_size =
            static_cast<uint32_t>(
                upper_bytes);

        append_native(output, disk_size);

        std::vector<uint8_t> upper(
            upper_bytes,
            0);

        for (uint16_t level = 0;
             level < node.level;
             ++level) {
            uint8_t *block =
                upper.data() +
                static_cast<size_t>(level) *
                    upper_stride;

            const uint16_t count =
                static_cast<uint16_t>(
                    node.upper[level].size());

            std::memcpy(
                block,
                &count,
                sizeof(count));

            for (size_t i = 0;
                 i < node.upper[level].size();
                 ++i) {
                const uint32_t native_neighbor =
                    node.upper[level][i] - 1;

                std::memcpy(
                    block +
                        sizeof(uint32_t) +
                        i * sizeof(uint32_t),
                    &native_neighbor,
                    sizeof(native_neighbor));
            }
        }

        append_bytes(
            output,
            upper.data(),
            upper.size());
    }

    write_file(path, output);
}

struct Options {
    enum class Mode {
        None,
        FromHnsw,
        ToHnsw
    };

    Mode mode = Mode::None;

    std::string input;
    std::string output;

    uint16_t distance = 0;
    uint16_t max_levels = 0;

    uint32_t default_ef_search = 64;
};

uint32_t parse_u32(const std::string &value,
                   const char *name) {
    size_t consumed = 0;
    unsigned long result;

    errno = 0;

    result = std::stoul(
        value,
        &consumed,
        0);

    if (errno ||
        consumed != value.size() ||
        result >
            std::numeric_limits<uint32_t>::max())
        throw std::runtime_error(
            std::string("invalid ") + name);

    return static_cast<uint32_t>(result);
}

void usage(const char *program) {
    std::cerr
        << "Usage:\n"
        << "  " << program
        << " from-hnsw"
        << " --distance l2|ip"
        << " --max-levels N"
        << " [--ef-search N]"
        << " <hnsw-index> <exchange-file>\n"
        << "  " << program
        << " to-hnsw"
        << " <exchange-file> <hnsw-index>\n";
}

Options parse_options(int argc,
                      char **argv) {
    Options options;

    if (argc < 2)
        return options;

    int index = 1;

    const std::string mode = argv[index++];

    if (mode == "from-hnsw") {
        options.mode =
            Options::Mode::FromHnsw;

        while (index < argc &&
               argv[index][0] == '-') {
            const std::string argument =
                argv[index++];

            if (argument == "--distance") {
                if (index >= argc)
                    throw std::runtime_error(
                        "--distance requires a value");

                options.distance =
                    parse_distance(argv[index++]);
            } else if (
                argument == "--max-levels") {
                if (index >= argc)
                    throw std::runtime_error(
                        "--max-levels requires a value");

                const uint32_t value =
                    parse_u32(
                        argv[index++],
                        "max levels");

                if (!value ||
                    value >
                        std::numeric_limits<
                            uint16_t>::max())
                    throw std::runtime_error(
                        "max levels is out of range");

                options.max_levels =
                    static_cast<uint16_t>(
                        value);
            } else if (
                argument == "--ef-search") {
                if (index >= argc)
                    throw std::runtime_error(
                        "--ef-search requires a value");

                options.default_ef_search =
                    parse_u32(
                        argv[index++],
                        "ef search");

                if (!options.default_ef_search)
                    throw std::runtime_error(
                        "ef search must be positive");
            } else {
                throw std::runtime_error(
                    "unknown option: " + argument);
            }
        }

        if (!options.distance)
            throw std::runtime_error(
                "--distance is required");

        if (!options.max_levels)
            throw std::runtime_error(
                "--max-levels is required");
    } else if (mode == "to-hnsw") {
        options.mode =
            Options::Mode::ToHnsw;
    } else {
        throw std::runtime_error(
            "unknown command: " + mode);
    }

    if (argc - index != 2)
        throw std::runtime_error(
            "input and output paths are required");

    options.input = argv[index++];
    options.output = argv[index++];

    return options;
}

}  // namespace

int main(int argc,
         char **argv) {
    try {
        const Options options =
            parse_options(argc, argv);

        if (options.mode ==
            Options::Mode::None) {
            usage(argv[0]);
            return EXIT_FAILURE;
        }

        if (options.mode ==
            Options::Mode::FromHnsw) {
            const Graph graph =
                read_hnsw_native(
                    options.input,
                    options.distance,
                    options.max_levels,
                    options.default_ef_search);

            write_exchange(
                options.output,
                graph);

            std::cout
                << "converted hnswlib index to VecFS exchange\n"
                << "  input:       " << options.input << '\n'
                << "  output:      " << options.output << '\n'
                << "  nodes:       " << graph.nodes.size() << '\n'
                << "  dimension:   " << graph.dimension << '\n'
                << "  distance:    "
                << distance_name(graph.distance_type) << '\n'
                << "  M/M0:        "
                << graph.m << '/' << graph.m0 << '\n'
                << "  max levels:  "
                << graph.max_levels << '\n';

            return EXIT_SUCCESS;
        }

        const Graph exchange =
            read_exchange(options.input);

        write_hnsw_native(
            options.output,
            exchange);

        std::cout
            << "converted VecFS exchange to hnswlib index\n"
            << "  input:       " << options.input << '\n'
            << "  output:      " << options.output << '\n'
            << "  nodes:       " << exchange.nodes.size() << '\n'
            << "  dimension:   " << exchange.dimension << '\n'
            << "  distance:    "
            << distance_name(exchange.distance_type) << '\n'
            << "  M/M0:        "
            << exchange.m << '/' << exchange.m0 << '\n';

        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr
            << "hnswlib-converter: "
            << error.what()
            << '\n';

        usage(argv[0]);
        return EXIT_FAILURE;
    }
}
