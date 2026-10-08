// Single-case host-only MACA module collector. CPU acceptance is a separate process.
#include <mcr/mc_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned kInputWords = 1024, kOutputWords = 384, kGuardWords = 64;
constexpr size_t kNativeImageBytes = 18232, kBundleImageBytes = 15324;
struct BundleEntry {
    const char* target;
    uint64_t offset, size;
};
constexpr BundleEntry kBundleEntries[] = {
    {"host-x86_64-unknown-linux-gnu", 4096, 0},
    {"maca-mxc-metax-macahca--xcore1000-bc", 4096, 11216}
};
constexpr const char* kSymbols[] = {
    "_ZN12_GLOBAL__N_116wmma_tile_kernelEPK6__halfS2_Pfj",
    "_ZN12_GLOBAL__N_118scalar_tile_kernelEPK6__halfS2_Pfj"
};
static_assert(sizeof(void*) == 8 && sizeof(float) == 4 && sizeof(unsigned) == 4,
              "Requires 64-bit pointers, 32-bit float and 32-bit unsigned");
static_assert(sizeof(uint16_t) == 2, "Requires 16-bit input words");

void check_mc(mcError_t status, const char* expression) {
    if (status != mcSuccess)
        throw std::runtime_error(std::string(expression) + ": " + mcGetErrorString(status));
}
#define MC_CHECK(expression) check_mc((expression), #expression)

std::string json_string(const std::string& value) {
    std::ostringstream result;
    result << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') result << '\\' << c;
        else if (c < 32) result << "\\u" << std::hex << std::setw(4)
                                << std::setfill('0') << static_cast<int>(c) << std::dec;
        else result << c;
    }
    result << '"';
    return result.str();
}

std::string environment_value(const char* key) {
    const char* value = std::getenv(key);
    return value ? json_string(value) : "null";
}

template <typename T>
void write_words(const std::string& directory, const std::string& name, const std::vector<T>& words) {
    std::ofstream file(directory + "/" + name, std::ios::binary);
    file.write(reinterpret_cast<const char*>(words.data()), words.size() * sizeof(T));
    file.close();
    if (!file) throw std::runtime_error("Could not retain " + name);
}

void flush_record(std::ostream& records) {
    records << '\n';
    records.flush();
    if (!records) throw std::runtime_error("Could not retain metadata");
}

// These are the four explicit parameters of the retained kernel symbols. The
// runtime owns argument packing and hidden arguments. Installed and published
// examples use kernelParams; the installed header warning conflicts with them.
void launch_once(mcFunction_t function, bool scalar, void* a, void* b, float* output) {
    unsigned chunks = 1;
    void* parameters[] = {&a, &b, &output, &chunks};
    MC_CHECK(mcModuleLaunchKernel(function, 1, 1, 1, scalar ? 256 : 64, 1, 1,
                                 0, nullptr, parameters, nullptr));
    MC_CHECK(mcGetLastError());
    MC_CHECK(mcDeviceSynchronize());
}

// Only fixed, in-range offsets and widths are used after checking the file size.
uint64_t fixed_little_endian(const std::vector<uint8_t>& bytes, size_t offset, unsigned width) {
    uint64_t value = 0;
    for (unsigned index = 0; index < width; ++index)
        value |= static_cast<uint64_t>(bytes[offset + index]) << (8 * index);
    return value;
}

void validate_bundle(const std::vector<uint8_t>& bytes) {
    const char* error = "Module image must be the retained two-entry xcore1000 bitcode bundle";
    if (std::string(reinterpret_cast<const char*>(bytes.data()), 24) != "__CLANG_OFFLOAD_BUNDLE__" ||
        fixed_little_endian(bytes, 24, 8) != 2)
        throw std::runtime_error(error);
    size_t position = 32;
    for (const auto& entry : kBundleEntries) {
        const std::string target(entry.target);
        if (fixed_little_endian(bytes, position, 8) != entry.offset ||
            fixed_little_endian(bytes, position + 8, 8) != entry.size ||
            fixed_little_endian(bytes, position + 16, 8) != target.size() ||
            std::string(reinterpret_cast<const char*>(bytes.data() + position + 24), target.size()) != target)
            throw std::runtime_error(error);
        position += 24 + target.size();
    }
    if (position != 145) throw std::runtime_error(error);
    for (size_t index = position; index < 4096; ++index)
        if (bytes[index] != 0) throw std::runtime_error(error);
    // The payload is the original wrapped bitcode, with its 20-byte wrapper and
    // 12 trailing zero bytes. It is not an arbitrary IR input or native fallback.
    if (fixed_little_endian(bytes, 4096, 4) != 0x0b17c0de ||
        fixed_little_endian(bytes, 4100, 4) != 0 ||
        fixed_little_endian(bytes, 4104, 4) != 20 ||
        fixed_little_endian(bytes, 4108, 4) != 11184 ||
        fixed_little_endian(bytes, 4112, 4) != 255 ||
        bytes[4116] != 'B' || bytes[4117] != 'C' || bytes[4118] != 0xc0 || bytes[4119] != 0xde)
        throw std::runtime_error(error);
    for (size_t index = 15300; index < 15312; ++index)
        if (bytes[index] != 0) throw std::runtime_error(error);
    if (std::string(reinterpret_cast<const char*>(bytes.data() + 15312), 12) != "__FILE_END__")
        throw std::runtime_error(error);
}

std::vector<uint8_t> read_image(const std::string& path, const std::string& kind) {
    const bool native = kind == "native-elf";
    const size_t image_bytes = native ? kNativeImageBytes : kBundleImageBytes;
    std::ifstream image(path, std::ios::binary | std::ios::ate);
    if (!image) throw std::runtime_error("Cannot read module image");
    if (image.tellg() != static_cast<std::streamoff>(image_bytes))
        throw std::runtime_error("Module image must contain exactly " + std::to_string(image_bytes) + " bytes");
    std::vector<uint8_t> bytes(image_bytes);
    image.seekg(0);
    image.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!image || image.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Cannot read module image");
    // These are bounded structural checks. The external CPU checker compares the
    // complete image (and the bundle payload) against separate retained references.
    if (!native) {
        validate_bundle(bytes);
        return bytes;
    }
    if (bytes[0] != 0x7f || bytes[1] != 'E' || bytes[2] != 'L' || bytes[3] != 'F' ||
        bytes[4] != 2 || bytes[5] != 1 || bytes[6] != 1 ||
        bytes[16] != 3 || bytes[17] != 0 || bytes[18] != 253 || bytes[19] != 0 ||
        bytes[20] != 1 || bytes[21] != 0 || bytes[22] != 0 || bytes[23] != 0 ||
        bytes[52] != 64 || bytes[53] != 0)
        throw std::runtime_error("Module image must be the retained ELF64LE ET_DYN image with machine 253");
    return bytes;
}

int function_attribute(mcFunction_t function, mcFunction_attribute attribute) {
    int value = 0;
    MC_CHECK(mcFuncGetAttribute(&value, attribute, function));
    return value;
}

uintptr_t observed_alignment(const void* pointer) {
    const uintptr_t value = reinterpret_cast<uintptr_t>(pointer);
    if (!value) throw std::runtime_error("Device allocation returned a null pointer");
    return value & (~value + 1);
}

void retain_snapshot(const std::string& directory, const std::string& order, const char* phase,
                     const char* after, const char* before, const void* a, const void* b,
                     std::ostream& records) {
    MC_CHECK(mcDeviceSynchronize());
    const void* pointers[] = {a, b};
    const char* operands[] = {"a", "b"};
    for (unsigned index = 0; index < 2; ++index) {
        std::vector<uint16_t> words(kInputWords);
        MC_CHECK(mcMemcpy(words.data(), pointers[index], words.size() * sizeof(uint16_t), mcMemcpyDeviceToHost));
        write_words(directory, std::string(phase) + "." + operands[index] + ".f16", words);
    }
    records << "{\"type\":\"input_snapshot\",\"order\":" << json_string(order)
            << ",\"phase\":" << json_string(phase)
            << ",\"after_variant\":" << (after ? json_string(after) : "null")
            << ",\"before_variant\":" << (before ? json_string(before) : "null")
            << ",\"operand_halfwords_each\":1024,\"a_file\":" << json_string(std::string(phase) + ".a.f16")
            << ",\"b_file\":" << json_string(std::string(phase) + ".b.f16") << '}';
    flush_record(records);
}

void collect(const std::string& order, const std::string& kind, const std::string& image_path,
             const std::string& directory) {
    if (order != "wmma-first" && order != "scalar-first")
        throw std::runtime_error("Order must be wmma-first or scalar-first");
    if (kind != "native-elf" && kind != "retained-bitcode-bundle")
        throw std::runtime_error("Image kind must be native-elf or retained-bitcode-bundle");
    if (!std::filesystem::is_directory(directory) || !std::filesystem::is_empty(directory))
        throw std::runtime_error("Output directory must already exist and be empty");
    const uint16_t endian = 1;
    if (*reinterpret_cast<const uint8_t*>(&endian) != 1)
        throw std::runtime_error("Requires a little-endian host");

    const std::vector<uint8_t> image = read_image(image_path, kind);
    write_words(directory, "loaded-image.bin", image);

    // Fixed raw words; no arithmetic-based or observed-output-based input generation.
    std::vector<uint16_t> a(kInputWords, 0), b(kInputWords, 0);
    a[0] = 0xb700; a[1] = 0x2c00;
    b[0] = 0xac00; b[1] = 0xb800;
    write_words(directory, "prepared.a.f16", a);
    write_words(directory, "prepared.b.f16", b);
    std::ofstream records(directory + "/raw.jsonl");
    if (!records) throw std::runtime_error("Cannot create raw.jsonl");

    int count = 0;
    MC_CHECK(mcGetDeviceCount(&count));
    if (count != 1) throw std::runtime_error("Exactly one leased MACA device must be visible");
    MC_CHECK(mcSetDevice(0));
    mcDeviceProp_t properties{};
    MC_CHECK(mcGetDeviceProperties(&properties, 0));
    if (std::string(properties.name) != "MetaX C550" || properties.waveSize != 64)
        throw std::runtime_error("Expected exact device MetaX C550 with observed wave size 64");
    if (properties.maxThreadsPerBlock < 256)
        throw std::runtime_error("Device block limit is below 256");
    char pci[64]{};
    int runtime = 0, driver = 0;
    MC_CHECK(mcDeviceGetPCIBusId(pci, sizeof(pci), 0));
    MC_CHECK(mcRuntimeGetVersion(&runtime));
    MC_CHECK(mcDriverGetVersion(&driver));
    records << "{\"type\":\"device\",\"name\":" << json_string(properties.name)
            << ",\"logical_device\":0,\"visible_device_count\":" << count
            << ",\"wave_size_api\":" << properties.waveSize << ",\"pci_bus_id\":" << json_string(pci)
            << ",\"runtime_version_api\":" << runtime << ",\"driver_version_api\":" << driver
            << ",\"max_threads_per_block\":" << properties.maxThreadsPerBlock << '}';
    flush_record(records);

    mcModule_t module = nullptr;
    MC_CHECK(mcModuleLoadData(&module, image.data()));
    records << "{\"type\":\"module\",\"image_file\":\"loaded-image.bin\",\"image_kind\":" << json_string(kind)
            << ",\"image_bytes\":" << image.size();
    if (kind == "native-elf") {
        records << ",\"image_format\":\"ELF64LE\",\"machine_raw\":253";
    } else {
        records << ",\"image_format\":\"CLANG_OFFLOAD_BUNDLE\",\"bundle_entries\":[";
        for (unsigned index = 0; index < 2; ++index) {
            const auto& entry = kBundleEntries[index];
            if (index) records << ',';
            records << "{\"target\":" << json_string(entry.target)
                    << ",\"offset\":" << entry.offset << ",\"size\":" << entry.size << '}';
        }
        records << "],\"wrapped_bitcode_bytes\":11216,\"inner_bitcode_bytes\":11184";
    }
    records << ",\"load_api\":\"mcModuleLoadData\","
               "\"launch_api\":\"mcModuleLaunchKernel\",\"argument_interface\":\"kernelParams\","
               "\"extra_is_null\":true,\"symbols\":{\"wmma\":" << json_string(kSymbols[0])
            << ",\"scalar\":" << json_string(kSymbols[1]) << "},\"module_loaded\":true}";
    flush_record(records);
    mcFunction_t functions[2] = {nullptr, nullptr};
    for (unsigned slot = 0; slot < 2; ++slot)
        MC_CHECK(mcModuleGetFunction(&functions[slot], module, kSymbols[slot]));

    const char* variants[] = {order == "wmma-first" ? "wmma" : "scalar", order == "wmma-first" ? "scalar" : "wmma"};
    records << "{\"type\":\"protocol\",\"schema\":\"metax-kernelwiki.wmma-module-q7.v2\",\"order\":" << json_string(order)
            << ",\"image_kind\":" << json_string(kind)
            << ",\"variant_order\":[" << json_string(variants[0]) << ',' << json_string(variants[1]) << "],"
               "\"logical_shape\":[16,16,2],\"tile\":[16,16,16],\"packed_chunks\":4,\"k_chunks\":1,"
               "\"operand_halfwords_each\":1024,\"input_words_prefix\":{\"a\":[46848,11264],\"b\":[44032,47104]},"
               "\"input_padding_uint16\":0,\"prepared_files\":{\"a\":\"prepared.a.f16\",\"b\":\"prepared.b.f16\"},"
               "\"layouts\":{\"a\":\"row_major\",\"b\":\"col_major\",\"c\":\"row_major\"},\"leading_dimension\":16,"
               "\"operand_dtype\":\"float16\",\"accumulator_dtype\":\"float32\",\"output_dtype\":\"float32\","
               "\"output_elements\":256,\"guard_elements_each_side\":64,\"guard_uint32\":4294967295,"
               "\"initial_payload_uint32\":4294967295,\"launches_per_variant\":1,\"total_launches\":2,\"warmups\":0,"
               "\"input_rewrite_between_variants\":false,\"snapshot_phases\":[\"before\",\"between\",\"after\"],"
               "\"comparison\":\"all 256 outputs finite and numerically exact; signed zeros equivalent; no tolerance\",\"environment\":{";
    const char* environment[] = {"MACA_LAUNCH_MODE", "MACA_LAUNCH_BLOCKING", "MACA_DIRECT_DISPATCH", "MACA_CACHE_PATH", "MACA_CACHE_DISABLE", "MACA_MODULE_LOADING"};
    for (unsigned index = 0; index < 6; ++index) {
        if (index) records << ',';
        records << json_string(environment[index]) << ':' << environment_value(environment[index]);
    }
    records << "}}";
    flush_record(records);

    void *device_a = nullptr, *device_b = nullptr;
    float* device_c[2] = {nullptr, nullptr};  // Fixed slots: WMMA=0, scalar=1, independent of order.
    MC_CHECK(mcMalloc(&device_a, a.size() * sizeof(uint16_t)));
    MC_CHECK(mcMalloc(&device_b, b.size() * sizeof(uint16_t)));
    for (auto& pointer : device_c)
        MC_CHECK(mcMalloc(reinterpret_cast<void**>(&pointer), kOutputWords * sizeof(float)));
    MC_CHECK(mcMemcpy(device_a, a.data(), a.size() * sizeof(uint16_t), mcMemcpyHostToDevice));
    MC_CHECK(mcMemcpy(device_b, b.data(), b.size() * sizeof(uint16_t), mcMemcpyHostToDevice));
    retain_snapshot(directory, order, "before", nullptr, variants[0], device_a, device_b, records);

    for (unsigned stage = 0; stage < 2; ++stage) {
        const bool scalar = std::string(variants[stage]) == "scalar";
        const unsigned slot = scalar ? 1 : 0;
        float* base = device_c[slot];
        float* payload = base + kGuardWords;
        MC_CHECK(mcMemset(base, 0xff, kOutputWords * sizeof(float)));
        const mcFunction_t function = functions[slot];
        const int max_threads = function_attribute(function, MC_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
        const int num_regs = function_attribute(function, MC_FUNC_ATTRIBUTE_NUM_REGS);
        const int shared_bytes = function_attribute(function, MC_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
        const int local_bytes = function_attribute(function, MC_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);
        const uintptr_t aa = observed_alignment(device_a), ba = observed_alignment(device_b), ca = observed_alignment(payload);
        launch_once(function, scalar, device_a, device_b, payload);
        std::vector<uint32_t> output(kOutputWords);
        MC_CHECK(mcMemcpy(output.data(), base, output.size() * sizeof(uint32_t), mcMemcpyDeviceToHost));
        write_words(directory, std::string(variants[stage]) + ".f32", output);
        records << "{\"type\":\"variant\",\"order\":" << json_string(order)
                << ",\"variant\":" << json_string(variants[stage]) << ",\"kernel\":" << json_string(std::string(variants[stage]) + "_tile_kernel")
                << ",\"launched_symbol\":" << json_string(kSymbols[slot]) << ",\"argument_count\":4"
                << ",\"block\":[" << (scalar ? 256 : 64) << ",1,1],\"grid\":[1,1,1],\"k_chunks\":1,\"launch_count\":1,\"c_allocation_slot\":" << slot
                << ",\"output_file\":" << json_string(std::string(variants[stage]) + ".f32")
                << ",\"output_words\":384,\"payload_offset_words\":64,\"payload_elements\":256,\"guard_elements_each_side\":64,"
                   "\"function_attributes_before_launch\":{\"maxThreadsPerBlock\":" << max_threads
                << ",\"numRegs\":" << num_regs << ",\"sharedSizeBytes\":" << shared_bytes << ",\"localSizeBytes\":" << local_bytes << "},"
                   "\"pointer_alignment_observed_bytes\":{\"a\":" << aa << ",\"b\":" << ba << ",\"c_payload\":" << ca << "}}";
        flush_record(records);
        retain_snapshot(directory, order, stage == 0 ? "between" : "after", variants[stage],
                        stage == 0 ? variants[1] : nullptr, device_a, device_b, records);
    }
    for (auto pointer : device_c) MC_CHECK(mcFree(pointer));
    MC_CHECK(mcFree(device_b));
    MC_CHECK(mcFree(device_a));
    MC_CHECK(mcDeviceSynchronize());
    MC_CHECK(mcModuleUnload(module));
    records << "{\"type\":\"complete\",\"order\":" << json_string(order)
            << ",\"image_kind\":" << json_string(kind)
            << ",\"variant_count\":2,\"total_launches\":2,\"outputs_retained\":true,\"device_buffers_freed\":true,\"module_unloaded\":true,\"cpu_correctness_checked\":false}";
    flush_record(records);
    records.close();
    if (!records) throw std::runtime_error("Could not finish metadata");
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 6 || std::string(argv[1]) != "--run") {
        std::cerr << "Usage: repro --run {wmma-first|scalar-first} {native-elf|retained-bitcode-bundle} IMAGE FRESH_EMPTY_OUTPUT_DIRECTORY\n";
        return 2;
    }
    try {
        collect(argv[2], argv[3], argv[4], argv[5]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "{\"type\":\"error\",\"message\":" << json_string(error.what()) << "}\n";
        return 1;
    }
}
