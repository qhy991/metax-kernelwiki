// Version-bound native MACA WMMA fragment probe; no cu-bridge or NVIDIA alias.
#include <mcr/mc_runtime.h>
#include <__clang_maca_mma_functions.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef C550_WMMA_CONTROL
#define C550_WMMA_CONTROL 0
#endif
#if C550_WMMA_CONTROL != 0 && C550_WMMA_CONTROL != 1
#error "C550_WMMA_CONTROL must be 0 or 1"
#endif
#ifndef C550_WMMA_PREFIX
#define C550_WMMA_PREFIX 0
#endif
#if C550_WMMA_PREFIX != 0 && C550_WMMA_PREFIX != 1
#error "C550_WMMA_PREFIX must be 0 or 1"
#endif
#if C550_WMMA_PREFIX && !C550_WMMA_CONTROL
#error "C550_WMMA_PREFIX requires paired scalar control"
#endif
#ifndef C550_WMMA_WITNESS
#define C550_WMMA_WITNESS 0
#endif
#if C550_WMMA_WITNESS < 0 || C550_WMMA_WITNESS > 5
#error "C550_WMMA_WITNESS must be 0, 1, 2, 3, 4 or 5"
#endif
#if C550_WMMA_WITNESS && (!C550_WMMA_CONTROL || C550_WMMA_PREFIX)
#error "C550_WMMA_WITNESS requires C550_WMMA_CONTROL=1 and C550_WMMA_PREFIX=0"
#endif

namespace {
constexpr unsigned kOperandHalfwords = 1024;
constexpr unsigned kOutputWords = 256;
constexpr unsigned kGuardWords = 64;
constexpr const char* kExperiment = C550_WMMA_WITNESS == 5 ? "wmma-scalar-fp32-scale-control"
                                 : C550_WMMA_WITNESS == 4 ? "wmma-scalar-fp32-magnitude-control"
                                 : C550_WMMA_WITNESS == 3 ? "wmma-scalar-fp32-sign-control"
                                 : C550_WMMA_WITNESS == 2 ? "wmma-scalar-fp32-product-control"
                                 : C550_WMMA_WITNESS == 1 ? "wmma-scalar-fp32-witness-control"
                                 : C550_WMMA_PREFIX ? "wmma-scalar-fp32-prefix-control"
                                 : C550_WMMA_CONTROL ? "wmma-scalar-fp32-input-control" : "native-wmma-fp16-fp32-16x16";
static_assert(sizeof(__half) == 2 && sizeof(float) == 4, "Probe requires FP16 operands and FP32 output");
#if C550_WMMA_WITNESS
constexpr const char* kPatternLabel = C550_WMMA_WITNESS == 5 ? "scale" : C550_WMMA_WITNESS == 4 ? "magnitude" : C550_WMMA_WITNESS == 3 ? "sign" : C550_WMMA_WITNESS == 2 ? "product" : "witness";
constexpr unsigned kPatternMaximum = C550_WMMA_WITNESS == 5 ? 45 : C550_WMMA_WITNESS == 4 ? 42 : C550_WMMA_WITNESS == 3 ? 8 : C550_WMMA_WITNESS == 2 ? 6 : 3;
#endif

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

std::string json_environment(const char* key) {
    const char* value = std::getenv(key);
    return value ? json_string(value) : "null";
}

struct Case {
    std::string id;
#if C550_WMMA_CONTROL
    std::string order;
#endif
#if C550_WMMA_PREFIX
    std::string suite, family;
#endif
#if C550_WMMA_WITNESS
    std::string suite, pattern, input_rule;
    unsigned target_row = 0, target_col = 0;
#endif
#if C550_WMMA_WITNESS >= 4
    unsigned q = 0;
    std::string role;
#endif
#if C550_WMMA_WITNESS == 5
    int scale_exp = 0;
#endif
    unsigned m = 0, n = 0, k = 0, warmups = 0, samples = 0, launches = 0;
    std::vector<uint16_t> a, b;
};

#if C550_WMMA_WITNESS >= 2
struct ProductPattern {
    const char* name; int a[2], b[2]; const char* matching_product_pattern = nullptr;
    unsigned q = 0; const char* role = nullptr; const char* matching_sign_pattern = nullptr;
    int scale_exp = 0; const char* matching_magnitude_pattern = nullptr;
};
#if C550_WMMA_WITNESS == 2
constexpr ProductPattern kProductPatterns[] = {
    {"positive-k0", {-12,0}, {-1,0}}, {"positive-k1", {0,-12}, {0,-1}},
    {"negative-k0", {1,0}, {-13,0}}, {"negative-k1", {0,1}, {0,-13}},
    {"pair-forward", {-12,1}, {-1,-13}}, {"pair-reversed", {1,-12}, {-13,-1}}
};
#elif C550_WMMA_WITNESS == 3
constexpr ProductPattern kProductPatterns[] = {
    {"positive12-k0", {-12,0}, {-1,0}, "positive-k0"}, {"negative12-k0", {12,0}, {-1,0}},
    {"positive13-k1", {0,-1}, {0,-13}}, {"negative13-k1", {0,1}, {0,-13}, "negative-k1"},
    {"pair-pp", {-12,-1}, {-1,-13}}, {"pair-pn", {-12,1}, {-1,-13}, "pair-forward"},
    {"pair-np", {12,-1}, {-1,-13}}, {"pair-nn", {12,1}, {-1,-13}}
};
#elif C550_WMMA_WITNESS == 4
// A closed 42-entry table, independent of user-supplied q or role.
#define MAGNITUDE_ROW(Q, NAME) \
    {NAME "-positive", {-(Q),0}, {-1,0}, nullptr, Q, "positive", Q == 12 ? "positive12-k0" : nullptr}, \
    {NAME "-negative", {0,1}, {0,-((Q)+1)}, nullptr, Q, "negative", Q == 12 ? "negative13-k1" : nullptr}, \
    {NAME "-pair", {-(Q),1}, {-1,-((Q)+1)}, nullptr, Q, "pair", Q == 12 ? "pair-pn" : nullptr},
constexpr ProductPattern kProductPatterns[] = {
    MAGNITUDE_ROW(1, "q01") MAGNITUDE_ROW(2, "q02") MAGNITUDE_ROW(3, "q03")
    MAGNITUDE_ROW(4, "q04") MAGNITUDE_ROW(5, "q05") MAGNITUDE_ROW(6, "q06")
    MAGNITUDE_ROW(7, "q07") MAGNITUDE_ROW(8, "q08") MAGNITUDE_ROW(9, "q09")
    MAGNITUDE_ROW(10, "q10") MAGNITUDE_ROW(11, "q11") MAGNITUDE_ROW(12, "q12")
    MAGNITUDE_ROW(13, "q13") MAGNITUDE_ROW(14, "q14")
};
#undef MAGNITUDE_ROW
#else
#define SCALE_ROW(Q, NAME, E, TAG, POSITIVE_MATCH) \
    {NAME "-" TAG "-positive", {-(Q),0}, {-1,0}, nullptr, Q, "positive", nullptr, E, POSITIVE_MATCH}, \
    {NAME "-" TAG "-negative", {0,1}, {0,-((Q)+1)}, nullptr, Q, "negative", nullptr, E, E == 0 ? NAME "-negative" : nullptr}, \
    {NAME "-" TAG "-pair", {-(Q),1}, {-1,-((Q)+1)}, nullptr, Q, "pair", nullptr, E, E == 0 ? NAME "-pair" : nullptr},
constexpr ProductPattern kProductPatterns[] = {
    SCALE_ROW(6, "q06", -2, "em2", nullptr) SCALE_ROW(6, "q06", -1, "em1", "q03-positive")
    SCALE_ROW(6, "q06", 0, "e0", "q06-positive") SCALE_ROW(6, "q06", 1, "ep1", "q12-positive")
    SCALE_ROW(6, "q06", 2, "ep2", nullptr)
    SCALE_ROW(7, "q07", -2, "em2", nullptr) SCALE_ROW(7, "q07", -1, "em1", nullptr)
    SCALE_ROW(7, "q07", 0, "e0", "q07-positive") SCALE_ROW(7, "q07", 1, "ep1", "q14-positive")
    SCALE_ROW(7, "q07", 2, "ep2", nullptr)
    SCALE_ROW(12, "q12", -2, "em2", "q03-positive") SCALE_ROW(12, "q12", -1, "em1", "q06-positive")
    SCALE_ROW(12, "q12", 0, "e0", "q12-positive") SCALE_ROW(12, "q12", 1, "ep1", nullptr)
    SCALE_ROW(12, "q12", 2, "ep2", nullptr)
};
#undef SCALE_ROW
#endif
constexpr const char* kOrderedSuite = C550_WMMA_WITNESS == 5 ? "scale-control" : C550_WMMA_WITNESS == 4 ? "magnitude-control" : C550_WMMA_WITNESS == 3 ? "sign-control" : "product-control";
constexpr const char* kOrderedRule = C550_WMMA_WITNESS == 5 ? "isolated-a-power-of-two-scale" : C550_WMMA_WITNESS == 4 ? "isolated-adjacent-magnitudes" : C550_WMMA_WITNESS == 3 ? "isolated-signed-products" : "isolated-ordered-products";

const ProductPattern* product_pattern(const std::string& name) {
    for (const auto& pattern : kProductPatterns)
        if (name == pattern.name) return &pattern;
    return nullptr;
}

void product_terms(std::ostream& records, const ProductPattern& pattern) {
    const int scale_num = C550_WMMA_WITNESS == 5 ? 1 << (pattern.scale_exp > 0 ? pattern.scale_exp : 0) : 1;
    const int a0 = pattern.a[0] * scale_num, a1 = pattern.a[1] * scale_num;
    const int p0 = a0 * pattern.b[0], p1 = a1 * pattern.b[1];
    records << ",\"k_slots\":[0,1],\"a_numerators\":[" << a0 << ',' << a1
            << "],\"b_numerators\":[" << pattern.b[0] << ',' << pattern.b[1]
            << "],\"product_numerators\":[" << p0 << ',' << p1 << "],\"nonzero_product_k_slots\":[";
    if (p0) records << '0';
    if (p0 && p1) records << ',';
    if (p1) records << '1';
    records << "],\"target_reference_numerator\":" << p0 + p1;
#if C550_WMMA_WITNESS >= 3
    records << ",\"product_signs\":[" << ((p0 > 0) - (p0 < 0)) << ',' << ((p1 > 0) - (p1 < 0))
            << "],\"product_magnitudes\":[" << std::abs(p0) << ',' << std::abs(p1) << ']';
#if C550_WMMA_WITNESS == 3
    records << ",\"matching_product_pattern\":"
            << (pattern.matching_product_pattern ? json_string(pattern.matching_product_pattern) : "null");
#elif C550_WMMA_WITNESS == 4
    records << ",\"q\":" << pattern.q << ",\"role\":" << json_string(pattern.role)
            << ",\"matching_sign_pattern\":"
            << (pattern.matching_sign_pattern ? json_string(pattern.matching_sign_pattern) : "null");
#else
    const int scale_den = 1 << (pattern.scale_exp < 0 ? -pattern.scale_exp : 0);
    records << ",\"q\":" << pattern.q << ",\"scale_exp\":" << pattern.scale_exp << ",\"role\":" << json_string(pattern.role)
            << ",\"a_base_numerators\":[" << pattern.a[0] << ',' << pattern.a[1] << ']'
            << ",\"scale_numerator\":" << scale_num << ",\"scale_denominator\":" << scale_den
            << ",\"a_denominator\":" << 16 * scale_den << ",\"b_denominator\":16"
            << ",\"product_denominator\":" << 256 * scale_den << ",\"target_reference_denominator\":" << 256 * scale_den
            << ",\"matching_magnitude_pattern\":"
            << (pattern.matching_magnitude_pattern ? json_string(pattern.matching_magnitude_pattern) : "null");
#endif
#endif
}
#endif

bool admitted_shape(const Case& c) {
#if C550_WMMA_WITNESS >= 2
    const auto* pattern = product_pattern(c.pattern);
    return c.suite == kOrderedSuite && c.m == 16 && c.n == 16 && c.k == 2
        && c.target_row == 0 && c.target_col == 0 && c.input_rule == kOrderedRule
        && pattern != nullptr
#if C550_WMMA_WITNESS >= 4
        && c.q == pattern->q && c.role == pattern->role
#endif
#if C550_WMMA_WITNESS == 5
        && c.scale_exp == pattern->scale_exp
#endif
        ;
#elif C550_WMMA_WITNESS == 1
    if (c.suite != "witness-control" || c.m != 16 || c.n != 16 || c.k != 2) return false;
    if (c.pattern == "dense-origin")
        return c.target_row == 13 && c.target_col == 2 && c.input_rule == "dense-formulas";
    if (c.pattern == "isolated-origin")
        return c.target_row == 13 && c.target_col == 2 && c.input_rule == "isolated-fixed-pairs";
    if (c.pattern == "isolated-c00")
        return c.target_row == 0 && c.target_col == 0 && c.input_rule == "isolated-fixed-pairs";
    return false;
#elif C550_WMMA_PREFIX
    return c.suite == "prefix-control" && c.k <= 16
        && ((c.family == "singleton" && c.m == 1 && c.n == 1)
            || (c.family == "dense" && c.m == 16 && c.n == 16));
#else
    const unsigned shapes[][3] = {{16,16,0},{1,1,1},{16,16,16},{15,16,16},{16,15,16},{15,15,15},
                                 {7,9,17},{15,16,31},{16,15,32},{16,16,33},{9,7,63},{16,16,64}};
    for (const auto& shape : shapes)
        if (c.m == shape[0] && c.n == shape[1] && c.k == shape[2]) return true;
    return false;
#endif
}

#if C550_WMMA_WITNESS
void witness_fields(std::ostream& records, const Case& c) {
    records << ",\"suite\":" << json_string(c.suite) << ",\"pattern\":" << json_string(c.pattern)
            << ",\"target_row\":" << c.target_row << ",\"target_col\":" << c.target_col
            << ",\"input_rule\":" << json_string(c.input_rule);
#if C550_WMMA_WITNESS >= 2
    product_terms(records, *product_pattern(c.pattern));
#endif
}
#endif

uint16_t sixteenth_bits(int numerator) {
    // Exact normal half encoding for this contract's integer numerators in [-15,15].
    if (!numerator) return 0;
    const unsigned magnitude = numerator < 0 ? -numerator : numerator;
    unsigned exponent = 0;
    while ((1U << (exponent + 1)) <= magnitude) ++exponent;
    return static_cast<uint16_t>((numerator < 0 ? 0x8000 : 0)
           | ((exponent + 11) << 10) | ((magnitude - (1U << exponent)) << (10 - exponent)));
}

#if C550_WMMA_WITNESS == 5
uint16_t scaled_sixteenth_bits(int numerator, int scale_exp) {
    if (numerator < -15 || numerator > 15 || scale_exp < -2 || scale_exp > 2)
        throw std::runtime_error("Scaled FP16 input outside the fixed dyadic contract");
    const uint16_t base = sixteenth_bits(numerator);
    if (!base) return 0;
    const int exponent = static_cast<int>((base >> 10) & 31) + scale_exp;
    if (exponent <= 0 || exponent >= 31)
        throw std::runtime_error("Scaled FP16 input is outside the normal finite domain");
    return static_cast<uint16_t>((base & 0x83ffU) | (static_cast<unsigned>(exponent) << 10));
}
#endif

std::vector<uint16_t> read_operand(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<uint16_t> words(kOperandHalfwords);
    if (!file.read(reinterpret_cast<char*>(words.data()), words.size() * sizeof(uint16_t))
        || file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Missing or incorrect packed operand file extent");
    return words;
}

void validate_inputs(const Case& c) {
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        for (unsigned outer = 0; outer < 16; ++outer) {
            for (unsigned inner = 0; inner < 16; ++inner) {
                const unsigned k = chunk * 16 + inner;
                const unsigned index = chunk * 256 + outer * 16 + inner;
                int a_num = outer < c.m && k < c.k ? static_cast<int>((67 * outer + 13 * k) % 31) - 15 : 0;
                int b_num = outer < c.n && k < c.k ? static_cast<int>((17 * k + 5 * outer + 3) % 29) - 14 : 0;
#if C550_WMMA_WITNESS >= 2
                const auto& pattern = *product_pattern(c.pattern);  // Admission already resolved the closed pattern.
                a_num = outer == 0 && k < 2 ? pattern.a[k] : 0;
                b_num = outer == 0 && k < 2 ? pattern.b[k] : 0;
#elif C550_WMMA_WITNESS == 1
                if (c.pattern != "dense-origin") {
                    const int a_pair[] = {-12, 1}, b_pair[] = {-1, -13};
                    a_num = outer == c.target_row && k < 2 ? a_pair[k] : 0;
                    b_num = outer == c.target_col && k < 2 ? b_pair[k] : 0;
                }
#endif
#if C550_WMMA_WITNESS == 5
                const uint16_t a_bits = scaled_sixteenth_bits(a_num, c.scale_exp);
#else
                const uint16_t a_bits = sixteenth_bits(a_num);
#endif
                if (c.a[index] != a_bits || c.b[index] != sixteenth_bits(b_num))
                    throw std::runtime_error("Packed input words differ from the declared A-row/B-column contract");
            }
        }
    }
}

std::vector<Case> read_plan(const std::string& directory) {
    std::ifstream file(directory + "/cases.tsv");
    std::string line;
    const char* header = C550_WMMA_WITNESS == 5 ? "id\tm\tn\tk\tsuite\tpattern\tq\tscale_exp\trole\ttarget_row\ttarget_col\tinput_rule\torder\twarmups\tsamples\tlaunches"
                        : C550_WMMA_WITNESS == 4 ? "id\tm\tn\tk\tsuite\tpattern\tq\trole\ttarget_row\ttarget_col\tinput_rule\torder\twarmups\tsamples\tlaunches"
                        : C550_WMMA_WITNESS ? "id\tm\tn\tk\tsuite\tpattern\ttarget_row\ttarget_col\tinput_rule\torder\twarmups\tsamples\tlaunches"
                        : C550_WMMA_PREFIX ? "id\tm\tn\tk\tsuite\tfamily\torder\twarmups\tsamples\tlaunches"
                        : C550_WMMA_CONTROL ? "id\tm\tn\tk\torder\twarmups\tsamples\tlaunches"
                                         : "id\tm\tn\tk\twarmups\tsamples\tlaunches";
    if (!file || !std::getline(file, line) || line != header)
        throw std::runtime_error("Unsupported or missing cases.tsv");
    std::vector<Case> cases;
    while (std::getline(file, line)) {
#if C550_WMMA_WITNESS
        if (cases.size() >= kPatternMaximum)
            throw std::runtime_error("Maximum " + std::to_string(kPatternMaximum) + " " + kPatternLabel + " cases");
#endif
        std::istringstream row(line);
        Case c;
        std::string trailing;
#if C550_WMMA_CONTROL
#if C550_WMMA_WITNESS == 5
        if (!(row >> c.id >> c.m >> c.n >> c.k >> c.suite >> c.pattern >> c.q >> c.scale_exp >> c.role >> c.target_row >> c.target_col
                  >> c.input_rule >> c.order >> c.warmups >> c.samples >> c.launches) || (row >> trailing))
            throw std::runtime_error("Invalid scale-control case row");
#elif C550_WMMA_WITNESS == 4
        if (!(row >> c.id >> c.m >> c.n >> c.k >> c.suite >> c.pattern >> c.q >> c.role >> c.target_row >> c.target_col
                  >> c.input_rule >> c.order >> c.warmups >> c.samples >> c.launches) || (row >> trailing))
            throw std::runtime_error("Invalid magnitude-control case row");
#elif C550_WMMA_WITNESS
        if (!(row >> c.id >> c.m >> c.n >> c.k >> c.suite >> c.pattern >> c.target_row >> c.target_col
                  >> c.input_rule >> c.order >> c.warmups >> c.samples >> c.launches) || (row >> trailing))
            throw std::runtime_error(std::string("Invalid ") + kPatternLabel + "-control case row");
#elif C550_WMMA_PREFIX
        if (!(row >> c.id >> c.m >> c.n >> c.k >> c.suite >> c.family >> c.order >> c.warmups >> c.samples >> c.launches) || (row >> trailing))
            throw std::runtime_error("Invalid prefix-control case row");
#else
        if (!(row >> c.id >> c.m >> c.n >> c.k >> c.order >> c.warmups >> c.samples >> c.launches) || (row >> trailing))
            throw std::runtime_error("Invalid control case row");
#endif
        if (c.order != "wmma-first" && c.order != "scalar-first")
            throw std::runtime_error("Unknown WMMA/scalar variant order");
#else
        if (!(row >> c.id >> c.m >> c.n >> c.k >> c.warmups >> c.samples >> c.launches) || (row >> trailing))
            throw std::runtime_error("Invalid case row");
#endif
        if (c.id.empty() || c.id.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-") != std::string::npos)
            throw std::runtime_error("Unsafe case id");
        for (const auto& old : cases) {
            if (old.id == c.id) throw std::runtime_error("Duplicate case id");
#if C550_WMMA_WITNESS
            if (old.pattern == c.pattern)
                throw std::runtime_error(std::string("Duplicate ") + kPatternLabel + " pattern");
#endif
        }
        if (!admitted_shape(c)) throw std::runtime_error(C550_WMMA_WITNESS == 5
            ? "Shape, q, scale_exp, role, pattern, target or input rule outside the fixed WMMA scale cases" : C550_WMMA_WITNESS == 4
            ? "Shape, q, role, pattern, target or input rule outside the fixed WMMA magnitude cases" : C550_WMMA_WITNESS == 3
            ? "Shape, pattern, target or input rule outside the fixed WMMA sign cases" : C550_WMMA_WITNESS == 2
            ? "Shape, pattern, target or input rule outside the fixed WMMA product cases" : C550_WMMA_WITNESS == 1
            ? "Shape, pattern, target or input rule outside the fixed WMMA witness cases" : C550_WMMA_PREFIX
            ? "Shape, suite or family outside the fixed WMMA prefix cases"
            : "Shape outside the fixed WMMA boundary cases");
        if (c.warmups != 10 || c.samples != 10 || c.launches != 10)
            throw std::runtime_error("Probe fixes 10 warmups and 10 samples of 10 launches");
        c.a = read_operand(directory + "/" + c.id + ".a.f16");
        c.b = read_operand(directory + "/" + c.id + ".b.f16");
        validate_inputs(c);
        cases.push_back(std::move(c));
#if !C550_WMMA_WITNESS
        if (cases.size() > (C550_WMMA_PREFIX ? 34U : 12U))
            throw std::runtime_error(C550_WMMA_PREFIX ? "Maximum 34 prefix cases" : "Maximum 12 cases");
#endif
    }
    if (!file.eof() || cases.empty()) throw std::runtime_error("Invalid or empty plan");
    return cases;
}

__global__ void wmma_tile_kernel(const __half* a, const __half* b, float* output, unsigned chunks) {
    using namespace mxmaca::wmma;
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_fragment;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b_fragment;
    fragment<accumulator, 16, 16, 16, float> accumulator_fragment;
    fill_fragment(accumulator_fragment, 0.0f);
    // Every physical thread follows the same loop, including the zero-chunk case.
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        load_matrix_sync(a_fragment, a + chunk * 256, 16);
        load_matrix_sync(b_fragment, b + chunk * 256, 16);
        mma_sync(accumulator_fragment, a_fragment, b_fragment, accumulator_fragment);
    }
    store_matrix_sync(output, accumulator_fragment, 16, mem_row_major);
}

#if C550_WMMA_CONTROL
__global__ void scalar_tile_kernel(const __half* a, const __half* b, float* output, unsigned chunks) {
    const unsigned index = threadIdx.x;
    const unsigned row = index / 16, col = index % 16;
    float total = 0.0f;
    for (unsigned k = 0; k < chunks * 16; ++k) {
        const unsigned chunk = k / 16, local_k = k % 16;
        const float left = __half2float(a[chunk * 256 + row * 16 + local_k]);
        const float right = __half2float(b[chunk * 256 + col * 16 + local_k]);
        total += left * right;
    }
    output[index] = total;
}
#endif

void launch(const Case& c, const __half* a, const __half* b, float* output, bool scalar = false) {
#if C550_WMMA_CONTROL
    if (scalar) scalar_tile_kernel<<<1, 256>>>(a, b, output, (c.k + 15) / 16);
    else
#endif
    wmma_tile_kernel<<<1, 64>>>(a, b, output, (c.k + 15) / 16);
    MC_CHECK(mcGetLastError());
}

uintptr_t observed_alignment(const void* pointer) {
    const uintptr_t value = reinterpret_cast<uintptr_t>(pointer);
    if (!value) throw std::runtime_error("Device allocation returned a null pointer");
    return value & (~value + 1);
}

#if C550_WMMA_CONTROL
void retain_inputs(const Case& c, const char* phase, const __half* a, const __half* b,
                   const std::string& directory, std::ostream& records) {
    MC_CHECK(mcDeviceSynchronize());
    const __half* pointers[] = {a, b};
    const char* labels[] = {"a", "b"};
    for (unsigned operand = 0; operand < 2; ++operand) {
        std::vector<uint16_t> words(kOperandHalfwords);
        MC_CHECK(mcMemcpy(words.data(), pointers[operand], words.size() * sizeof(uint16_t), mcMemcpyDeviceToHost));
        const std::string name = c.id + "." + phase + "." + labels[operand] + ".f16";
        std::ofstream file(directory + "/" + name, std::ios::binary);
        file.write(reinterpret_cast<const char*>(words.data()), words.size() * sizeof(uint16_t));
        file.close();
        if (!file) throw std::runtime_error("Could not retain device input snapshot");
    }
    records << "{\"type\":\"input_snapshot\",\"id\":" << json_string(c.id)
            << ",\"phase\":" << json_string(phase) << ",\"order\":" << json_string(c.order);
#if C550_WMMA_PREFIX
    records << ",\"suite\":" << json_string(c.suite) << ",\"family\":" << json_string(c.family);
#endif
#if C550_WMMA_WITNESS
    witness_fields(records, c);
#endif
    records << ",\"operand_halfwords_each\":1024,\"a_file\":" << json_string(c.id + "." + phase + ".a.f16")
            << ",\"b_file\":" << json_string(c.id + "." + phase + ".b.f16") << "}\n";
    records.flush();
}
#endif

void run(const std::string& input_directory, const std::string& output_directory) {
    const uint16_t endian = 1;
    if (*reinterpret_cast<const uint8_t*>(&endian) != 1)
        throw std::runtime_error("Probe requires a little-endian host");
    // Complete host file admission and packing validation precede every device API.
    const auto cases = read_plan(input_directory);
    if (!std::filesystem::is_directory(output_directory) || !std::filesystem::is_empty(output_directory))
        throw std::runtime_error("Output directory must already exist and be empty");
    std::ofstream records(output_directory + "/raw.jsonl");
    if (!records) throw std::runtime_error("Cannot create raw.jsonl");
    records << std::setprecision(12);
    int count = 0;
    MC_CHECK(mcGetDeviceCount(&count));
    if (count != 1) throw std::runtime_error("Exactly one leased MACA device must be visible");
    MC_CHECK(mcSetDevice(0));
    mcDeviceProp_t property{};
    MC_CHECK(mcGetDeviceProperties(&property, 0));
    if (std::string(property.name) != "MetaX C550" || property.waveSize != 64)
        throw std::runtime_error("Expected exact device MetaX C550 with observed wave size 64");
    if (property.maxThreadsPerBlock < (C550_WMMA_CONTROL ? 256 : 64))
        throw std::runtime_error("Runtime device limit below the required probe block size");
    char pci[64]{};
    MC_CHECK(mcDeviceGetPCIBusId(pci, sizeof(pci), 0));
    int runtime_version = 0, driver_version = 0;
    MC_CHECK(mcRuntimeGetVersion(&runtime_version));
    MC_CHECK(mcDriverGetVersion(&driver_version));
    records << "{\"type\":\"device\",\"logical_device\":0,\"visible_device_count\":" << count
            << ",\"name\":" << json_string(property.name) << ",\"pci_bus_id\":" << json_string(pci)
            << ",\"runtime_version_api\":" << runtime_version << ",\"driver_version_api\":" << driver_version
            << ",\"api_major\":" << property.major << ",\"api_minor\":" << property.minor
            << ",\"total_global_mem_bytes\":" << property.totalGlobalMem
            << ",\"wave_size_api\":" << property.waveSize << ",\"warp_size_compat_api\":" << property.warpSize
            << ",\"multiprocessor_count\":" << property.multiProcessorCount
            << ",\"l2_cache_bytes\":" << property.l2CacheSize
            << ",\"shared_mem_per_block_bytes\":" << property.sharedMemPerBlock
            << ",\"max_threads_per_multiprocessor\":" << property.maxThreadsPerMultiProcessor
            << ",\"max_threads_per_block\":" << property.maxThreadsPerBlock << "}\n";
    records << "{\"type\":\"protocol\",\"schema_version\":1,\"experiment\":" << json_string(kExperiment) << ","
               "\"operand_dtype\":\"float16\",\"accumulator_dtype\":\"float32\",\"output_elements\":256,"
               "\"guard_elements_each_side\":64,\"required_wave_size\":64,\"timer\":\"mcEventElapsedTime\","
               "\"comparison\":\"finite exact numeric equality; signed zero equivalent; no tolerance\","
               "\"header_route\":\"native __clang_maca_mma_functions.h; mxmaca::wmma; no cu-bridge\","
               "\"timed_scope\":\"default-stream events around 10 full tile launches; may include host submission gaps\","
               "\"transfer_scope\":\"CPU packing, allocation, H2D/D2H copies, reset, attribute queries and file writes excluded\","
               "\"cache_policy\":\"repeated addresses; no explicit application cache reset; runtime cache policy unknown\","
               "\"interpretation\":\"descriptive full tile kernel; no native-instruction throughput or end-to-end GEMM claim\","
               "\"pointer_alignment_scope\":\"largest power of two dividing each observed device address; no imported alignment requirement\","
               "\"MACA_LAUNCH_MODE\":" << json_environment("MACA_LAUNCH_MODE")
            << ",\"MACA_LAUNCH_BLOCKING\":" << json_environment("MACA_LAUNCH_BLOCKING")
            << ",\"MACA_DIRECT_DISPATCH\":" << json_environment("MACA_DIRECT_DISPATCH")
            << ",\"MACA_CACHE_PATH\":" << json_environment("MACA_CACHE_PATH")
            << ",\"MACA_CACHE_DISABLE\":" << json_environment("MACA_CACHE_DISABLE");
#if C550_WMMA_CONTROL
    records << ",\"control_mode\":1,\"variants_per_case\":2,\"input_snapshots\":[\"before\",\"between\",\"after\"],"
               "\"input_rewrite_between_variants\":false,\"purpose\":\"correctness_diagnostic\",\"performance_accepted\":false,"
               "\"scalar_source_scope\":\"float operands and accumulator; compiler contraction and emitted instructions are not prescribed\"";
#endif
#if C550_WMMA_PREFIX
    records << ",\"prefix_mode\":1,\"suite\":\"prefix-control\",\"prefix_families\":[\"singleton\",\"dense\"],"
               "\"prefix_k_min\":0,\"prefix_k_max\":16,\"maximum_cases\":34";
#endif
#if C550_WMMA_WITNESS == 1
    records << ",\"witness_mode\":1,\"suite\":\"witness-control\",\"witness_patterns\":["
               "{\"pattern\":\"dense-origin\",\"target_row\":13,\"target_col\":2,\"input_rule\":\"dense-formulas\"},"
               "{\"pattern\":\"isolated-origin\",\"target_row\":13,\"target_col\":2,\"input_rule\":\"isolated-fixed-pairs\"},"
               "{\"pattern\":\"isolated-c00\",\"target_row\":0,\"target_col\":0,\"input_rule\":\"isolated-fixed-pairs\"}],"
               "\"logical_shape\":[16,16,2],\"maximum_cases\":3,"
               "\"dense_a_rule\":\"A_num(i,k)=((67*i+13*k)%31)-15\","
               "\"dense_b_rule\":\"B_num(k,j)=((17*k+5*j+3)%29)-14\","
               "\"isolated_a_numerators\":[-12,1],\"isolated_b_numerators\":[-1,-13],\"input_scale_denominator\":16,"
               "\"isolated_rule\":\"only declared A target_row and B target_col retain the ordered pairs at k0,k1; all other input words are positive zero\","
               "\"target_reference_numerator\":-1,\"target_reference_denominator\":256";
#endif
#if C550_WMMA_WITNESS >= 2
    records << ",\"witness_mode\":" << C550_WMMA_WITNESS << ",\"suite\":" << json_string(kOrderedSuite)
            << ',' << json_string(std::string(kPatternLabel) + "_patterns") << ":[";
    bool first_pattern = true;
    for (const auto& pattern : kProductPatterns) {
        if (!first_pattern) records << ',';
        first_pattern = false;
        records << "{\"pattern\":" << json_string(pattern.name)
                << ",\"target_row\":0,\"target_col\":0,\"input_rule\":" << json_string(kOrderedRule);
        product_terms(records, pattern);
        records << '}';
    }
    records << "],\"logical_shape\":[16,16,2],\"maximum_cases\":" << kPatternMaximum;
#if C550_WMMA_WITNESS == 5
    records << ",\"base_input_denominator\":16,\"b_denominator\":16,"
               "\"scale_rule\":\"only A is multiplied by 2^scale_exp from its base numerator/16; B is unchanged; all other input words are positive zero\","
               "\"reference_rule\":\"target_reference_numerator/target_reference_denominator; all other outputs are zero\","
               "\"q_values\":[6,7,12],\"scale_exponents\":[-2,-1,0,1,2],\"roles\":[\"positive\",\"negative\",\"pair\"],"
               "\"standalone_inactive_slot_policy\":\"both A and B are positive zero at the inactive K slot\","
               "\"baseline_scope\":\"matching_magnitude_pattern compares complete declared operand words, including cross-scale matches; actual historical equality must be checked separately\"";
#else
    records << ",\"input_scale_denominator\":16,"
               "\"target_reference_denominator\":256," << json_string(std::string(kPatternLabel) + "_rule")
            << ":\"only A row0 and B column0 contain the declared ordered pairs at k0,k1; all other input words are positive zero\"";
#endif
#if C550_WMMA_WITNESS == 3
    records << ",\"paired_b_numerators\":[-1,-13],"
               "\"standalone_inactive_slot_policy\":\"both A and B are positive zero at the inactive K slot\","
               "\"baseline_scope\":\"matching_product_pattern identifies a declared equal operand contract, not a new device comparison\"";
#endif
#if C550_WMMA_WITNESS == 4
    records << ",\"q_min\":1,\"q_max\":14,\"roles\":[\"positive\",\"negative\",\"pair\"],"
               "\"standalone_inactive_slot_policy\":\"both A and B are positive zero at the inactive K slot\","
               "\"baseline_scope\":\"matching_sign_pattern identifies a declared equal operand contract, not a new device comparison\"";
#endif
#endif
    records << "}\n";
    records.flush();
    __half *device_a = nullptr, *device_b = nullptr;
    float* device_c[2] = {nullptr, nullptr};
    MC_CHECK(mcMalloc(reinterpret_cast<void**>(&device_a), kOperandHalfwords * sizeof(__half)));
    MC_CHECK(mcMalloc(reinterpret_cast<void**>(&device_b), kOperandHalfwords * sizeof(__half)));
    const unsigned variant_count = C550_WMMA_CONTROL ? 2 : 1;
    for (unsigned slot = 0; slot < variant_count; ++slot)
        MC_CHECK(mcMalloc(reinterpret_cast<void**>(&device_c[slot]), (kOutputWords + 2 * kGuardWords) * sizeof(float)));
    const uintptr_t a_alignment = observed_alignment(device_a), b_alignment = observed_alignment(device_b);
    mcEvent_t start, stop;
    MC_CHECK(mcEventCreate(&start));
    MC_CHECK(mcEventCreate(&stop));
    for (const auto& c : cases) {
        MC_CHECK(mcMemcpy(device_a, c.a.data(), c.a.size() * sizeof(uint16_t), mcMemcpyHostToDevice));
        MC_CHECK(mcMemcpy(device_b, c.b.data(), c.b.size() * sizeof(uint16_t), mcMemcpyHostToDevice));
#if C550_WMMA_CONTROL
        records << "{\"type\":\"logical_case\",\"id\":" << json_string(c.id)
                << ",\"m\":" << c.m << ",\"n\":" << c.n << ",\"k\":" << c.k
                << ",\"order\":" << json_string(c.order);
#if C550_WMMA_PREFIX
        records << ",\"suite\":" << json_string(c.suite) << ",\"family\":" << json_string(c.family);
#endif
#if C550_WMMA_WITNESS
        witness_fields(records, c);
#endif
        records << "}\n";
        retain_inputs(c, "before", device_a, device_b, output_directory, records);
#endif
        for (unsigned stage = 0; stage < variant_count; ++stage) {
            bool scalar = false;
#if C550_WMMA_CONTROL
            scalar = stage == 0 ? c.order == "scalar-first" : c.order == "wmma-first";
            if (stage == 1) retain_inputs(c, "between", device_a, device_b, output_directory, records);
#endif
            float* output_base = device_c[scalar ? 1 : 0];
            float* payload = output_base + kGuardWords;
            const uintptr_t c_alignment = observed_alignment(payload);
            const unsigned threads = scalar ? 256 : 64;
            std::string output_name = c.id + ".f32";
#if C550_WMMA_CONTROL
            const char* variant = scalar ? "scalar" : "wmma";
            output_name = c.id + "." + variant + ".f32";
#endif
            MC_CHECK(mcMemset(output_base, 0xff, (kOutputWords + 2 * kGuardWords) * sizeof(float)));
            mcFuncAttributes attributes{};
#if C550_WMMA_CONTROL
            const void* function = scalar ? reinterpret_cast<const void*>(scalar_tile_kernel)
                                          : reinterpret_cast<const void*>(wmma_tile_kernel);
            MC_CHECK(mcFuncGetAttributes(&attributes, function));
#else
            MC_CHECK(mcFuncGetAttributes(&attributes, reinterpret_cast<const void*>(wmma_tile_kernel)));
#endif
            records << "{\"type\":\"case\",\"id\":" << json_string(c.id)
                    << ",\"m\":" << c.m << ",\"n\":" << c.n << ",\"k\":" << c.k
                    << ",\"tile_m\":16,\"tile_n\":16,\"tile_k\":16,\"k_chunks\":" << (c.k + 15) / 16
                    << ",\"packed_chunks\":4,\"a_layout\":\"row_major\",\"b_layout\":\"col_major\",\"c_layout\":\"row_major\","
                       "\"leading_dimension\":16,\"operand_dtype\":\"float16\",\"accumulator_dtype\":\"float32\","
                       "\"output_elements\":256,\"operand_halfwords_each\":1024,\"physical_threads\":" << threads
                    << ",\"block_x\":" << threads << ",\"block_y\":1,\"block_z\":1,\"grid_x\":1,\"grid_y\":1,\"grid_z\":1,"
                       "\"required_wave_size\":64,\"participation\":" << json_string(scalar ? "one thread per C element; full padded K chunks" : "all 64 physical threads; uniform K-chunk loop")
                    << ",\"a_file\":" << json_string(c.id + ".a.f16") << ",\"b_file\":" << json_string(c.id + ".b.f16")
                    << ",\"output_file\":" << json_string(output_name) << ",\"guard_elements_each_side\":64,"
                       "\"warmups\":10,\"samples\":10,\"launches_per_sample\":10,\"total_launches\":110,"
                       "\"pointer_alignment_observed_bytes\":{\"a\":" << a_alignment << ",\"b\":" << b_alignment << ",\"c_payload\":" << c_alignment << "},"
                       "\"function_attributes_before_timing\":{\"maxThreadsPerBlock\":" << attributes.maxThreadsPerBlock
                    << ",\"numRegs\":" << attributes.numRegs << ",\"sharedSizeBytes\":" << attributes.sharedSizeBytes
                    << ",\"localSizeBytes\":" << attributes.localSizeBytes << "}";
#if C550_WMMA_CONTROL
            records << ",\"variant\":" << json_string(variant) << ",\"order\":" << json_string(c.order);
#endif
#if C550_WMMA_PREFIX
            records << ",\"suite\":" << json_string(c.suite) << ",\"family\":" << json_string(c.family);
#endif
#if C550_WMMA_WITNESS
            witness_fields(records, c);
#endif
            records << "}\n";
            records.flush();
            for (unsigned i = 0; i < c.warmups; ++i) launch(c, device_a, device_b, payload, scalar);
            MC_CHECK(mcDeviceSynchronize());
            for (unsigned sample = 0; sample < c.samples; ++sample) {
                MC_CHECK(mcEventRecord(start, 0));
                const auto host_start = std::chrono::steady_clock::now();
                for (unsigned i = 0; i < c.launches; ++i) launch(c, device_a, device_b, payload, scalar);
                const auto host_stop = std::chrono::steady_clock::now();
                MC_CHECK(mcEventRecord(stop, 0));
                MC_CHECK(mcEventSynchronize(stop));
                float elapsed_ms = 0;
                MC_CHECK(mcEventElapsedTime(&elapsed_ms, start, stop));
                const double host_us = std::chrono::duration<double, std::micro>(host_stop - host_start).count();
                if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0 || !std::isfinite(host_us) || host_us <= 0)
                    throw std::runtime_error("Nonpositive or nonfinite timing");
                records << "{\"type\":\"sample\",\"id\":" << json_string(c.id);
#if C550_WMMA_CONTROL
                records << ",\"variant\":" << json_string(variant);
#endif
                records << ",\"sample\":" << sample
                        << ",\"event_batch_ms\":" << elapsed_ms << ",\"host_enqueue_batch_us\":" << host_us << "}\n";
            }
            std::vector<float> output(kOutputWords + 2 * kGuardWords);
            MC_CHECK(mcMemcpy(output.data(), output_base, output.size() * sizeof(float), mcMemcpyDeviceToHost));
            std::ofstream file(output_directory + "/" + output_name, std::ios::binary);
            file.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
            file.close();
            if (!file) throw std::runtime_error("Could not retain complete C output and guards");
            records.flush();
            if (!records) throw std::runtime_error("Could not retain raw records");
        }
#if C550_WMMA_CONTROL
        retain_inputs(c, "after", device_a, device_b, output_directory, records);
#endif
    }
    MC_CHECK(mcEventDestroy(stop));
    MC_CHECK(mcEventDestroy(start));
    for (unsigned slot = 0; slot < variant_count; ++slot) MC_CHECK(mcFree(device_c[slot]));
    MC_CHECK(mcFree(device_b));
    MC_CHECK(mcFree(device_a));
    MC_CHECK(mcDeviceSynchronize());
    records << "{\"type\":\"complete\",\"cases\":" << cases.size() << ",\"cpu_correctness_checked\":false";
#if C550_WMMA_CONTROL
    records << ",\"variant_executions\":" << cases.size() * 2;
#endif
    records << "}\n";
    records.close();
    if (!records) throw std::runtime_error("Could not finish raw records");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4 || std::string(argv[1]) != "--run") {
            std::cerr << "Usage: probe --run INPUT_DIRECTORY FRESH_EMPTY_OUTPUT_DIRECTORY\n";
            return 2;
        }
        run(argv[2], argv[3]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "{\"type\":\"error\",\"message\":" << json_string(error.what()) << "}\n";
        return 1;
    }
}
