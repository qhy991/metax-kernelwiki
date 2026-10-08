// Compiler API observation; no device enumeration, module load or kernel launch calls.
#include <mcr/mcrtc.h>

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
constexpr size_t kMaximumBufferBytes = 64 * 1024 * 1024;
constexpr const char* kProgramName = "mcrtc_format_probe.mc";
constexpr const char* kValidSource = "extern \"C\" __global__ void mcrtc_format_probe() {}\n";
constexpr const char* kVisibility[] = {
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "MACA_VISIBLE_DEVICES"
};

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

void record(std::ofstream& records, const std::string& text) {
    records << text << '\n';
    records.flush();
    if (!records) throw std::runtime_error("Could not retain raw.jsonl");
}

const char* status_name(mcrtcResult status) {
    switch (status) {
        case MCRTC_SUCCESS: return "MCRTC_SUCCESS";
        case MCRTC_ERROR_OUT_OF_MEMORY: return "MCRTC_ERROR_OUT_OF_MEMORY";
        case MCRTC_ERROR_PROGRAM_CREATION_FAILURE: return "MCRTC_ERROR_PROGRAM_CREATION_FAILURE";
        case MCRTC_ERROR_INVALID_INPUT: return "MCRTC_ERROR_INVALID_INPUT";
        case MCRTC_ERROR_INVALID_PROGRAM: return "MCRTC_ERROR_INVALID_PROGRAM";
        case MCRTC_ERROR_INVALID_OPTION: return "MCRTC_ERROR_INVALID_OPTION";
        case MCRTC_ERROR_COMPILATION: return "MCRTC_ERROR_COMPILATION";
        case MCRTC_ERROR_BUILTIN_OPERATION_FAILURE: return "MCRTC_ERROR_BUILTIN_OPERATION_FAILURE";
        case MCRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION: return "MCRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
        case MCRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION: return "MCRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
        case MCRTC_ERROR_NAME_EXPRESSION_NOT_VALID: return "MCRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
        case MCRTC_ERROR_INTERNAL_ERROR: return "MCRTC_ERROR_INTERNAL_ERROR";
    }
    return "UNKNOWN_MCRTC_RESULT";
}

void record_api(std::ofstream& records, const char* api, mcrtcResult status) {
    const char* description = mcrtcGetErrorString(status);
    record(records, "{\"type\":\"api\",\"api\":" + json_string(api) +
           ",\"status_code\":" + std::to_string(static_cast<int>(status)) +
           ",\"status_name\":" + json_string(status_name(status)) +
           ",\"error_string\":" + (description ? json_string(description) : "null") + "}");
    if (!description) throw std::runtime_error("mcrtcGetErrorString returned null");
}

void require_success(mcrtcResult status, const char* api) {
    if (status != MCRTC_SUCCESS)
        throw std::runtime_error(std::string(api) + " returned " + status_name(status));
}

// Preserve the original exception and any cleanup failure even if the metadata
// stream itself has failed. A failed write never becomes a successful capture.
void record_error(std::ofstream& records, const std::string& message) {
    const std::string text = "{\"type\":\"error\",\"message\":" + json_string(message) + "}";
    std::cerr << text << '\n';
    try {
        record(records, text);
    } catch (const std::exception& error) {
        std::cerr << "Could not retain error record: " << error.what() << '\n';
    }
}

void record_size(std::ofstream& records, const char* buffer, size_t bytes) {
    record(records, "{\"type\":\"size\",\"buffer\":" + json_string(buffer) +
           ",\"bytes\":" + std::to_string(bytes) + "}");
    if (bytes > kMaximumBufferBytes)
        throw std::runtime_error(std::string(buffer) + " exceeds the recorded 64 MiB buffer cap");
}

void retain_bytes(std::ofstream& records, const std::string& directory, const char* buffer,
                  const char* filename, const std::vector<char>& bytes, size_t reported_size) {
    std::ofstream output(directory + "/" + filename, std::ios::binary);
    output.write(bytes.data(), static_cast<std::streamsize>(reported_size));
    output.close();
    if (!output) throw std::runtime_error(std::string("Could not retain ") + filename);
    record(records, "{\"type\":\"file\",\"buffer\":" + json_string(buffer) +
           ",\"file\":" + json_string(filename) + ",\"bytes\":" + std::to_string(reported_size) + "}");
}

int capture(const std::string& selected_case, const std::string& directory) {
    if (selected_case != "valid" && selected_case != "compile-error")
        throw std::runtime_error("Case must be valid or compile-error");
    if (!std::filesystem::is_directory(directory) || !std::filesystem::is_empty(directory))
        throw std::runtime_error("Output directory must already exist and be empty");
    // Admission precedes every MCRTC call, including its error-string helper.
    for (const char* key : kVisibility) {
        const char* value = std::getenv(key);
        if (!value || value[0] != '\0')
            throw std::runtime_error(std::string(key) + " must be present and empty");
    }
    const std::string source = (selected_case == "compile-error"
        ? "#error MCRTC_FORMAT_NEGATIVE_CONTROL\n" : "") + std::string(kValidSource);
    std::ofstream records(directory + "/raw.jsonl");
    if (!records) throw std::runtime_error("Cannot create raw.jsonl");
    std::string request = "{\"type\":\"request\",\"schema\":\"metax-kernelwiki.mcrtc-format.v1\",\"case\":" +
        json_string(selected_case) + ",\"source\":" + json_string(source) +
        ",\"program_name\":" + json_string(kProgramName) +
        ",\"num_options\":0,\"options\":[],\"options_pointer_is_null\":true,\"num_headers\":0,"
        "\"headers_pointer_is_null\":true,\"include_names_pointer_is_null\":true,\"visibility\":{";
    for (unsigned index = 0; index < 4; ++index) {
        if (index) request += ',';
        request += json_string(kVisibility[index]) + ":\"\"";
    }
    record(records, request + "},\"maximum_buffer_bytes\":" + std::to_string(kMaximumBufferBytes) + "}");

    mcrtcProgram program = nullptr;
    bool program_created = false, program_destroyed = false, log_retained = false, output_retained = false;
    int exit_code = 2;
    std::string completion_status = "error";
    try {
        int major = -1, minor = -1;
        mcrtcResult status = mcrtcVersion(&major, &minor);
        record_api(records, "mcrtcVersion", status);
        require_success(status, "mcrtcVersion");
        record(records, "{\"type\":\"version\",\"major\":" + std::to_string(major) +
               ",\"minor\":" + std::to_string(minor) + "}");

        status = mcrtcCreateProgram(&program, source.c_str(), kProgramName, 0, nullptr, nullptr);
        program_created = program != nullptr;
        record_api(records, "mcrtcCreateProgram", status);
        require_success(status, "mcrtcCreateProgram");
        if (!program) throw std::runtime_error("mcrtcCreateProgram succeeded with a null program");

        const mcrtcResult compiled = mcrtcCompileProgram(program, 0, nullptr);
        record_api(records, "mcrtcCompileProgram", compiled);
        size_t log_size = kMaximumBufferBytes + 1;
        status = mcrtcGetProgramLogSize(program, &log_size);
        record_api(records, "mcrtcGetProgramLogSize", status);
        require_success(status, "mcrtcGetProgramLogSize");
        record_size(records, "compile-log", log_size);
        // A one-byte scratch allocation keeps the API pointer nonnull for a
        // reported zero size. Only the exact reported extent is ever saved.
        std::vector<char> log(log_size ? log_size : 1, '\0');
        status = mcrtcGetProgramLog(program, log.data());
        record_api(records, "mcrtcGetProgramLog", status);
        require_success(status, "mcrtcGetProgramLog");
        retain_bytes(records, directory, "compile-log", "compile.log", log, log_size);
        log_retained = true;

        if (compiled == MCRTC_SUCCESS) {
            size_t output_size = kMaximumBufferBytes + 1;
            status = mcrtcGetBitcodeSize(program, &output_size);
            record_api(records, "mcrtcGetBitcodeSize", status);
            require_success(status, "mcrtcGetBitcodeSize");
            record_size(records, "producer-output", output_size);
            std::vector<char> output(output_size ? output_size : 1, '\0');
            status = mcrtcGetBitcode(program, output.data());
            record_api(records, "mcrtcGetBitcode", status);
            require_success(status, "mcrtcGetBitcode");
            retain_bytes(records, directory, "producer-output", "producer-output.bin", output, output_size);
            output_retained = true;
            completion_status = "producer-success";
            exit_code = 0;
        } else if (compiled == MCRTC_ERROR_COMPILATION) {
            completion_status = "compile-failed";
            exit_code = 1;
        } else {
            require_success(compiled, "mcrtcCompileProgram");
        }
    } catch (const std::exception& error) {
        record_error(records, error.what());
        exit_code = 2;
        completion_status = "error";
    }

    // Exactly one destroy attempt for every returned handle, including after an
    // API or retention exception. Never retry a possibly invalidated handle.
    if (program) {
        const mcrtcResult status = mcrtcDestroyProgram(&program);
        program_destroyed = status == MCRTC_SUCCESS;
        try {
            record_api(records, "mcrtcDestroyProgram", status);
            require_success(status, "mcrtcDestroyProgram");
        } catch (const std::exception& error) {
            record_error(records, error.what());
            exit_code = 2;
            completion_status = "error";
        }
    }
    try {
        record(records, "{\"type\":\"complete\",\"case\":" + json_string(selected_case) +
               ",\"status\":" + json_string(completion_status) + ",\"exit_code\":" + std::to_string(exit_code) +
               ",\"program_created\":" + (program_created ? "true" : "false") +
               ",\"program_destroyed\":" + (program_destroyed ? "true" : "false") +
               ",\"log_retained\":" + (log_retained ? "true" : "false") +
               ",\"output_retained\":" + (output_retained ? "true" : "false") + "}");
        records.close();
        if (!records) throw std::runtime_error("Could not finish raw.jsonl");
    } catch (const std::exception& error) {
        record_error(records, error.what());
        return 2;
    }
    return exit_code;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 4 || std::string(argv[1]) != "--capture") {
        std::cerr << "Usage: producer --capture {valid|compile-error} FRESH_EMPTY_OUTPUT_DIRECTORY\n";
        return 2;
    }
    try {
        return capture(argv[2], argv[3]);
    } catch (const std::exception& error) {
        std::cerr << "{\"type\":\"error\",\"message\":" << json_string(error.what()) << "}\n";
        return 2;
    }
}
