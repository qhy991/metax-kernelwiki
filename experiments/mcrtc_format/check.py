#!/usr/bin/env python3
"""Verify CPU producer receipts without interpreting or loading returned code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCHEMA = "metax-kernelwiki.mcrtc-format.v1"
CASES = ("valid", "compile-error")
VALID_SOURCE = 'extern "C" __global__ void mcrtc_format_probe() {}\n'
NEGATIVE_MARKER = b"MCRTC_FORMAT_NEGATIVE_CONTROL"
MASKS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "MACA_VISIBLE_DEVICES")
MAXIMUM_BUFFER_BYTES = 64 * 1024 * 1024


class ReceiptError(ValueError):
    """The files do not verify the declared producer observation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def exact(actual, expected, context: str) -> None:
    require(type(actual) is type(expected), context + ": incorrect type")
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), context + ": missing or extra fields")
        for key, value in expected.items():
            exact(actual[key], value, context + "." + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), context + ": incorrect length")
        for index, (left, right) in enumerate(zip(actual, expected)):
            exact(left, right, context + f"[{index}]")
    else:
        require(actual == expected, context + ": incorrect value")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ReceiptError("Nonfinite JSON constant: " + value)


def request_metadata(expected_case: str) -> dict:
    require(type(expected_case) is str and expected_case in CASES, "Unsupported expected case")
    source = ("#error MCRTC_FORMAT_NEGATIVE_CONTROL\n" if expected_case == "compile-error" else "") + VALID_SOURCE
    return dict(type="request", schema=SCHEMA, case=expected_case, source=source,
                program_name="mcrtc_format_probe.mc", num_options=0, options=[], options_pointer_is_null=True,
                num_headers=0, headers_pointer_is_null=True, include_names_pointer_is_null=True,
                visibility={key: "" for key in MASKS}, maximum_buffer_bytes=MAXIMUM_BUFFER_BYTES)


def api_record(row: dict, name: str, code: int = 0) -> None:
    require(type(row) is dict and row.keys() == {"type", "api", "status_code", "status_name", "error_string"},
            name + ": API record fields differ")
    require(type(row["error_string"]) is str, name + ": missing error-string observation")
    exact({key: value for key, value in row.items() if key != "error_string"},
          dict(type="api", api=name, status_code=code,
               status_name="MCRTC_ERROR_COMPILATION" if code == 6 else "MCRTC_SUCCESS"), name)


def buffer_extent(row: dict, buffer: str) -> int:
    require(type(row) is dict and row.keys() == {"type", "buffer", "bytes"}, buffer + ": size fields differ")
    exact(row["type"], "size", buffer + ".type")
    exact(row["buffer"], buffer, buffer + ".buffer")
    size = row["bytes"]
    require(type(size) is int and 0 <= size <= MAXIMUM_BUFFER_BYTES, buffer + ": invalid reported byte extent")
    return size


def retained_bytes(directory: Path, row: dict, buffer: str, filename: str, size: int) -> bytes:
    exact(row, dict(type="file", buffer=buffer, file=filename, bytes=size), filename + " record")
    data = (directory / filename).read_bytes()
    require(len(data) == size, filename + ": retained extent differs from the API size")
    return data


def check(directory: Path, expected_case: str) -> dict:
    expected_request = request_metadata(expected_case)
    rows = [json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
            for line in (directory / "raw.jsonl").read_text().splitlines()]
    success = expected_case == "valid"
    require(len(rows) == (15 if success else 11) and all(type(row) is dict for row in rows),
            "Missing, extra or malformed producer records")
    exact(rows[0], expected_request, "request")
    api_record(rows[1], "mcrtcVersion")
    version = rows[2]
    require(version.keys() == {"type", "major", "minor"}, "Version fields differ")
    exact(version["type"], "version", "version.type")
    require(all(type(version[key]) is int and 0 <= version[key] <= 0x7fffffff for key in ("major", "minor")),
            "Version must contain observed nonnegative integers")
    api_record(rows[3], "mcrtcCreateProgram")
    api_record(rows[4], "mcrtcCompileProgram", 0 if success else 6)
    api_record(rows[5], "mcrtcGetProgramLogSize")
    log_size = buffer_extent(rows[6], "compile-log")
    api_record(rows[7], "mcrtcGetProgramLog")
    log = retained_bytes(directory, rows[8], "compile-log", "compile.log", log_size)
    # The installed header defines nonempty log extents to include a trailing NUL.
    # A reported zero remains an exact empty file rather than an invented byte.
    require(not log or log[-1] == 0, "Compile log does not retain its reported trailing NUL")
    output_size = None
    if success:
        api_record(rows[9], "mcrtcGetBitcodeSize")
        output_size = buffer_extent(rows[10], "producer-output")
        api_record(rows[11], "mcrtcGetBitcode")
        retained_bytes(directory, rows[12], "producer-output", "producer-output.bin", output_size)
    else:
        require(NEGATIVE_MARKER in log, "Compilation failed without the declared negative-control marker")
        require(not (directory / "producer-output.bin").exists(), "Compile-failed control must not retain a producer output")
    api_record(rows[-2], "mcrtcDestroyProgram")
    status, producer_exit = ("producer-success", 0) if success else ("compile-failed", 1)
    exact(rows[-1], dict(type="complete", case=expected_case, status=status, exit_code=producer_exit,
                        program_created=True, program_destroyed=True, log_retained=True, output_retained=success), "completion")
    return dict(schema=SCHEMA, case=expected_case, status=status, receipt_valid=True,
                expected_compile_failure=not success, recorded_producer_exit=producer_exit,
                observed_mcrtc_version={key: version[key] for key in ("major", "minor")},
                api_records_checked=sum(row["type"] == "api" for row in rows),
                log_bytes_checked=log_size, output_bytes_checked=output_size,
                log_retained=True, output_retained=success, program_destroyed=True,
                visibility={key: "" for key in MASKS}, compile_options=[],
                output_format_interpreted=False, loader_acceptance="not_tested", numerical_acceptance="not_applicable",
                scope="Declared CPU producer receipt and exact file extents only; no output-format, loader, device-execution or numerical acceptance is established. The caller must separately bind the recorded producer exit to its process receipt.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--case", choices=CASES, required=True, help="Expected independently declared producer case")
    args = parser.parse_args()
    try:
        result = check(args.directory, args.case)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0  # A matched negative retains compile-failed; it is not producer success.
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(status="error", error_kind="receipt", message=str(error))), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
