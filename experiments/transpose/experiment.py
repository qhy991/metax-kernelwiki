#!/usr/bin/env python3
"""CPU-only preparation and exact row-major transpose oracle."""
from __future__ import annotations

import argparse
import array
import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

_SPEC = importlib.util.spec_from_file_location(
    "transpose_native_helpers", Path(__file__).resolve().parents[1] / "native/native_probe.py")
_NATIVE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_NATIVE)
INPUT_ELEMENTS = _NATIVE.INPUT_ELEMENTS
GUARD_ELEMENTS = _NATIVE.GUARD_ELEMENTS
GUARD_WORD = _NATIVE.GUARD_WORD
read_words = _NATIVE.read_words
validate_input = _NATIVE.validate_input
EXPERIMENT = "fp32-row-major-transpose"
SHAPES = ((1, 1), (1, 65), (65, 1), (31, 33), (33, 31),
          *((rows, cols) for rows in (63, 64, 65) for cols in (63, 64, 65)),
          (262144, 64), (65536, 256), (4096, 4096), (262143, 63), (4095, 4097))
VARIANTS = ("direct", "tile64", "tile64_pad1")
SHARED_PITCH_VARIANTS = {"runtime_pitch64": 64, "runtime_pitch65": 65}
COLUMNS = ("id", "rows", "cols", "variant", "warmups", "samples", "launches")
ENVIRONMENT = ("MACA_LAUNCH_MODE", "MACA_LAUNCH_BLOCKING", "MACA_DIRECT_DISPATCH",
               "MACA_CACHE_PATH", "MACA_CACHE_DISABLE")


def validate_shape(rows: int, cols: int) -> int:
    if (type(rows) is not int or type(cols) is not int or rows <= 0 or cols <= 0
            or rows * cols > INPUT_ELEMENTS):
        raise ValueError("Positive matrix dimensions with rows * cols <= 16777216 required")
    return rows * cols


def reference_pairs(rows: int, cols: int):
    """Map each input matrix coordinate to its transposed output coordinate."""
    validate_shape(rows, cols)
    for row in range(rows):
        for col in range(cols):
            yield row * cols + col, col * rows + row


def default_cases() -> list[dict]:
    return [dict(id=f"transpose_r{rows}_c{cols}_{variant}", rows=rows, cols=cols, variant=variant,
                 warmups=10, samples=10, launches=10)
            for rows, cols in SHAPES for variant in VARIANTS]


def shared_pitch_cases() -> list[dict]:
    return [dict(id=f"transpose_r{rows}_c{cols}_{variant}", rows=rows, cols=cols, variant=variant,
                 warmups=10, samples=10, launches=10)
            for rows, cols in SHAPES for variant in SHARED_PITCH_VARIANTS]


def launch_metadata(case: dict) -> dict:
    rows, cols, variant = case["rows"], case["cols"], case["variant"]
    n = validate_shape(rows, cols)
    if variant not in VARIANTS and variant not in SHARED_PITCH_VARIANTS:
        raise ValueError("Unknown transpose variant")
    tiled = variant != "direct"
    padding = int(variant in ("tile64_pad1", "runtime_pitch65"))
    result = dict(n=n, block_x=64 if tiled else 256, block_y=4 if tiled else 1, block_z=1,
                  grid_x=(cols + 63) // 64 if tiled else (n + 255) // 256,
                  grid_y=(rows + 63) // 64 if tiled else 1, grid_z=1,
                  tile_rows=64 if tiled else 0, tile_cols=64 if tiled else 0, padding=padding,
                  intended_static_shared_bytes=64 * (64 + padding) * 4 if tiled else 0)
    if variant in SHARED_PITCH_VARIANTS:
        result.update(shared_pitch_elements=SHARED_PITCH_VARIANTS[variant],
                      allocated_shared_elements=64 * 65, intended_static_shared_bytes=64 * 65 * 4)
    return result


def read_plan(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("Unsupported cases.tsv header")
        rows = list(reader)
    if not 1 <= len(rows) <= 57:
        raise ValueError("A plan must contain 1 to 57 cases")
    cases, ids = [], set()
    for row in rows:
        if None in row or any(row[key] is None for key in COLUMNS):
            raise ValueError("Malformed plan row")
        case = {key: row[key] if key in ("id", "variant") else int(row[key]) for key in COLUMNS}
        key = case["id"]
        if not key or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for ch in key) or key in ids:
            raise ValueError("Invalid or duplicate case id")
        ids.add(key)
        validate_shape(case["rows"], case["cols"])
        if ((case["rows"], case["cols"]) not in SHAPES
                or (case["variant"] not in VARIANTS and case["variant"] not in SHARED_PITCH_VARIANTS)):
            raise ValueError("Case is outside the fixed experiment plan")
        if (case["warmups"], case["samples"], case["launches"]) != (10, 10, 10):
            raise ValueError("Unsupported launch or timing protocol")
        cases.append(case)
    return cases


def oracle_metadata() -> dict:
    return dict(schema_version=1, experiment=EXPERIMENT, input_elements=INPUT_ELEMENTS,
                input_dtype="little-endian IEEE-754 binary32", input_rule="input[i] = float32(i)",
                output_rule="output[col * rows + row] = input[row * cols + col]",
                guard_elements_each_side=GUARD_ELEMENTS, guard_uint32=GUARD_WORD,
                comparison="bitwise equality for every payload element and guard")


def prepare(destination: Path, suite: str = "default") -> dict:
    _NATIVE.require_binary32_little_endian()
    if suite not in ("default", "shared-pitch"):
        raise ValueError("Unknown preparation suite")
    cases = default_cases() if suite == "default" else shared_pitch_cases()
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "input.f32").open("wb") as handle:
        for start in range(0, INPUT_ELEMENTS, 1 << 18):
            array.array("f", range(start, min(start + (1 << 18), INPUT_ELEMENTS))).tofile(handle)
    with (destination / "cases.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(cases)
    (destination / "oracle.json").write_text(json.dumps(oracle_metadata(), indent=2) + "\n")
    return dict(prepared=str(destination), suite=suite, cases=len(cases), input_bytes=INPUT_ELEMENTS * 4)


def validate_output(case: dict, source: array.array, output: array.array) -> dict:
    n = validate_shape(case["rows"], case["cols"])
    if len(source) < n or len(output) != n + 2 * GUARD_ELEMENTS:
        raise ValueError(f"{case['id']}: incorrect input or output extent")
    for side, guard in (("prefix", output[:GUARD_ELEMENTS]), ("suffix", output[-GUARD_ELEMENTS:])):
        if any(word != GUARD_WORD for word in guard):
            raise ValueError(f"{case['id']}: {side} guard overwritten")
    floats = memoryview(output).cast("B").cast("f")
    mismatches = finite_count = 0
    first_mismatch = None
    for source_index, output_index in reference_pairs(case["rows"], case["cols"]):
        finite_count += math.isfinite(floats[GUARD_ELEMENTS + output_index])
        if output[GUARD_ELEMENTS + output_index] != source[source_index]:
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = output_index
    if mismatches or finite_count != n:
        raise ValueError(f"{case['id']}: {mismatches} mismatches, finite_count={finite_count}/{n}, "
                         f"first_mismatch={first_mismatch}")
    return dict(payload_elements_checked=n, finite_count=finite_count, unique_input_indices=n,
                guard_elements_checked=2 * GUARD_ELEMENTS, mismatches=0)


def validate_records(records: list[dict], cases: list[dict]) -> dict[str, list[dict]]:
    for kind in ("device", "protocol", "complete"):
        if sum(record.get("type") == kind for record in records) != 1:
            raise ValueError(f"Expected exactly one {kind} record")
    if records[-1].get("type") != "complete":
        raise ValueError("Run did not complete")
    device = next(row for row in records if row["type"] == "device")
    protocol = next(row for row in records if row["type"] == "protocol")
    complete = records[-1]
    if (device.get("name") != "MetaX C550" or type(device.get("visible_device_count")) is not int
            or device["visible_device_count"] != 1):
        raise ValueError("Expected exactly one visible MetaX C550")
    if (not isinstance(device.get("pci_bus_id"), str) or not device["pci_bus_id"]
            or type(device.get("wave_size_api")) is not int or device["wave_size_api"] <= 0):
        raise ValueError("Missing device identity or wave-size metadata")
    for field in ("runtime_version_api", "driver_version_api"):
        if type(device.get(field)) is not int or device[field] < 0:
            raise ValueError(f"Missing or invalid version field: {field}")
    if (type(protocol.get("schema_version")) is not int or protocol["schema_version"] != 1
            or protocol.get("experiment") != EXPERIMENT
            or protocol.get("input_elements") != INPUT_ELEMENTS
            or protocol.get("guard_elements_each_side") != GUARD_ELEMENTS
            or protocol.get("timer") != "mcEventElapsedTime"
            or type(complete.get("cases")) is not int or complete["cases"] != len(cases)
            or complete.get("cpu_correctness_checked") is not False):
        raise ValueError("Run protocol mismatch")
    for field in ENVIRONMENT:
        if field not in protocol or (protocol[field] is not None and not isinstance(protocol[field], str)):
            raise ValueError(f"Missing or invalid environment field: {field}")
    by_id = {case["id"]: case for case in cases}
    declarations, samples = [], {key: [] for key in by_id}
    for record in records:
        kind = record.get("type")
        if kind in ("device", "protocol", "complete"):
            continue
        key = record.get("id")
        if key not in by_id:
            raise ValueError(f"Unknown case: {key}")
        case = by_id[key]
        if kind == "case":
            if key in declarations:
                raise ValueError(f"Duplicate case: {key}")
            expected = {field: case[field] for field in ("rows", "cols", "variant", "warmups", "samples")}
            expected.update(launch_metadata(case))
            n = case["rows"] * case["cols"]
            expected.update(launches_per_sample=case["launches"],
                            total_launches=case["warmups"] + case["samples"] * case["launches"],
                            unique_input_elements=n, input_span_bytes=n * 4, logical_bytes_per_launch=n * 8,
                            output_file=f"{key}.f32")
            if any(type(record.get(field)) is not type(value) or record[field] != value
                   for field, value in expected.items()):
                raise ValueError(f"Case metadata mismatch: {key}")
            attributes = record.get("function_attributes_before_timing")
            if (not isinstance(attributes, dict)
                    or any(type(attributes.get(field)) is not int or attributes[field] < minimum
                           for field, minimum in (("maxThreadsPerBlock", 1), ("numRegs", 0),
                                                  ("sharedSizeBytes", 0), ("localSizeBytes", 0)))):
                raise ValueError(f"Missing or invalid function attributes: {key}")
            declarations.append(key)
        elif kind == "sample":
            if not declarations or key != declarations[-1]:
                raise ValueError(f"Sample outside its declared case: {key}")
            for field in ("event_batch_ms", "host_enqueue_batch_us"):
                value = record.get(field)
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid {field}: {key}")
            samples[key].append(record)
        else:
            raise ValueError(f"Unknown record type: {kind}")
    if declarations != list(by_id):
        raise ValueError("Missing or misordered case declaration")
    for key, case in by_id.items():
        indices = [row.get("sample") for row in samples[key]]
        if any(type(index) is not int for index in indices) or indices != list(range(case["samples"])):
            raise ValueError(f"Missing, duplicate or misordered samples: {key}")
    return samples


def check(input_directory: Path, output_directory: Path) -> dict:
    _NATIVE.require_binary32_little_endian()
    if json.loads((input_directory / "oracle.json").read_text()) != oracle_metadata():
        raise ValueError("Unsupported oracle metadata")
    cases = read_plan(input_directory / "cases.tsv")
    source = read_words(input_directory / "input.f32", INPUT_ELEMENTS)
    validate_input(source)
    records = [json.loads(line) for line in (output_directory / "raw.jsonl").read_text().splitlines()]
    samples = validate_records(records, cases)
    declarations = {record["id"]: record for record in records if record["type"] == "case"}
    summaries = []
    for case in cases:
        row = {field: case[field] for field in ("id", "rows", "cols", "variant")}
        row.update(launch_metadata(case))
        row["function_attributes_before_timing"] = declarations[case["id"]]["function_attributes_before_timing"]
        n = case["rows"] * case["cols"]
        output = read_words(output_directory / f"{case['id']}.f32", n + 2 * GUARD_ELEMENTS)
        row.update(validate_output(case, source, output))
        timings = [sample["event_batch_ms"] * 1000 / case["launches"] for sample in samples[case["id"]]]
        median = statistics.median(timings)
        row.update(event_mean_per_launch_us_median=median, event_mean_per_launch_us_min=min(timings),
                   event_mean_per_launch_us_max=max(timings), sample_count=len(timings),
                   launches_per_sample=case["launches"], logical_gbps_at_median=n * 8 / (median * 1000))
        summaries.append(row)
    return dict(status="pass", experiment=EXPERIMENT, input_finite_count=INPUT_ELEMENTS,
                checker="CPU nested (row,col) oracle; full bitwise payload, finite counts and guard checks",
                cases_checked=len(cases), cases=summaries,
                timing_scope="default-stream batch averages, including host submission gaps; repeated addresses; runtime cache policy unknown",
                throughput_scope="logical read+write GB/s, not DRAM traffic or bandwidth")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("directory", type=Path)
    preparation.add_argument("--suite", choices=("default", "shared-pitch"), default="default")
    checker = commands.add_parser("check")
    checker.add_argument("input_directory", type=Path)
    checker.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    try:
        result = prepare(args.directory, args.suite) if args.command == "prepare" else check(args.input_directory, args.output_directory)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(status="error", message=str(error))), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
