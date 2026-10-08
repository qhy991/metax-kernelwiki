#!/usr/bin/env python3
"""CPU-only preparation and full oracle for fixed-footprint read-order probes."""
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
    "memory_order_native_helpers", Path(__file__).resolve().parents[1] / "native/native_probe.py")
_NATIVE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_NATIVE)
INPUT_ELEMENTS = _NATIVE.INPUT_ELEMENTS
GUARD_ELEMENTS = _NATIVE.GUARD_ELEMENTS
GUARD_WORD = _NATIVE.GUARD_WORD
read_words = _NATIVE.read_words
validate_input = _NATIVE.validate_input
EXPERIMENT = "fixed-footprint-memory-order"
LENGTHS = (1 << 16, 1 << 20, 1 << 24)
SHIFTS = (0, 2, 4, 6, 8, 12)
COLUMNS = ("id", "n", "shift", "block", "warmups", "samples", "launches")
ENVIRONMENT = ("MACA_LAUNCH_MODE", "MACA_LAUNCH_BLOCKING", "MACA_DIRECT_DISPATCH",
               "MACA_CACHE_PATH", "MACA_CACHE_DISABLE")


def validate_permutation(n: int, shift: int) -> tuple[int, int]:
    if type(n) is not int or n < 2 or n > INPUT_ELEMENTS or n & (n - 1):
        raise ValueError("n must be a power of two between 2 and 16777216")
    if type(shift) is not int or not 0 <= shift < n.bit_length() - 1:
        raise ValueError("shift must satisfy 0 <= shift < log2(n)")
    rows = 2 ** shift
    return rows, n // rows


def reference_indices(n: int, shift: int):
    """Flatten the transpose of input with shape (n/2**shift, 2**shift)."""
    rows, columns = validate_permutation(n, shift)
    for index in range(n):
        yield (index % columns) * rows + index // columns


def inverse_index(index: int, n: int, shift: int) -> int:
    rows, columns = validate_permutation(n, shift)
    if type(index) is not int or not 0 <= index < n:
        raise ValueError("Index outside the permutation")
    return (index % rows) * columns + index // rows


def default_cases() -> list[dict]:
    return [dict(id=f"order_n{n}_s{shift}_b256", n=n, shift=shift, block=256,
                 warmups=10, samples=10, launches=10) for n in LENGTHS for shift in SHIFTS]


def read_plan(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("Unsupported cases.tsv header")
        rows = list(reader)
    if not 1 <= len(rows) <= 18:
        raise ValueError("A plan must contain 1 to 18 cases")
    cases, ids = [], set()
    for row in rows:
        if None in row or any(row[key] is None for key in COLUMNS):
            raise ValueError("Malformed plan row")
        case = {key: row[key] if key == "id" else int(row[key]) for key in COLUMNS}
        key = case["id"]
        if not key or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for ch in key) or key in ids:
            raise ValueError("Invalid or duplicate case id")
        ids.add(key)
        validate_permutation(case["n"], case["shift"])
        if case["n"] not in LENGTHS or case["shift"] not in SHIFTS:
            raise ValueError("Case is outside the fixed experiment plan")
        if (case["block"], case["warmups"], case["samples"], case["launches"]) != (256, 10, 10, 10):
            raise ValueError("Unsupported launch or timing protocol")
        cases.append(case)
    return cases


def oracle_metadata() -> dict:
    return dict(schema_version=1, experiment=EXPERIMENT, input_elements=INPUT_ELEMENTS,
                input_dtype="little-endian IEEE-754 binary32", input_rule="input[i] = float32(i)",
                output_rule="output[i] = input[(i % (n / 2**shift)) * 2**shift + i // (n / 2**shift)]",
                guard_elements_each_side=GUARD_ELEMENTS, guard_uint32=GUARD_WORD,
                comparison="bitwise equality for every payload element and guard")


def prepare(destination: Path) -> dict:
    _NATIVE.require_binary32_little_endian()
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "input.f32").open("wb") as handle:
        for start in range(0, INPUT_ELEMENTS, 1 << 18):
            array.array("f", range(start, min(start + (1 << 18), INPUT_ELEMENTS))).tofile(handle)
    cases = default_cases()
    with (destination / "cases.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(cases)
    (destination / "oracle.json").write_text(json.dumps(oracle_metadata(), indent=2) + "\n")
    return dict(prepared=str(destination), cases=len(cases), input_bytes=INPUT_ELEMENTS * 4)


def validate_output(case: dict, source: array.array, output: array.array) -> dict:
    n = case["n"]
    indices = reference_indices(n, case["shift"])
    if len(source) < n or len(output) != n + 2 * GUARD_ELEMENTS:
        raise ValueError(f"{case['id']}: incorrect input or output extent")
    for side, guard in (("prefix", output[:GUARD_ELEMENTS]), ("suffix", output[-GUARD_ELEMENTS:])):
        if any(word != GUARD_WORD for word in guard):
            raise ValueError(f"{case['id']}: {side} guard overwritten")
    floats = memoryview(output).cast("B").cast("f")
    mismatches = finite_count = 0
    first_mismatch = None
    for index, source_index in enumerate(indices):
        finite_count += math.isfinite(floats[GUARD_ELEMENTS + index])
        if output[GUARD_ELEMENTS + index] != source[source_index]:
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = index
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
            expected = {field: case[field] for field in ("n", "shift", "block", "warmups", "samples")}
            expected.update(log2_n=case["n"].bit_length() - 1, launches_per_sample=case["launches"],
                            total_launches=case["warmups"] + case["samples"] * case["launches"],
                            grid=case["n"] // case["block"], unique_input_elements=case["n"],
                            input_span_bytes=case["n"] * 4, logical_bytes_per_launch=case["n"] * 8,
                            output_file=f"{key}.f32")
            if any(type(record.get(field)) is not type(value) or record[field] != value
                   for field, value in expected.items()):
                raise ValueError(f"Case metadata mismatch: {key}")
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
    summaries = []
    for case in cases:
        row = {field: case[field] for field in ("id", "n", "shift", "block")}
        output = read_words(output_directory / f"{case['id']}.f32", case["n"] + 2 * GUARD_ELEMENTS)
        row.update(validate_output(case, source, output))
        timings = [sample["event_batch_ms"] * 1000 / case["launches"] for sample in samples[case["id"]]]
        median = statistics.median(timings)
        row.update(event_mean_per_launch_us_median=median, event_mean_per_launch_us_min=min(timings),
                   event_mean_per_launch_us_max=max(timings), sample_count=len(timings),
                   launches_per_sample=case["launches"], logical_gbps_at_median=case["n"] * 8 / (median * 1000))
        summaries.append(row)
    return dict(status="pass", experiment=EXPERIMENT, input_finite_count=INPUT_ELEMENTS,
                checker="CPU transpose-index oracle; full bitwise payload, finite counts and guard checks",
                cases_checked=len(cases), cases=summaries,
                timing_scope="default-stream batch averages, including host submission gaps; repeated addresses; runtime cache policy unknown",
                throughput_scope="logical read+write GB/s, not DRAM traffic or bandwidth")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare").add_argument("directory", type=Path)
    checker = commands.add_parser("check")
    checker.add_argument("input_directory", type=Path)
    checker.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    try:
        result = prepare(args.directory) if args.command == "prepare" else check(args.input_directory, args.output_directory)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(status="error", message=str(error))), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
