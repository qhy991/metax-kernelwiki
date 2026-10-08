#!/usr/bin/env python3
"""CPU-only preparation and exact full-wave shuffle/reduction oracle."""
from __future__ import annotations

import argparse
import array
import csv
import json
from pathlib import Path
import statistics
import sys
import math

EXPERIMENT = "wave64-int32-collectives"
INPUT_ELEMENTS = 128
GUARD_ELEMENTS = 32
GUARD_WORD = 0xFFFFFFFF
MASK_HEX = "0xffffffffffffffff"
CHANNELS = tuple(f"shfl_width{width}_src{source}" for width in (64, 32) for source in (0, 31, 32, 63)) + ("reduce_add_wave64",)
COLUMNS = ("id", "block", "n", "warmups", "samples", "launches")
COMMON_N = (0, 1, 31, 32, 33, 63, 64)
BOUNDARIES = {64: COMMON_N, 128: (*COMMON_N, 65, 95, 96, 97, 127, 128)}
ENVIRONMENT = ("MACA_LAUNCH_MODE", "MACA_LAUNCH_BLOCKING", "MACA_DIRECT_DISPATCH", "MACA_CACHE_PATH", "MACA_CACHE_DISABLE")


def require_int32_little_endian() -> None:
    if sys.byteorder != "little" or array.array("i").itemsize != 4 or array.array("I").itemsize != 4:
        raise ValueError("This format requires little-endian signed/unsigned int32")


def default_cases() -> list[dict]:
    return [dict(id=f"wave_b{block}_n{n}", block=block, n=n, warmups=10, samples=10, launches=10)
            for block, lengths in BOUNDARIES.items() for n in lengths]


def validate_case(case: dict) -> None:
    block, n = case["block"], case["n"]
    if type(block) is not int or block not in BOUNDARIES:
        raise ValueError("Block must contain exactly 64 or 128 physical threads")
    if type(n) is not int or n not in BOUNDARIES[block]:
        raise ValueError("Logical length outside the fixed block/length boundary cases")


def read_plan(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("Unsupported cases.tsv header")
        rows = list(reader)
    if not 1 <= len(rows) <= 20:
        raise ValueError("A plan must contain 1 to 20 cases")
    cases, ids = [], set()
    for row in rows:
        if None in row or any(row[field] is None for field in COLUMNS):
            raise ValueError("Malformed plan row")
        case = {field: row[field] if field == "id" else int(row[field]) for field in COLUMNS}
        key = case["id"]
        if not key or key in ids or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for ch in key):
            raise ValueError("Invalid or duplicate case id")
        validate_case(case)
        if (case["warmups"], case["samples"], case["launches"]) != (10, 10, 10):
            raise ValueError("Unsupported timing protocol")
        cases.append(case)
        ids.add(key)
    return cases


def oracle_metadata() -> dict:
    return dict(schema_version=1, experiment=EXPERIMENT, input_elements=128,
                dtype="little-endian int32", input_rule="input[i] = i + 1 for 0 <= i < 128",
                value_rule="physical thread i uses input[i] when i < n, else zero; no thread exits before collectives",
                required_wave_size=64, participation="all physical threads", mask_hex=MASK_HEX, mask_bits=64,
                output_layout="thread-major: output[thread * 9 + channel]", channels=list(CHANNELS),
                shuffle_rule="independent width-sized list slices; direct source index wraps within each slice",
                reduction_rule="exact sum of each independent 64-element wave slice, returned to every thread in that wave",
                guard_elements_each_side=GUARD_ELEMENTS, guard_uint32=GUARD_WORD,
                comparison="exact int32 bits for every physical thread and all nine channels, including zeros")


def prepare(destination: Path) -> dict:
    require_int32_little_endian()
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "input.i32").open("wb") as handle:
        array.array("i", range(1, INPUT_ELEMENTS + 1)).tofile(handle)
    cases = default_cases()
    with (destination / "cases.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(cases)
    (destination / "oracle.json").write_text(json.dumps(oracle_metadata(), indent=2) + "\n")
    return dict(prepared=str(destination), cases=len(cases), input_elements=INPUT_ELEMENTS, input_bytes=4 * INPUT_ELEMENTS)


def read_words(path: Path, count: int) -> array.array:
    if path.stat().st_size != count * 4:
        raise ValueError(f"{path.name}: incorrect int32 file extent")
    result = array.array("I")
    with path.open("rb") as handle:
        result.fromfile(handle, count)
    return result


def validate_input(source: array.array) -> None:
    if len(source) != INPUT_ELEMENTS or list(source) != list(range(1, INPUT_ELEMENTS + 1)):
        raise ValueError("Input must contain all 128 exact consecutive values 1..128")


def reference_output(case: dict, source: array.array) -> list[int]:
    validate_case(case)
    validate_input(source)
    values = list(source[:case["n"]]) + [0] * (case["block"] - case["n"])
    per_thread = [[] for _ in values]
    for width in (64, 32):
        # Explicit subgroup slices; no hardware lane bit manipulation is used.
        groups = [values[start:start + width] for start in range(0, len(values), width)]
        for group_index, group in enumerate(groups):
            selected = [group[source_lane % len(group)] for source_lane in (0, 31, 32, 63)]
            for thread in range(group_index * width, (group_index + 1) * width):
                per_thread[thread].extend(selected)
    for start in range(0, len(values), 64):
        total = sum(values[start:start + 64])
        for row in per_thread[start:start + 64]:
            row.append(total)
    return [value for row in per_thread for value in row]


def validate_output(case: dict, source: array.array, output: array.array) -> dict:
    expected = reference_output(case, source)
    if len(output) != len(expected) + 2 * GUARD_ELEMENTS:
        raise ValueError(f"{case['id']}: incorrect output extent")
    for side, guard in (("prefix", output[:GUARD_ELEMENTS]), ("suffix", output[-GUARD_ELEMENTS:])):
        if any(word != GUARD_WORD for word in guard):
            raise ValueError(f"{case['id']}: {side} guard overwritten")
    mismatches = [index for index, value in enumerate(expected) if output[GUARD_ELEMENTS + index] != value]
    if mismatches:
        index = mismatches[0]
        raise ValueError(f"{case['id']}: {len(mismatches)} mismatches; first thread={index // 9}, channel={index % 9}")
    return dict(payload_elements_checked=len(expected), physical_threads_checked=case["block"],
                channels_per_thread=9, zero_outputs_checked=expected.count(0), guard_elements_checked=64,
                mismatches=0)


def case_metadata(case: dict) -> dict:
    validate_case(case)
    return dict(block=case["block"], n=case["n"], physical_threads=case["block"], logical_threads=case["n"],
                zero_padded_threads=case["block"]-case["n"], wave_size=64, wave_count=case["block"]//64,
                block_x=case["block"], block_y=1, block_z=1, grid_x=1, grid_y=1, grid_z=1,
                mask_hex=MASK_HEX, mask_bits=64, participation="all physical threads", channels_per_thread=9,
                output_elements=case["block"]*9, output_layout="thread-major", output_file=case["id"]+".i32",
                warmups=10, samples=10, launches_per_sample=10, total_launches=110)


def validate_records(records: list[dict], cases: list[dict]) -> dict[str, list[dict]]:
    for kind in ("device", "protocol", "complete"):
        if sum(row.get("type") == kind for row in records) != 1:
            raise ValueError(f"Expected exactly one {kind} record")
    if records[-1].get("type") != "complete":
        raise ValueError("Run is incomplete")
    device = next(row for row in records if row["type"] == "device")
    protocol = next(row for row in records if row["type"] == "protocol")
    if (device.get("name") != "MetaX C550" or type(device.get("visible_device_count")) is not int
            or device["visible_device_count"] != 1 or type(device.get("wave_size_api")) is not int
            or device["wave_size_api"] != 64):
        raise ValueError("Expected exactly one visible MetaX C550 with observed wave size 64")
    if not isinstance(device.get("pci_bus_id"), str) or not device["pci_bus_id"]:
        raise ValueError("Missing PCI identity")
    for field in ("runtime_version_api", "driver_version_api"):
        if type(device.get(field)) is not int or device[field] < 0:
            raise ValueError(f"Missing or invalid device field: {field}")
    expected_protocol = dict(schema_version=1, experiment=EXPERIMENT, input_elements=128,
                             dtype="int32", guard_elements_each_side=32, timer="mcEventElapsedTime",
                             required_wave_size=64, mask_hex=MASK_HEX, mask_bits=64,
                             channels=list(CHANNELS), participation="all physical threads")
    for field, value in expected_protocol.items():
        if type(protocol.get(field)) is not type(value) or protocol[field] != value:
            raise ValueError(f"Protocol metadata mismatch: {field}")
    for field in ENVIRONMENT:
        if field not in protocol or (protocol[field] is not None and not isinstance(protocol[field], str)):
            raise ValueError(f"Missing or invalid environment field: {field}")
    if (type(records[-1].get("cases")) is not int or records[-1]["cases"] != len(cases)
            or records[-1].get("cpu_correctness_checked") is not False):
        raise ValueError("Completion metadata mismatch")
    by_id = {case["id"]: case for case in cases}
    declarations, samples = [], {key: [] for key in by_id}
    for record in records:
        kind = record.get("type")
        if kind in ("device", "protocol", "complete"):
            continue
        key = record.get("id")
        if key not in by_id:
            raise ValueError(f"Unknown case: {key}")
        if kind == "case":
            if key in declarations:
                raise ValueError(f"Duplicate case: {key}")
            for field, value in case_metadata(by_id[key]).items():
                if type(record.get(field)) is not type(value) or record[field] != value:
                    raise ValueError(f"Case metadata mismatch: {key}: {field}")
            attributes = record.get("function_attributes_before_timing")
            if (not isinstance(attributes, dict) or any(type(attributes.get(field)) is not int or attributes[field] < minimum
                    for field, minimum in (("maxThreadsPerBlock", 1), ("numRegs", 0), ("sharedSizeBytes", 0), ("localSizeBytes", 0)))):
                raise ValueError(f"Missing or invalid function attributes: {key}")
            declarations.append(key)
        elif kind == "sample":
            if not declarations or key != declarations[-1]:
                raise ValueError("Sample precedes its case or is interleaved")
            for field in ("event_batch_ms", "host_enqueue_batch_us"):
                value = record.get(field)
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid {field}: {key}")
            samples[key].append(record)
        else:
            raise ValueError(f"Unknown record type: {kind}")
    if declarations != list(by_id):
        raise ValueError("Missing or misordered case declarations")
    for key in by_id:
        indices = [sample.get("sample") for sample in samples[key]]
        if any(type(index) is not int for index in indices) or indices != list(range(10)):
            raise ValueError(f"Missing, duplicate or misordered samples: {key}")
    return samples


def check(input_directory: Path, output_directory: Path) -> dict:
    require_int32_little_endian()
    if json.loads((input_directory / "oracle.json").read_text()) != oracle_metadata():
        raise ValueError("Unsupported oracle metadata")
    cases = read_plan(input_directory / "cases.tsv")
    source = read_words(input_directory / "input.i32", INPUT_ELEMENTS)
    validate_input(source)
    records = [json.loads(line) for line in (output_directory / "raw.jsonl").read_text().splitlines()]
    samples = validate_records(records, cases)
    declarations = {record["id"]: record for record in records if record["type"] == "case"}
    summaries = []
    for case in cases:
        row = dict(id=case["id"], **case_metadata(case))
        row["function_attributes_before_timing"] = declarations[case["id"]]["function_attributes_before_timing"]
        output = read_words(output_directory / row["output_file"], row["output_elements"] + 2 * GUARD_ELEMENTS)
        row.update(validate_output(case, source, output))
        timings = [sample["event_batch_ms"] * 1000 / 10 for sample in samples[case["id"]]]
        row.update(event_mean_per_launch_us_median=statistics.median(timings),
                   event_mean_per_launch_us_min=min(timings), event_mean_per_launch_us_max=max(timings), sample_count=10)
        summaries.append(row)
    return dict(status="pass", experiment=EXPERIMENT, input_elements_checked=128, cases_checked=len(cases),
                checker="independent subgroup list slices and exact integer sums; every physical thread/channel and both guards checked",
                timing_scope="descriptive full nine-collective kernel batch averages only; not per-intrinsic latency or an implementation comparison",
                cases=summaries)


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
