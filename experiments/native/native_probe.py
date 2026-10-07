#!/usr/bin/env python3
"""CPU-only preparation and exact checking for the standalone MACA probe."""
from __future__ import annotations

import argparse
import array
import csv
import json
import math
import statistics
import sys
from pathlib import Path

INPUT_ELEMENTS = 1 << 24
GUARD_ELEMENTS = 32
GUARD_WORD = 0xFFFFFFFF
COLUMNS = ("id", "kind", "n", "stride", "block", "warmups", "samples", "launches")


def default_cases() -> list[dict]:
    cases = []
    for block in (64, 128, 256, 512):
        for n in (1, 63, 64, 65, 127, 129, 511, 513, 4097, (1 << 22) + 13):
            cases.append(dict(id=f"copy_n{n}_b{block}", kind="copy", n=n, stride=1, block=block))
    for stride in (1, 2, 4, 8, 16):
        cases.append(dict(id=f"gather_n1048573_s{stride}_b256", kind="gather",
                          n=(1 << 20) - 3, stride=stride, block=256))
    for block in (64, 128, 256, 512):
        cases.append(dict(id=f"empty_b{block}", kind="empty", n=0, stride=0, block=block))
    return [dict(c, warmups=20, samples=10, launches=100) for c in cases]


def block_boundary_cases() -> list[dict]:
    cases = []
    for block in (512, 1024):
        for n in (1, 511, 513, 1023, 1024, 1025, 4097, (1 << 22) + 13):
            cases.append(dict(id=f"copy_n{n}_b{block}", kind="copy", n=n, stride=1, block=block))
    for block in (512, 1024):
        cases.append(dict(id=f"empty_b{block}", kind="empty", n=0, stride=0, block=block))
    return [dict(c, warmups=20, samples=10, launches=100) for c in cases]


def read_plan(path: Path, input_elements: int = INPUT_ELEMENTS) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("Unsupported cases.tsv header")
        rows = list(reader)
    if not rows or len(rows) > 64:
        raise ValueError("A plan must contain 1 to 64 cases")
    ids = set()
    cases = []
    for row in rows:
        if None in row or any(row[column] is None for column in COLUMNS):
            raise ValueError("Malformed plan row")
        c = {k: row[k] if k in ("id", "kind") else int(row[k]) for k in COLUMNS}
        if not c["id"] or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                             for ch in c["id"]) or c["id"] in ids:
            raise ValueError("Invalid or duplicate case id")
        ids.add(c["id"])
        if c["block"] not in (64, 128, 256, 512, 1024):
            raise ValueError("Unsupported block")
        if (c["warmups"], c["samples"], c["launches"]) != (20, 10, 100):
            raise ValueError("Unsupported timing protocol")
        if c["kind"] == "empty":
            if c["n"] != 0 or c["stride"] != 0:
                raise ValueError("Invalid empty case")
        elif c["kind"] in ("copy", "gather"):
            if (not 0 < c["n"] <= INPUT_ELEMENTS - 2 * GUARD_ELEMENTS
                    or not 1 <= c["stride"] <= 16
                    or (c["n"] - 1) * c["stride"] >= input_elements
                    or (c["kind"] == "copy" and c["stride"] != 1)):
                raise ValueError("Invalid input or output extent")
        else:
            raise ValueError("Unknown kernel kind")
        cases.append(c)
    return cases


def require_binary32_little_endian() -> None:
    if sys.byteorder != "little" or array.array("f").itemsize != 4 or array.array("I").itemsize != 4:
        raise ValueError("This format requires little-endian float32 and uint32")


def prepare(destination: Path, suite: str = "default") -> dict:
    require_binary32_little_endian()
    if suite not in ("default", "block-boundary"):
        raise ValueError("Unknown preparation suite")
    destination.mkdir(parents=True, exist_ok=False)
    cases = default_cases() if suite == "default" else block_boundary_cases()
    with (destination / "input.f32").open("wb") as handle:
        # All values are finite, unique and exactly representable in float32.
        chunk = 1 << 18
        for start in range(0, INPUT_ELEMENTS, chunk):
            array.array("f", range(start, min(start + chunk, INPUT_ELEMENTS))).tofile(handle)
    with (destination / "cases.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(cases)
    oracle = {
        "schema_version": 1,
        "input_file": "input.f32",
        "input_elements": INPUT_ELEMENTS,
        "input_dtype": "little-endian IEEE-754 binary32",
        "input_rule": "input[i] = float32(i), 0 <= i < 16777216",
        "input_finite_count": INPUT_ELEMENTS,
        "input_unique_count": INPUT_ELEMENTS,
        "output_rule": "copy: output[i] = input[i]; gather: output[i] = input[i * stride]",
        "guard_elements_each_side": GUARD_ELEMENTS,
        "guard_uint32": GUARD_WORD,
        "comparison": "bitwise equality for all payload elements and guards",
    }
    (destination / "oracle.json").write_text(json.dumps(oracle, indent=2) + "\n")
    return {"prepared": str(destination), "suite": suite, "cases": len(cases),
            "input_bytes": INPUT_ELEMENTS * 4}


def read_words(path: Path, count: int) -> array.array:
    if path.stat().st_size != count * 4:
        raise ValueError(f"{path.name}: expected {count * 4} bytes, found {path.stat().st_size}")
    words = array.array("I")
    with path.open("rb") as handle:
        words.fromfile(handle, count)
    return words


def validate_input(source: array.array) -> None:
    floats = memoryview(source).cast("B").cast("f")
    for index, value in enumerate(floats):
        if not math.isfinite(value) or value != index:
            raise ValueError(f"Input oracle failed at index {index}: {value!r}")
    if source[0] != 0:
        raise ValueError("Input zero must have the positive-zero bit pattern")


def validate_output(c: dict, source: array.array, output: array.array) -> dict:
    n = c["n"]
    if len(output) != n + 2 * GUARD_ELEMENTS:
        raise ValueError(f"{c['id']}: incorrect output length")
    if any(word != GUARD_WORD for word in output[:GUARD_ELEMENTS]):
        raise ValueError(f"{c['id']}: prefix guard overwritten")
    if any(word != GUARD_WORD for word in output[-GUARD_ELEMENTS:]):
        raise ValueError(f"{c['id']}: suffix guard overwritten")
    floats = memoryview(output).cast("B").cast("f")
    mismatches = 0
    finite_count = 0
    first_mismatch = None
    for index in range(n):
        actual = output[GUARD_ELEMENTS + index]
        expected = source[index * c["stride"]]
        finite_count += math.isfinite(floats[GUARD_ELEMENTS + index])
        if actual != expected:
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = index
    if mismatches or finite_count != n:
        raise ValueError(f"{c['id']}: {mismatches} mismatches, finite_count={finite_count}/{n}, "
                         f"first_mismatch={first_mismatch}")
    return {"payload_elements_checked": n, "finite_count": finite_count,
            "unique_input_indices": n, "guard_elements_checked": 2 * GUARD_ELEMENTS,
            "mismatches": 0}


def validate_records(records: list[dict], cases: list[dict], input_elements: int) -> dict[str, list[dict]]:
    for kind in ("device", "protocol", "complete"):
        if sum(record.get("type") == kind for record in records) != 1:
            raise ValueError(f"Expected exactly one {kind} record")
    if records[-1].get("type") != "complete":
        raise ValueError("Device run did not complete")
    device = next(r for r in records if r["type"] == "device")
    protocol = next(r for r in records if r["type"] == "protocol")
    complete = records[-1]
    if (protocol.get("schema_version") != 1 or protocol.get("input_elements") != input_elements
            or protocol.get("guard_elements_each_side") != GUARD_ELEMENTS
            or protocol.get("timer") != "mcEventElapsedTime"
            or complete.get("cases") != len(cases)
            or complete.get("cpu_correctness_checked") is not False):
        raise ValueError("Run protocol does not match the prepared input")
    if device.get("name") != "MetaX C550" or device.get("visible_device_count") != 1:
        raise ValueError("Expected exactly one visible MetaX C550")
    if type(device.get("wave_size_api")) is not int or device["wave_size_api"] <= 0:
        raise ValueError("Missing or invalid device wave size")
    if "copy_launch_bound" in protocol:
        value = protocol["copy_launch_bound"]
        if type(value) is not int or value not in (0, 1024):
            raise ValueError("Invalid copy_launch_bound protocol field")
    record_first_launch = protocol.get("record_first_launch", False)
    additional_launches = protocol.get("additional_launches_per_case", 0)
    if (type(record_first_launch) is not bool
            or type(additional_launches) is not int
            or additional_launches != int(record_first_launch)):
        raise ValueError("Invalid first-launch protocol")
    if record_first_launch:
        for field in ("MACA_CACHE_PATH", "MACA_CACHE_DISABLE"):
            if field not in protocol or (protocol[field] is not None and not isinstance(protocol[field], str)):
                raise ValueError(f"Missing or invalid first-launch protocol field: {field}")
    by_id = {c["id"]: c for c in cases}
    declarations = {}
    first_launches = {}
    samples = {key: [] for key in by_id}
    for record in records:
        kind = record.get("type")
        if kind in ("device", "protocol", "complete"):
            continue
        key = record.get("id")
        if key not in by_id:
            raise ValueError(f"Unknown case in records: {key}")
        c = by_id[key]
        if kind == "case":
            if key in declarations:
                raise ValueError(f"Duplicate case declaration: {key}")
            expected = {k: c[k] for k in ("kind", "n", "stride", "block", "warmups", "samples")}
            expected.update(launches_per_sample=c["launches"], logical_bytes_per_launch=c["n"] * 8,
                            unique_input_elements=c["n"],
                            input_span_bytes=((c["n"] - 1) * c["stride"] + 1) * 4 if c["n"] else 0,
                            grid=(c["n"] + c["block"] - 1) // c["block"] if c["n"] else 1,
                            output_file=f"{key}.f32" if c["kind"] != "empty" else None)
            if any(record.get(k) != value for k, value in expected.items()):
                raise ValueError(f"Case metadata mismatch: {key}")
            declarations[key] = record
        elif kind == "first_launch":
            if not record_first_launch:
                raise ValueError(f"Unexpected first-launch record: {key}")
            if key not in declarations or key in first_launches or samples[key]:
                raise ValueError(f"Misordered or duplicate first-launch record: {key}")
            value = record.get("first_launch_host_complete_us")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid first_launch_host_complete_us: {key}")
            attributes = record.get("function_attributes_after_first_launch")
            if (not isinstance(attributes, dict)
                    or any(type(attributes.get(field)) is not int or attributes[field] < minimum
                           for field, minimum in (("maxThreadsPerBlock", 1), ("numRegs", 0),
                                                  ("sharedSizeBytes", 0), ("localSizeBytes", 0)))):
                raise ValueError(f"Missing or invalid function attributes: {key}")
            first_launches[key] = record
        elif kind == "sample":
            if key not in declarations:
                raise ValueError(f"Sample precedes declaration: {key}")
            if record_first_launch and key not in first_launches:
                raise ValueError(f"Sample precedes first-launch record: {key}")
            for field in ("event_batch_ms", "host_enqueue_batch_us"):
                value = record.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid {field}: {key}")
            samples[key].append(record)
        else:
            raise ValueError(f"Unknown record type: {kind}")
    if set(declarations) != set(by_id):
        raise ValueError("Missing case declaration")
    if record_first_launch and set(first_launches) != set(by_id):
        raise ValueError("Missing first-launch record")
    for key, c in by_id.items():
        ids = [row.get("sample") for row in samples[key]]
        if any(type(index) is not int for index in ids) or sorted(ids) != list(range(c["samples"])):
            raise ValueError(f"Missing or duplicated samples: {key}")
    return samples


def check(input_directory: Path, output_directory: Path) -> dict:
    require_binary32_little_endian()
    oracle = json.loads((input_directory / "oracle.json").read_text())
    if (oracle.get("schema_version") != 1 or oracle.get("input_elements") != INPUT_ELEMENTS
            or oracle.get("guard_elements_each_side") != GUARD_ELEMENTS
            or oracle.get("guard_uint32") != GUARD_WORD):
        raise ValueError("Unsupported oracle")
    cases = read_plan(input_directory / "cases.tsv")
    source = read_words(input_directory / "input.f32", INPUT_ELEMENTS)
    validate_input(source)
    records = [json.loads(line) for line in (output_directory / "raw.jsonl").read_text().splitlines()]
    samples = validate_records(records, cases, INPUT_ELEMENTS)
    first_launches = {r["id"]: r
                      for r in records if r["type"] == "first_launch"}
    summaries = []
    for c in cases:
        row = {"id": c["id"], "kind": c["kind"], "n": c["n"], "stride": c["stride"], "block": c["block"]}
        if c["id"] in first_launches:
            first = first_launches[c["id"]]
            row["first_launch_host_complete_us"] = first["first_launch_host_complete_us"]
            row["function_attributes_after_first_launch"] = first["function_attributes_after_first_launch"]
        if c["kind"] != "empty":
            output = read_words(output_directory / f"{c['id']}.f32", c["n"] + 2 * GUARD_ELEMENTS)
            row.update(validate_output(c, source, output))
        else:
            row["correctness_scope"] = "no payload; runtime launch and synchronization errors checked"
        timings = [sample["event_batch_ms"] * 1000 / c["launches"] for sample in samples[c["id"]]]
        row.update(event_mean_per_launch_us_median=statistics.median(timings),
                   event_mean_per_launch_us_min=min(timings), event_mean_per_launch_us_max=max(timings),
                   sample_count=len(timings), launches_per_sample=c["launches"])
        if c["n"]:
            row["logical_gbps_at_median"] = c["n"] * 8 / (statistics.median(timings) * 1000)
        summaries.append(row)
    result = {
        "status": "pass",
        "checker": "CPU-only bitwise oracle over every output element; guard words verified",
        "input_finite_count": INPUT_ELEMENTS,
        "cases_checked": len(cases),
        "timing_scope": "repeated addresses; no explicit application cache reset; runtime cache policy unverified; batch averages include host submission gaps",
        "throughput_scope": "logical GB/s in decimal units, not DRAM bandwidth",
        "cases": summaries,
    }
    protocol = next(r for r in records if r["type"] == "protocol")
    if "copy_launch_bound" in protocol:
        result["copy_launch_bound"] = protocol["copy_launch_bound"]
    if first_launches:
        result["first_launch_timing_scope"] = (
            "one additional launch per case before the 20 warmups; host monotonic clock around "
            "launch, error check and device synchronization; includes any lazy initialization/JIT "
            "triggered there; previous device work synchronized before timing; not pure GPU latency "
            "or a fresh process per case")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    preparation = command.add_parser("prepare")
    preparation.add_argument("directory", type=Path)
    preparation.add_argument("--suite", choices=("default", "block-boundary"), default="default")
    checker = command.add_parser("check")
    checker.add_argument("input_directory", type=Path)
    checker.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    try:
        result = prepare(args.directory, args.suite) if args.command == "prepare" else check(args.input_directory, args.output_directory)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "error", "message": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
