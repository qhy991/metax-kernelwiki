#!/usr/bin/env python3
"""Check a retained single-launch q7 collection, without importing a GPU runtime."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import struct
import sys

SCHEMA = "metax-kernelwiki.wmma-q7-repro.v1"
ORDERS = {"wmma-first": ("wmma", "scalar"), "scalar-first": ("scalar", "wmma")}
PHASES = ("before", "between", "after")
ENVIRONMENT = ("MACA_LAUNCH_MODE", "MACA_LAUNCH_BLOCKING", "MACA_DIRECT_DISPATCH", "MACA_CACHE_PATH", "MACA_CACHE_DISABLE")
INPUT_COUNT, PAYLOAD_WORDS, GUARD_WORDS, OUTPUT_WORDS = 1024, 256, 64, 384
SENTINEL = 0xFFFFFFFF
INPUT_PREFIX = {"a": [0xB700, 0x2C00], "b": [0xAC00, 0xB800]}


class StructuralError(ValueError):
    """The retained files cannot be attributed to this complete contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StructuralError(message)


def exact(actual, expected, context: str) -> None:
    require(type(actual) is type(expected), context + ": wrong type")
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), context + ": missing or extra fields")
        for key, value in expected.items():
            exact(actual[key], value, context + "." + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), context + ": wrong length")
        for index, (left, right) in enumerate(zip(actual, expected)):
            exact(left, right, context + f"[{index}]")
    else:
        require(actual == expected, context + ": wrong value")


def fixed_inputs() -> dict[str, list[int]]:
    return {operand: prefix + [0] * 1022 for operand, prefix in INPUT_PREFIX.items()}


def reference_output() -> list[float]:
    # Independent logical arithmetic; do not multiply observed device/host inputs.
    target = Fraction(-7, 16) * Fraction(-1, 16) + Fraction(1, 16) * Fraction(-8, 16)
    return [float(target)] + [0.0] * 255


def protocol_metadata(order: str) -> dict:
    require(order in ORDERS, "Unsupported order")
    return dict(type="protocol", schema=SCHEMA, order=order, variant_order=list(ORDERS[order]),
                logical_shape=[16, 16, 2], tile=[16, 16, 16], packed_chunks=4, k_chunks=1,
                operand_halfwords_each=1024, input_words_prefix={op: words.copy() for op, words in INPUT_PREFIX.items()}, input_padding_uint16=0,
                prepared_files={"a": "prepared.a.f16", "b": "prepared.b.f16"},
                layouts={"a": "row_major", "b": "col_major", "c": "row_major"}, leading_dimension=16,
                operand_dtype="float16", accumulator_dtype="float32", output_dtype="float32",
                output_elements=256, guard_elements_each_side=64, guard_uint32=SENTINEL,
                initial_payload_uint32=SENTINEL, launches_per_variant=1, total_launches=2, warmups=0,
                input_rewrite_between_variants=False, snapshot_phases=list(PHASES),
                comparison="all 256 outputs finite and numerically exact; signed zeros equivalent; no tolerance")


def snapshot_metadata(order: str, phase: str) -> dict:
    require(order in ORDERS and phase in PHASES, "Unknown snapshot binding")
    position = PHASES.index(phase)
    variants = ORDERS[order]
    return dict(type="input_snapshot", order=order, phase=phase,
                after_variant=variants[position - 1] if position else None,
                before_variant=variants[position] if position < 2 else None,
                operand_halfwords_each=1024, a_file=phase + ".a.f16", b_file=phase + ".b.f16")


def variant_metadata(order: str, variant: str) -> dict:
    require(order in ORDERS and variant in ("wmma", "scalar"), "Unknown variant binding")
    return dict(type="variant", order=order, variant=variant, kernel=variant + "_tile_kernel",
                block=[64 if variant == "wmma" else 256, 1, 1], grid=[1, 1, 1], k_chunks=1,
                launch_count=1, c_allocation_slot=0 if variant == "wmma" else 1,
                output_file=variant + ".f32", output_words=384, payload_offset_words=64,
                payload_elements=256, guard_elements_each_side=64)


def completion_metadata(order: str) -> dict:
    require(order in ORDERS, "Unsupported completion order")
    return dict(type="complete", order=order, variant_count=2, total_launches=2,
                outputs_retained=True, device_buffers_freed=True, cpu_correctness_checked=False)


def validate_metadata(records: list[dict]) -> str:
    require(len(records) == 8 and all(type(row) is dict for row in records), "Expected exactly eight metadata records")
    device, protocol = records[:2]
    fixed_device = dict(type="device", name="MetaX C550", logical_device=0, visible_device_count=1, wave_size_api=64)
    require(device.keys() == fixed_device.keys() | {"pci_bus_id", "runtime_version_api", "driver_version_api", "max_threads_per_block"}, "Device fields differ")
    for key, value in fixed_device.items():
        exact(device[key], value, "device." + key)
    require(type(device["pci_bus_id"]) is str and bool(device["pci_bus_id"]), "Missing PCI identity")
    for key, minimum in (("runtime_version_api", 0), ("driver_version_api", 0), ("max_threads_per_block", 256)):
        require(type(device[key]) is int and device[key] >= minimum, "Invalid device field: " + key)
    order = protocol.get("order")
    require(type(order) is str and order in ORDERS, "Unsupported protocol order")
    require(protocol.keys() == protocol_metadata(order).keys() | {"environment"}, "Protocol fields differ")
    exact({key: value for key, value in protocol.items() if key != "environment"}, protocol_metadata(order), "protocol")
    environment = protocol["environment"]
    require(type(environment) is dict and environment.keys() == set(ENVIRONMENT), "Environment fields differ")
    require(all(value is None or type(value) is str for value in environment.values()), "Invalid environment value")
    for position, phase in zip((2, 4, 6), PHASES):
        exact(records[position], snapshot_metadata(order, phase), phase + " snapshot")
    for position, variant in zip((3, 5), ORDERS[order]):
        row = records[position]
        observed = {"function_attributes_before_launch", "pointer_alignment_observed_bytes"}
        require(row.keys() == variant_metadata(order, variant).keys() | observed, "Variant fields differ")
        exact({key: value for key, value in row.items() if key not in observed}, variant_metadata(order, variant), variant)
        attributes = row["function_attributes_before_launch"]
        require(type(attributes) is dict and attributes.keys() == {"maxThreadsPerBlock", "numRegs", "sharedSizeBytes", "localSizeBytes"}, "Function attribute fields differ")
        for key, value in attributes.items():
            require(type(value) is int and value >= (1 if key == "maxThreadsPerBlock" else 0), "Invalid function attribute")
        alignment = row["pointer_alignment_observed_bytes"]
        require(type(alignment) is dict and alignment.keys() == {"a", "b", "c_payload"}, "Alignment fields differ")
        require(all(type(value) is int and value > 0 and not value & (value - 1) for value in alignment.values()), "Invalid pointer alignment observation")
    for operand in ("a", "b"):
        exact(records[3]["pointer_alignment_observed_bytes"][operand],
              records[5]["pointer_alignment_observed_bytes"][operand], "Shared input alignment: " + operand)
    exact(records[7], completion_metadata(order), "completion")
    return order


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON field: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise StructuralError("Nonfinite JSON constant: " + value)


def read_words(path: Path, count: int, width: int) -> list[int]:
    require(width in (2, 4), "Unsupported word width")
    data = path.read_bytes()
    require(len(data) == count * width, path.name + ": incorrect file extent")
    return list(struct.unpack("<" + ("H" if width == 2 else "I") * count, data))


def compare_words(actual: list[int], expected: list[int]) -> dict:
    require(len(actual) == len(expected), "Mismatched word counts")
    indices = [index for index, (left, right) in enumerate(zip(actual, expected)) if left != right]
    return dict(passed=not indices, halfwords_checked=len(actual), mismatch_count=len(indices), mismatch_indices=indices)


def analyze_output(words: list[int]) -> dict:
    require(len(words) == OUTPUT_WORDS, "Incorrect output extent")
    values = struct.unpack("<256f", struct.pack("<256I", *words[64:-64]))
    mismatches = []
    for index, (actual, expected) in enumerate(zip(values, reference_output())):
        if not math.isfinite(actual) or actual != expected:
            mismatches.append(dict(row=index // 16, col=index % 16, observed_word_uint32=words[64 + index],
                                   observed_value=actual if math.isfinite(actual) else None, expected_value=expected,
                                   kind="finite_unequal" if math.isfinite(actual) else "nonfinite"))
    guards = [dict(side=side, index=index, observed_word_uint32=value)
              for side, data in (("prefix", words[:64]), ("suffix", words[-64:]))
              for index, value in enumerate(data) if value != SENTINEL]
    return dict(exact_passed=not mismatches, payload_elements_checked=256,
                finite_count=sum(math.isfinite(value) for value in values), mismatch_count=len(mismatches),
                mismatches=mismatches, guards_intact=not guards, guard_elements_checked=128, guard_mismatches=guards)


def analyze_files(directory: Path, order: str) -> dict:
    """Check shared fixed-q7 files after the caller validates its own protocol."""
    require(type(order) is str and order in ORDERS, "Unsupported file-analysis order")
    fixed = fixed_inputs()
    prepared = {op: read_words(directory / f"prepared.{op}.f16", 1024, 2) for op in ("a", "b")}
    inputs = {op: compare_words(prepared[op], fixed[op]) for op in ("a", "b")}
    snapshots = {}
    for phase in PHASES:
        snapshots[phase] = {}
        for op in ("a", "b"):
            actual = read_words(directory / f"{phase}.{op}.f16", 1024, 2)
            snapshots[phase][op] = dict(against_fixed=compare_words(actual, fixed[op]),
                                       against_prepared=compare_words(actual, prepared[op]))
    variants = {variant: analyze_output(read_words(directory / f"{variant}.f32", 384, 4)) for variant in ORDERS[order]}
    integrity = (all(row["passed"] for row in inputs.values())
                 and all(row["against_fixed"]["passed"] and row["against_prepared"]["passed"]
                         for phase in snapshots.values() for row in phase.values())
                 and all(row["guards_intact"] for row in variants.values()))
    numeric = all(row["exact_passed"] for row in variants.values())
    return dict(status="integrity_failed" if not integrity else "numeric_failed" if not numeric else "pass",
                passed=integrity and numeric, structural_valid=True, integrity_passed=integrity, numeric_passed=numeric,
                order=order, prepared_inputs=inputs, snapshots=snapshots, variants=variants,
                prepared_input_halfwords_checked=2048, input_snapshot_halfwords_checked=6144,
                payload_elements_checked=512, guard_elements_checked=256,
                reference="independent Fraction arithmetic: (-7/16)*(-1/16)+(1/16)*(-8/16)=-1/256 at C00; other 255 outputs zero",
                input_observation_scope="before/between/after capture boundaries only; not transient values inside a kernel")


def check(directory: Path) -> dict:
    records = [json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
               for line in (directory / "raw.jsonl").read_text().splitlines()]
    order = validate_metadata(records)
    return dict(schema=SCHEMA, **analyze_files(directory, order))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        result = check(args.directory)
        print(json.dumps(result, indent=2, allow_nan=False))
        return {"pass": 0, "numeric_failed": 1, "integrity_failed": 2}[result["status"]]
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(status="error", error_kind="structural", message=str(error))), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
