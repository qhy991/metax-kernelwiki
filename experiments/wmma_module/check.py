#!/usr/bin/env python3
"""Check a retained native-module q7 collection without loading a device runtime."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import struct
import sys

_HELPER = Path(__file__).resolve().parents[1] / "wmma_q7" / "check.py"
_SPEC = importlib.util.spec_from_file_location("wmma_q7_file_checks", _HELPER)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("Cannot load the fixed-q7 CPU file checks")
q7 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(q7)
require, exact, StructuralError = q7.require, q7.exact, q7.StructuralError

SCHEMA = "metax-kernelwiki.wmma-module-q7.v1"
IMAGE_BYTES = 18232
SYMBOLS = {
    "wmma": "_ZN12_GLOBAL__N_116wmma_tile_kernelEPK6__halfS2_Pfj",
    "scalar": "_ZN12_GLOBAL__N_118scalar_tile_kernelEPK6__halfS2_Pfj",
}
ORDERS, PHASES, ENVIRONMENT = q7.ORDERS, q7.PHASES, q7.ENVIRONMENT


def module_metadata() -> dict:
    return dict(type="module", image_file="loaded-image.elf", image_bytes=IMAGE_BYTES,
                image_format="ELF64LE", machine_raw=253, load_api="mcModuleLoadData",
                launch_api="mcModuleLaunchKernel", argument_interface="kernelParams",
                extra_is_null=True, symbols=SYMBOLS.copy(), module_loaded=True)


def protocol_metadata(order: str) -> dict:
    # Reuse expected fixed numerical fields, never rewrite observed records.
    return dict(q7.protocol_metadata(order), schema=SCHEMA)


def variant_metadata(order: str, variant: str) -> dict:
    return dict(q7.variant_metadata(order, variant), launched_symbol=SYMBOLS[variant], argument_count=4)


def completion_metadata(order: str) -> dict:
    return dict(q7.completion_metadata(order), module_unloaded=True)


def validate_metadata(records: list[dict]) -> str:
    require(len(records) == 9 and all(type(row) is dict for row in records), "Expected exactly nine module metadata records")
    device, module, protocol = records[:3]
    fixed_device = dict(type="device", name="MetaX C550", logical_device=0, visible_device_count=1, wave_size_api=64)
    require(device.keys() == fixed_device.keys() | {"pci_bus_id", "runtime_version_api", "driver_version_api", "max_threads_per_block"}, "Device fields differ")
    for key, value in fixed_device.items():
        exact(device[key], value, "device." + key)
    require(type(device["pci_bus_id"]) is str and bool(device["pci_bus_id"]), "Missing PCI identity")
    for key, minimum in (("runtime_version_api", 0), ("driver_version_api", 0), ("max_threads_per_block", 256)):
        require(type(device[key]) is int and device[key] >= minimum, "Invalid device field: " + key)
    exact(module, module_metadata(), "module")
    order = protocol.get("order")
    require(type(order) is str and order in ORDERS, "Unsupported protocol order")
    require(protocol.keys() == protocol_metadata(order).keys() | {"environment"}, "Protocol fields differ")
    exact({key: value for key, value in protocol.items() if key != "environment"}, protocol_metadata(order), "module protocol")
    environment = protocol["environment"]
    require(type(environment) is dict and environment.keys() == set(ENVIRONMENT), "Environment fields differ")
    require(all(value is None or type(value) is str for value in environment.values()), "Invalid environment value")
    for position, phase in zip((3, 5, 7), PHASES):
        exact(records[position], q7.snapshot_metadata(order, phase), phase + " snapshot")
    for position, variant in zip((4, 6), ORDERS[order]):
        row = records[position]
        observed = {"function_attributes_before_launch", "pointer_alignment_observed_bytes"}
        require(row.keys() == variant_metadata(order, variant).keys() | observed, "Variant fields differ")
        exact({key: value for key, value in row.items() if key not in observed}, variant_metadata(order, variant), variant)
        attributes = row["function_attributes_before_launch"]
        require(type(attributes) is dict and attributes.keys() == {"maxThreadsPerBlock", "numRegs", "sharedSizeBytes", "localSizeBytes"}, "Function attribute fields differ")
        require(all(type(value) is int and value >= (1 if key == "maxThreadsPerBlock" else 0)
                    for key, value in attributes.items()), "Invalid function attribute")
        alignment = row["pointer_alignment_observed_bytes"]
        require(type(alignment) is dict and alignment.keys() == {"a", "b", "c_payload"}, "Alignment fields differ")
        require(all(type(value) is int and value > 0 and not value & (value - 1) for value in alignment.values()), "Invalid pointer alignment observation")
    for operand in ("a", "b"):
        exact(records[4]["pointer_alignment_observed_bytes"][operand],
              records[6]["pointer_alignment_observed_bytes"][operand], "Shared input alignment: " + operand)
    exact(records[8], completion_metadata(order), "completion")
    return order


def validate_image(data: bytes, context: str) -> None:
    require(len(data) == IMAGE_BYTES, context + ": image must contain exactly 18232 bytes")
    require(data[:7] == b"\x7fELF\x02\x01\x01" and
            struct.unpack_from("<HHI", data, 16) == (3, 253, 1) and
            struct.unpack_from("<H", data, 52)[0] == 64,
            context + ": expected ELF64LE ET_DYN image with machine 253")


def compare_image(directory: Path, expected_image: Path) -> dict:
    loaded_path = directory / "loaded-image.elf"
    require(not loaded_path.samefile(expected_image), "Expected image must be separate from the retained loaded-image file")
    expected, loaded = expected_image.read_bytes(), loaded_path.read_bytes()
    validate_image(expected, "Expected image")
    validate_image(loaded, "Retained image")
    require(loaded == expected, "Retained module image differs from the expected image")
    return dict(file="loaded-image.elf", image_format="ELF64LE", machine_raw=253,
                bytes_checked=IMAGE_BYTES, expected_image_equal=True)


def check(directory: Path, expected_image: Path) -> dict:
    records = [json.loads(line, object_pairs_hook=q7.unique_object, parse_constant=q7.reject_constant)
               for line in (directory / "raw.jsonl").read_text().splitlines()]
    order = validate_metadata(records)
    image = compare_image(directory, expected_image)
    return dict(schema=SCHEMA, module_image=image, **q7.analyze_files(directory, order))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-image", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = check(args.directory, args.expected_image)
        print(json.dumps(result, indent=2, allow_nan=False))
        return {"pass": 0, "numeric_failed": 1, "integrity_failed": 2}[result["status"]]
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(status="error", error_kind="structural", message=str(error))), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
