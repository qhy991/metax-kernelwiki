#!/usr/bin/env python3
"""Check a v2 q7 module collection with a closed native or retained-bitcode input."""
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

SCHEMA = "metax-kernelwiki.wmma-module-q7.v2"
IMAGE_KINDS = ("native-elf", "retained-bitcode-bundle")
BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
BUNDLE_ENTRIES = (
    ("host-x86_64-unknown-linux-gnu", 4096, 0),
    ("maca-mxc-metax-macahca--xcore1000-bc", 4096, 11216),
)
SYMBOLS = {
    "wmma": "_ZN12_GLOBAL__N_116wmma_tile_kernelEPK6__halfS2_Pfj",
    "scalar": "_ZN12_GLOBAL__N_118scalar_tile_kernelEPK6__halfS2_Pfj",
}
ORDERS, PHASES = q7.ORDERS, q7.PHASES
ENVIRONMENT = (*q7.ENVIRONMENT, "MACA_MODULE_LOADING")


def image_metadata(image_kind: str) -> dict:
    require(type(image_kind) is str and image_kind in IMAGE_KINDS, "Unsupported image kind")
    if image_kind == "native-elf":
        return dict(image_kind=image_kind, image_bytes=18232, image_format="ELF64LE", machine_raw=253)
    return dict(image_kind=image_kind, image_bytes=15324, image_format="CLANG_OFFLOAD_BUNDLE",
                bundle_entries=[dict(target=target, offset=offset, size=size) for target, offset, size in BUNDLE_ENTRIES],
                wrapped_bitcode_bytes=11216, inner_bitcode_bytes=11184)


def module_metadata(image_kind: str) -> dict:
    return dict(type="module", image_file="loaded-image.bin", **image_metadata(image_kind), load_api="mcModuleLoadData",
                launch_api="mcModuleLaunchKernel", argument_interface="kernelParams",
                extra_is_null=True, symbols=SYMBOLS.copy(), module_loaded=True)


def protocol_metadata(order: str, image_kind: str) -> dict:
    # Reuse expected fixed numerical fields, never rewrite observed records.
    image_metadata(image_kind)
    return dict(q7.protocol_metadata(order), schema=SCHEMA, image_kind=image_kind)


def variant_metadata(order: str, variant: str) -> dict:
    return dict(q7.variant_metadata(order, variant), launched_symbol=SYMBOLS[variant], argument_count=4)


def completion_metadata(order: str, image_kind: str) -> dict:
    image_metadata(image_kind)
    return dict(q7.completion_metadata(order), module_unloaded=True, image_kind=image_kind)


def validate_metadata(records: list[dict], image_kind: str) -> str:
    require(len(records) == 9 and all(type(row) is dict for row in records), "Expected exactly nine module metadata records")
    device, module, protocol = records[:3]
    fixed_device = dict(type="device", name="MetaX C550", logical_device=0, visible_device_count=1, wave_size_api=64)
    require(device.keys() == fixed_device.keys() | {"pci_bus_id", "runtime_version_api", "driver_version_api", "max_threads_per_block"}, "Device fields differ")
    for key, value in fixed_device.items():
        exact(device[key], value, "device." + key)
    require(type(device["pci_bus_id"]) is str and bool(device["pci_bus_id"]), "Missing PCI identity")
    for key, minimum in (("runtime_version_api", 0), ("driver_version_api", 0), ("max_threads_per_block", 256)):
        require(type(device[key]) is int and device[key] >= minimum, "Invalid device field: " + key)
    exact(module, module_metadata(image_kind), "module")
    order = protocol.get("order")
    require(type(order) is str and order in ORDERS, "Unsupported protocol order")
    require(protocol.keys() == protocol_metadata(order, image_kind).keys() | {"environment"}, "Protocol fields differ")
    exact({key: value for key, value in protocol.items() if key != "environment"}, protocol_metadata(order, image_kind), "module protocol")
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
    exact(records[8], completion_metadata(order, image_kind), "completion")
    return order


def validate_wrapped_bitcode(data: bytes, context: str = "Bitcode") -> None:
    """Check this retained wrapper shape, not arbitrary LLVM bitcode validity."""
    require(len(data) == 11216, context + ": expected 11216 wrapped bitcode bytes")
    require(struct.unpack_from("<5I", data) == (0x0b17c0de, 0, 20, 11184, 255), context + ": wrapper fields differ")
    require(data[20:24] == b"BC\xc0\xde", context + ": missing inner bitcode magic")
    require(data[11204:] == bytes(12), context + ": wrapper padding differs")


def inspect_image(data: bytes, image_kind: str, context: str = "Image") -> dict:
    """Pure bounded inspection of two admitted forms; no file or GPU access."""
    expected = image_metadata(image_kind)
    require(len(data) == expected["image_bytes"], context + ": incorrect image extent")
    if image_kind == "native-elf":
        require(data[:7] == b"\x7fELF\x02\x01\x01" and
                struct.unpack_from("<HHI", data, 16) == (3, 253, 1) and
                struct.unpack_from("<H", data, 52)[0] == 64,
                context + ": expected ELF64LE ET_DYN image with machine 253")
        return expected
    require(data[:24] == BUNDLE_MAGIC and struct.unpack_from("<Q", data, 24)[0] == 2,
            context + ": expected exactly two bundle entries")
    cursor = 32
    observed = []
    for target, wanted_offset, wanted_size in BUNDLE_ENTRIES:
        require(cursor + 24 <= len(data), context + ": truncated bundle descriptor")
        offset, size, name_size = struct.unpack_from("<QQQ", data, cursor)
        cursor += 24
        require(name_size == len(target) and cursor + name_size <= len(data), context + ": target extent differs")
        require(data[cursor:cursor + name_size] == target.encode("ascii"), context + ": target identifier differs")
        cursor += name_size
        require(offset <= len(data) and size <= len(data) - offset, context + ": payload outside image")
        require(offset == wanted_offset and size == wanted_size, context + ": payload offset or size differs")
        observed.append(dict(target=target, offset=offset, size=size))
    require(cursor == 145 and data[cursor:4096] == bytes(4096 - cursor), context + ": descriptor padding differs")
    require(data[15312:] == b"__FILE_END__", context + ": bundle trailer differs")
    validate_wrapped_bitcode(data[4096:15312], context + " payload")
    exact(observed, expected["bundle_entries"], context + " entries")
    return expected


def verify_image_payload(data: bytes, image_kind: str, expected_bitcode: bytes | None = None) -> dict:
    """CPU preparation gate: format plus equality to an independent BC reference."""
    inspected = inspect_image(data, image_kind)
    if image_kind == "native-elf":
        require(expected_bitcode is None, "Native image forbids an expected bitcode reference")
        return inspected
    require(expected_bitcode is not None, "Bitcode bundle requires an independent expected bitcode reference")
    validate_wrapped_bitcode(expected_bitcode, "Expected bitcode")
    require(data[4096:15312] == expected_bitcode, "Bundle payload differs from the expected retained bitcode")
    return dict(inspected, bitcode_bytes_checked=11216, expected_bitcode_equal=True)


def read_bitcode_reference(image_path: Path, expected_bitcode: Path | None) -> bytes | None:
    if expected_bitcode is None:
        return None
    require(not image_path.samefile(expected_bitcode), "Expected bitcode must be separate from the carrier file")
    return expected_bitcode.read_bytes()


def compare_image(directory: Path, image_kind: str, expected_image: Path, expected_bitcode: Path | None = None) -> dict:
    require((image_kind == "retained-bitcode-bundle") == (expected_bitcode is not None),
            "Expected bitcode is required only for the bitcode-bundle kind")
    loaded_path = directory / "loaded-image.bin"
    require(not loaded_path.samefile(expected_image), "Expected image must be separate from the retained loaded-image file")
    expected, loaded = expected_image.read_bytes(), loaded_path.read_bytes()
    inspect_image(expected, image_kind, "Expected image")
    inspect_image(loaded, image_kind, "Retained image")
    require(loaded == expected, "Retained module image differs from the expected image")
    if expected_bitcode is not None:
        require(not expected_image.samefile(expected_bitcode), "Expected carrier and bitcode must be separate files")
    inspected = verify_image_payload(loaded, image_kind, read_bitcode_reference(loaded_path, expected_bitcode))
    return dict(file="loaded-image.bin", **inspected, bytes_checked=len(loaded), expected_image_equal=True)


def check(directory: Path, image_kind: str, expected_image: Path, expected_bitcode: Path | None = None) -> dict:
    records = [json.loads(line, object_pairs_hook=q7.unique_object, parse_constant=q7.reject_constant)
               for line in (directory / "raw.jsonl").read_text().splitlines()]
    order = validate_metadata(records, image_kind)
    image = compare_image(directory, image_kind, expected_image, expected_bitcode)
    return dict(schema=SCHEMA, module_image=image, **q7.analyze_files(directory, order))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?")
    parser.add_argument("--image-kind", choices=IMAGE_KINDS, required=True)
    parser.add_argument("--expected-image", type=Path)
    parser.add_argument("--expected-bitcode", type=Path)
    parser.add_argument("--inspect-image", type=Path,
                        help="CPU-only image admission; no collection directory or numerical checks")
    args = parser.parse_args()
    if (args.image_kind == "retained-bitcode-bundle") != (args.expected_bitcode is not None):
        parser.error("--expected-bitcode is required for retained-bitcode-bundle and forbidden for native-elf")
    if args.inspect_image is not None:
        if args.directory is not None or args.expected_image is not None:
            parser.error("--inspect-image forbids a collection directory and --expected-image")
    elif args.directory is None or args.expected_image is None:
        parser.error("Post-run checking requires a directory and --expected-image")
    try:
        if args.inspect_image is not None:
            image = verify_image_payload(args.inspect_image.read_bytes(), args.image_kind,
                                         read_bitcode_reference(args.inspect_image, args.expected_bitcode))
            result = dict(schema="metax-kernelwiki.wmma-module-image-inspection.v1", status="pass",
                          image=image, module_loaded=False, numerical_correctness_checked=False,
                          scope="CPU-only image structure and declared bitcode-reference comparison; no module load, device execution or numerical acceptance")
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0
        result = check(args.directory, args.image_kind, args.expected_image, args.expected_bitcode)
        print(json.dumps(result, indent=2, allow_nan=False))
        return {"pass": 0, "numeric_failed": 1, "integrity_failed": 2}[result["status"]]
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(status="error", error_kind="structural", message=str(error))), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
