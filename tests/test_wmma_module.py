"""CPU-only module-checker fixtures; synthetic images are not device evidence."""
import copy
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "experiments/wmma_module/check.py"
SPEC = importlib.util.spec_from_file_location("wmma_module_check", MODULE)
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def synthetic_image():
    # Only the documented CPU header checks are exercised; this is not loadable code.
    data = bytearray(18232)
    data[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HHI", data, 16, 3, 253, 1)
    struct.pack_into("<H", data, 52, 64)
    data[4096:4100] = b"TEST"
    return bytes(data)


def exact_output():
    reference = Fraction(-7, 16) * Fraction(-1, 16) + Fraction(1, 16) * Fraction(-8, 16)
    payload = list(struct.unpack("<256I", struct.pack("<256f", float(reference), *([0.0] * 255))))
    return [0xffffffff] * 64 + payload + [0xffffffff] * 64


class WmmaModuleTest(unittest.TestCase):
    def fixture(self, parent, order="wmma-first"):
        output, expected = parent / "output", parent / "expected-image.elf"
        output.mkdir()
        expected.write_bytes(synthetic_image())
        (output / "loaded-image.elf").write_bytes(synthetic_image())
        fixed = {"a": [0xb700, 0x2c00] + [0] * 1022, "b": [0xac00, 0xb800] + [0] * 1022}
        for operand, words in fixed.items():
            (output / f"prepared.{operand}.f16").write_bytes(struct.pack("<1024H", *words))
        records = [dict(type="device", name="MetaX C550", logical_device=0, visible_device_count=1,
                        wave_size_api=64, pci_bus_id="synthetic-test-device", runtime_version_api=1,
                        driver_version_api=1, max_threads_per_block=512), check.module_metadata(),
                   dict(check.protocol_metadata(order), environment={name: None for name in check.ENVIRONMENT})]

        def snapshot(phase):
            records.append(check.q7.snapshot_metadata(order, phase))
            for operand, words in fixed.items():
                (output / f"{phase}.{operand}.f16").write_bytes(struct.pack("<1024H", *words))

        snapshot("before")
        for index, variant in enumerate(check.ORDERS[order]):
            records.append(dict(check.variant_metadata(order, variant),
                function_attributes_before_launch=dict(maxThreadsPerBlock=512, numRegs=16, sharedSizeBytes=0, localSizeBytes=0),
                pointer_alignment_observed_bytes=dict(a=256, b=512, c_payload=256)))
            (output / (variant + ".f32")).write_bytes(struct.pack("<384I", *exact_output()))
            snapshot("between" if index == 0 else "after")
        records.append(check.completion_metadata(order))
        self.save(output, records)
        return output, expected, records

    def save(self, output, records):
        (output / "raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))

    def cli(self, output, expected):
        return subprocess.run(["python3", str(MODULE), str(output), "--expected-image", str(expected)],
                              capture_output=True, text=True)

    def assert_structural(self, output, expected):
        result = self.cli(output, expected)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        error = json.loads(result.stderr)
        self.assertEqual((error["status"], error["error_kind"]), ("error", "structural"))

    def test_both_orders_exact_outputs_pass_and_match_image(self):
        for order in check.ORDERS:
            with self.subTest(order=order), tempfile.TemporaryDirectory() as directory:
                output, expected, records = self.fixture(Path(directory), order)
                result = check.check(output, expected)
                self.assertEqual(result["schema"], "metax-kernelwiki.wmma-module-q7.v1")
                self.assertEqual(result["status"], "pass")
                self.assertTrue(result["passed"] and result["integrity_passed"] and result["numeric_passed"])
                self.assertEqual(list(result["variants"]), list(check.ORDERS[order]))
                self.assertEqual(result["module_image"], dict(file="loaded-image.elf", image_format="ELF64LE",
                    machine_raw=253, bytes_checked=18232, expected_image_equal=True))
                self.assertEqual([records[i]["variant"] for i in (4, 6)], list(check.ORDERS[order]))
                self.assertEqual((result["prepared_input_halfwords_checked"], result["input_snapshot_halfwords_checked"],
                                  result["payload_elements_checked"], result["guard_elements_checked"]), (2048, 6144, 512, 256))
                self.assertEqual(self.cli(output, expected).returncode, 0)

    def test_symbols_are_the_inspected_native_symbols(self):
        inspection = json.loads((ROOT / "data/inspections/20261008-q7-binary.json").read_text())
        self.assertEqual(check.SYMBOLS, {row["variant"]: row["symbol"] for row in inspection["device_elf"]["kernels"]})
        self.assertEqual(check.q7.reference_output(), [-1 / 256] + [0.0] * 255)

    def test_either_numerical_failure_remains_failure_and_retains_both(self):
        for variant in ("wmma", "scalar"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                output, expected, _ = self.fixture(Path(directory))
                words = exact_output()
                words[64] ^= 1
                (output / (variant + ".f32")).write_bytes(struct.pack("<384I", *words))
                result = check.check(output, expected)
                self.assertEqual((result["status"], result["passed"], result["numeric_passed"]), ("numeric_failed", False, False))
                self.assertTrue(result["integrity_passed"])
                self.assertEqual(set(result["variants"]), {"wmma", "scalar"})
                self.assertEqual(result["variants"][variant]["mismatch_count"], 1)
                self.assertEqual(result["variants"][variant]["mismatches"][0]["expected_value"], -1 / 256)
                cli = self.cli(output, expected)
                self.assertEqual((cli.returncode, cli.stderr), (1, ""))

    def test_image_substitution_and_file_extent_are_structural(self):
        for target in ("retained", "expected"):
            for mutation in ("substitute", "truncate", "append", "missing"):
                with self.subTest(target=target, mutation=mutation), tempfile.TemporaryDirectory() as directory:
                    output, expected, _ = self.fixture(Path(directory))
                    path = output / "loaded-image.elf" if target == "retained" else expected
                    data = bytearray(path.read_bytes())
                    if mutation == "missing":
                        path.unlink()
                    else:
                        if mutation == "substitute":
                            data[4096] ^= 1  # Same valid basic header and length, different image.
                        elif mutation == "truncate":
                            data = data[:-1]
                        else:
                            data += b"\0"
                        path.write_bytes(data)
                    self.assert_structural(output, expected)

    def test_matching_invalid_images_still_refuse_basic_format(self):
        for offset, value in ((0, 0), (4, 1), (5, 2), (6, 0), (16, 2), (18, 62), (20, 0), (52, 0)):
            with self.subTest(offset=offset), tempfile.TemporaryDirectory() as directory:
                output, expected, _ = self.fixture(Path(directory))
                data = bytearray(synthetic_image())
                data[offset] = value
                expected.write_bytes(data)
                (output / "loaded-image.elf").write_bytes(data)
                self.assert_structural(output, expected)

    def test_expected_image_is_required_and_not_self_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            output, expected, _ = self.fixture(Path(directory))
            missing = subprocess.run(["python3", str(MODULE), str(output)], capture_output=True, text=True)
            self.assertEqual((missing.returncode, missing.stdout), (2, ""))
            self.assertIn("--expected-image", missing.stderr)
            self.assert_structural(output, output / "loaded-image.elf")
            linked = Path(directory) / "hard-link.elf"
            os.link(output / "loaded-image.elf", linked)
            self.assert_structural(output, linked)

    def test_module_contract_symbol_argument_interface_and_typed_fields(self):
        mutations = [("load_api", "mcModuleLoad"), ("launch_api", "mcLaunchKernel"),
                     ("argument_interface", "extra"), ("extra_is_null", False), ("extra_is_null", 1),
                     ("image_bytes", True), ("image_bytes", 18232.0), ("image_format", "ELF32LE"),
                     ("machine_raw", 253.0), ("module_loaded", 1), ("module_loaded", False),
                     ("image_file", "other-image.elf"), ("symbols", {"wmma": check.SYMBOLS["scalar"], "scalar": check.SYMBOLS["wmma"]})]
        with tempfile.TemporaryDirectory() as directory:
            output, expected, records = self.fixture(Path(directory))
            for key, value in mutations:
                changed = copy.deepcopy(records)
                changed[1][key] = value
                self.save(output, changed)
                with self.subTest(key=key, value=value):
                    self.assert_structural(output, expected)
            for key in records[1]:
                changed = copy.deepcopy(records)
                del changed[1][key]
                with self.subTest(missing=key), self.assertRaises(check.StructuralError):
                    check.validate_metadata(changed)

    def test_protocol_variant_and_unload_cannot_masquerade_as_old_route(self):
        with tempfile.TemporaryDirectory() as directory:
            output, expected, records = self.fixture(Path(directory))
            for index, key, value in ((2, "schema", check.q7.SCHEMA), (2, "launches_per_variant", True),
                    (2, "total_launches", 220), (4, "argument_count", 3), (4, "argument_count", 4.0),
                    (4, "argument_count", True), (4, "launched_symbol", check.SYMBOLS["scalar"]),
                    (6, "c_allocation_slot", 0), (8, "module_unloaded", False), (8, "module_unloaded", 1)):
                changed = copy.deepcopy(records)
                changed[index][key] = value
                self.save(output, changed)
                with self.subTest(index=index, field=key, value=value):
                    self.assert_structural(output, expected)
            for index, key in ((4, "launched_symbol"), (6, "argument_count"), (8, "module_unloaded")):
                changed = copy.deepcopy(records)
                del changed[index][key]
                with self.subTest(missing=key), self.assertRaises(check.StructuralError):
                    check.validate_metadata(changed)

    def test_record_order_boundaries_extra_fields_and_resource_types_refuse(self):
        for order in check.ORDERS:
            with self.subTest(order=order), tempfile.TemporaryDirectory() as directory:
                output, expected, records = self.fixture(Path(directory), order)
                for mutation in ("module-order", "variant-order", "missing-complete", "duplicate-module",
                                 "neighbor", "missing-null", "attribute-bool", "alignment-float", "timing"):
                    changed = copy.deepcopy(records)
                    if mutation == "module-order": changed[1], changed[2] = changed[2], changed[1]
                    elif mutation == "variant-order": changed[4], changed[6] = changed[6], changed[4]
                    elif mutation == "missing-complete": changed.pop()
                    elif mutation == "duplicate-module": changed.insert(2, changed[1])
                    elif mutation == "neighbor": changed[5]["after_variant"] = changed[5]["before_variant"]
                    elif mutation == "missing-null": changed[3].pop("after_variant")
                    elif mutation == "attribute-bool": changed[4]["function_attributes_before_launch"]["numRegs"] = True
                    elif mutation == "alignment-float": changed[6]["pointer_alignment_observed_bytes"]["a"] = 256.0
                    else: changed[4]["elapsed_ms"] = 0.1
                    with self.subTest(mutation=mutation), self.assertRaises(check.StructuralError):
                        check.validate_metadata(changed)

    def test_shared_input_snapshot_guard_and_numeric_statuses_remain_distinct(self):
        for filename, width, count, index in (("prepared.a.f16", 2, 1024, 2), ("between.b.f16", 2, 1024, 1023),
                                              ("scalar.f32", 4, 384, 383)):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                output, expected, _ = self.fixture(Path(directory))
                path = output / filename
                fmt = "<" + ("H" if width == 2 else "I") * count
                words = list(struct.unpack(fmt, path.read_bytes()))
                words[index] ^= 1
                path.write_bytes(struct.pack(fmt, *words))
                result = check.check(output, expected)
                self.assertEqual(result["status"], "integrity_failed")
                self.assertFalse(result["passed"])
                self.assertTrue(result["numeric_passed"])
                self.assertEqual(self.cli(output, expected).returncode, 2)

    def test_truncated_output_and_duplicate_raw_keys_are_structural(self):
        for mutation in ("truncated", "duplicate", "nonfinite"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                output, expected, _ = self.fixture(Path(directory))
                if mutation == "truncated":
                    path = output / "wmma.f32"
                    path.write_bytes(path.read_bytes()[:-1])
                else:
                    (output / "raw.jsonl").write_text('{"type":"device","type":"device"}\n' if mutation == "duplicate" else "NaN\n")
                self.assert_structural(output, expected)

    def test_shared_file_analysis_preserves_old_schema_and_refusals(self):
        with tempfile.TemporaryDirectory() as directory:
            output, expected, records = self.fixture(Path(directory))
            with self.assertRaises(check.StructuralError):
                check.q7.check(output)  # The module protocol is never an old q7 protocol.
            old_records = [records[0], dict(check.q7.protocol_metadata("wmma-first"), environment=records[2]["environment"])]
            old_records += [records[3], {k: v for k, v in records[4].items() if k not in ("launched_symbol", "argument_count")},
                            records[5], {k: v for k, v in records[6].items() if k not in ("launched_symbol", "argument_count")},
                            records[7], check.q7.completion_metadata("wmma-first")]
            self.save(output, old_records)
            old_result = check.q7.check(output)
            self.assertEqual(old_result, dict(schema=check.q7.SCHEMA, **check.q7.analyze_files(output, "wmma-first")))
            self.assertEqual((old_result["schema"], old_result["status"]), ("metax-kernelwiki.wmma-q7-repro.v1", "pass"))
            self.assert_structural(output, expected)


if __name__ == "__main__":
    unittest.main()
