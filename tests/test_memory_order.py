"""CPU-only protocol and oracle tests; no claims about MACA compilation or execution."""
import array
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "memory_order", Path(__file__).resolve().parents[1] / "experiments/memory_order/experiment.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def words(values):
    result = array.array("I")
    result.frombytes(array.array("f", values).tobytes())
    return result


class MemoryOrderTest(unittest.TestCase):
    def setUp(self):
        self.source = words(range(128))
        self.case = dict(id="order_test", n=128, shift=2, block=256, warmups=10, samples=10, launches=10)
        # Construct the reference through an actual matrix transpose, not bit rotation.
        matrix = [self.source[start:start + 4] for start in range(0, 128, 4)]
        payload = array.array("I", (value for row in zip(*matrix) for value in row))
        self.output = (array.array("I", [probe.GUARD_WORD] * probe.GUARD_ELEMENTS)
                       + payload + array.array("I", [probe.GUARD_WORD] * probe.GUARD_ELEMENTS))

    def test_exhaustive_small_permutations_are_invertible_and_cover_every_address(self):
        for log2_n in range(1, 11):
            n = 2 ** log2_n
            for shift in range(log2_n):
                indices = list(probe.reference_indices(n, shift))
                with self.subTest(n=n, shift=shift):
                    self.assertEqual(sorted(indices), list(range(n)))
                    self.assertEqual([probe.inverse_index(j, n, shift) for j in indices], list(range(n)))
                    # Cross-check the GPU algebra only here; the oracle itself uses div/mod.
                    self.assertEqual(indices, [((i << shift) | (i >> (log2_n - shift))) & (n - 1)
                                               for i in range(n)])

    def test_invalid_permutation_parameters_are_refused(self):
        for n, shift in ((0, 0), (1, 0), (3, 0), (65535, 2), (1 << 25, 0),
                         (16, -1), (16, 4), (16, 5), (16, True), (16.0, 2)):
            with self.subTest(n=n, shift=shift), self.assertRaises(ValueError):
                list(probe.reference_indices(n, shift))

    def test_full_payload_and_guards_pass(self):
        probe.validate_input(self.source)
        result = probe.validate_output(self.case, self.source, self.output)
        self.assertEqual(result, dict(payload_elements_checked=128, finite_count=128,
                                     unique_input_indices=128, guard_elements_checked=64, mismatches=0))

    def test_wrong_order_and_unwritten_payload_are_refused(self):
        for mutation in ("sequential_copy", "wrong_position", "unwritten"):
            output = self.output[:]
            if mutation == "sequential_copy":
                output[32:160] = self.source
            elif mutation == "wrong_position":
                output[33] = self.source[1]
            else:
                output[159] = probe.GUARD_WORD
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "mismatches"):
                probe.validate_output(self.case, self.source, output)

    def test_prefix_suffix_and_extent_are_checked(self):
        for index in (0, 31, len(self.output) - 32, len(self.output) - 1):
            output = self.output[:]
            output[index] = 0
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "guard overwritten"):
                probe.validate_output(self.case, self.source, output)
        with self.assertRaisesRegex(ValueError, "extent"):
            probe.validate_output(self.case, self.source, self.output[:-1])

    def write_plan(self, directory, cases):
        path = Path(directory) / "cases.tsv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=probe.COLUMNS, delimiter="\t")
            writer.writeheader()
            writer.writerows(cases)
        return path

    def test_default_plan_has_fixed_footprints_and_110_launches(self):
        cases = probe.default_cases()
        self.assertEqual(len(cases), 18)
        self.assertEqual({(c["n"], c["shift"]) for c in cases},
                         {(n, shift) for n in (65536, 1048576, 16777216) for shift in (0, 2, 4, 6, 8, 12)})
        for case in cases:
            self.assertEqual(case["warmups"] + case["samples"] * case["launches"], 110)
            self.assertEqual(case["block"], 256)
            self.assertLessEqual((case["n"] + 64) * 4, (64 << 20) + 256)
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(probe.read_plan(self.write_plan(directory, cases)), cases)
            # A reordered subset is the same admitted experiment, useful for isolated traces.
            subset = [cases[-1], cases[0]]
            self.assertEqual(probe.read_plan(self.write_plan(directory, subset)), subset)

    def test_invalid_plan_cases_are_refused(self):
        base = probe.default_cases()[0]
        for field, value in (("n", 65535), ("n", 1 << 17), ("shift", 16), ("shift", -1),
                             ("shift", 1), ("block", 512), ("warmups", 20), ("launches", 100)):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    probe.read_plan(self.write_plan(directory, [dict(base, **{field: value})]))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "duplicate"):
                probe.read_plan(self.write_plan(directory, [base, base]))

    def records(self):
        c = probe.default_cases()[0]
        return c, [
            dict(type="device", name="MetaX C550", visible_device_count=1,
                 wave_size_api=64, pci_bus_id="0000:01:00.0", runtime_version_api=1, driver_version_api=1),
            dict(type="protocol", schema_version=1, experiment=probe.EXPERIMENT,
                 input_elements=probe.INPUT_ELEMENTS, guard_elements_each_side=32,
                 timer="mcEventElapsedTime", **{key: None for key in probe.ENVIRONMENT}),
            dict(type="case", id=c["id"], n=c["n"], shift=c["shift"], block=256,
                 log2_n=16, grid=256, warmups=10, samples=10, launches_per_sample=10,
                 total_launches=110, unique_input_elements=c["n"], input_span_bytes=c["n"] * 4,
                 logical_bytes_per_launch=c["n"] * 8, output_file=f"{c['id']}.f32"),
            *[dict(type="sample", id=c["id"], sample=index, event_batch_ms=0.1,
                   host_enqueue_batch_us=50.0) for index in range(10)],
            dict(type="complete", cases=1, cpu_correctness_checked=False),
        ]

    def test_complete_records_are_accepted(self):
        case, records = self.records()
        self.assertEqual(len(probe.validate_records(records, [case])[case["id"]]), 10)

    def test_missing_duplicate_and_misordered_samples_are_refused(self):
        for mutation in ("missing", "duplicate", "misordered", "before_case"):
            case, records = self.records()
            if mutation == "missing":
                records.pop(3)
            elif mutation == "duplicate":
                records[3]["sample"] = 1
            elif mutation == "misordered":
                records[3], records[4] = records[4], records[3]
            else:
                records[2], records[3] = records[3], records[2]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                probe.validate_records(records, [case])

    def test_metadata_and_environment_changes_are_refused(self):
        for field, value in (("shift", 2), ("n", 131072), ("input_span_bytes", 1),
                             ("total_launches", 1020), ("shift", False), ("output_file", "other.f32")):
            case, records = self.records()
            records[2][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "metadata mismatch"):
                probe.validate_records(records, [case])
        case, records = self.records()
        records[1].pop("MACA_CACHE_DISABLE")
        with self.assertRaisesRegex(ValueError, "environment field"):
            probe.validate_records(records, [case])

    def test_nonfinite_nonpositive_or_boolean_timing_is_refused(self):
        for field in ("event_batch_ms", "host_enqueue_batch_us"):
            for value in (float("nan"), float("inf"), 0, -1, True):
                case, records = self.records()
                records[3][field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "Invalid"):
                    probe.validate_records(records, [case])

    def test_partial_run_is_refused(self):
        case, records = self.records()
        with self.assertRaisesRegex(ValueError, "complete record"):
            probe.validate_records(records[:-1], [case])

    def test_device_version_metadata_is_required_and_uninterpreted(self):
        for field in ("runtime_version_api", "driver_version_api"):
            for value in (None, -1, True, "1"):
                case, records = self.records()
                if value is None:
                    records[0].pop(field)
                else:
                    records[0][field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "version field"):
                    probe.validate_records(records, [case])


if __name__ == "__main__":
    unittest.main()
