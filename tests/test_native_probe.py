"""CPU oracle tests only: these do not establish MACA compilation or GPU correctness."""
import array
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "native_probe", Path(__file__).resolve().parents[1] / "experiments/native/native_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def words(values):
    result = array.array("I")
    result.frombytes(array.array("f", values).tobytes())
    return result


class NativeOracleTest(unittest.TestCase):
    def setUp(self):
        self.source = words(range(160))
        self.case = dict(id="gather_test", kind="gather", n=10, stride=16, block=64,
                         warmups=20, samples=10, launches=100)
        self.output = (array.array("I", [probe.GUARD_WORD] * probe.GUARD_ELEMENTS)
                       + array.array("I", (self.source[i * 16] for i in range(10)))
                       + array.array("I", [probe.GUARD_WORD] * probe.GUARD_ELEMENTS))

    def test_exact_unique_gather_and_finite_count(self):
        probe.validate_input(self.source)
        result = probe.validate_output(self.case, self.source, self.output)
        self.assertEqual(result["finite_count"], 10)
        self.assertEqual(result["unique_input_indices"], 10)

    def test_tail_wrong_index_is_refused(self):
        self.output[probe.GUARD_ELEMENTS + 9] = self.source[9]
        with self.assertRaisesRegex(ValueError, "first_mismatch=9"):
            probe.validate_output(self.case, self.source, self.output)

    def test_unwritten_nan_payload_is_refused(self):
        self.output[probe.GUARD_ELEMENTS] = probe.GUARD_WORD
        with self.assertRaisesRegex(ValueError, "finite_count=9/10"):
            probe.validate_output(self.case, self.source, self.output)

    def test_each_guard_is_checked(self):
        for index in (0, len(self.output) - 1):
            output = self.output[:]
            output[index] = 0
            with self.assertRaisesRegex(ValueError, "guard overwritten"):
                probe.validate_output(self.case, self.source, output)

    def test_input_corruption_is_refused(self):
        self.source[159] = self.source[158]
        with self.assertRaisesRegex(ValueError, "index 159"):
            probe.validate_input(self.source)

    def test_default_plan_is_bounded_and_includes_tails(self):
        cases = probe.default_cases()
        self.assertEqual(len(cases), 49)
        self.assertEqual(len({c["id"] for c in cases}), 49)
        for c in cases:
            if c["n"]:
                self.assertLess((c["n"] - 1) * c["stride"], probe.INPUT_ELEMENTS)
                self.assertLessEqual((c["n"] + 64) * 4, 64 << 20)
        self.assertEqual({c["block"] for c in cases if c["kind"] == "copy"}, {64, 128, 256, 512})
        self.assertTrue(any(c["n"] % c["block"] for c in cases if c["kind"] == "copy"))

    def records(self):
        c = self.case
        return [
            dict(type="device", name="MetaX C550", visible_device_count=1, wave_size_api=64),
            dict(type="protocol", schema_version=1, input_elements=160,
                 guard_elements_each_side=32, timer="mcEventElapsedTime"),
            dict(type="case", id=c["id"], kind=c["kind"], n=c["n"], stride=c["stride"],
                 block=c["block"], warmups=20, samples=10, launches_per_sample=100,
                 logical_bytes_per_launch=80, unique_input_elements=10,
                 input_span_bytes=580, grid=1, output_file="gather_test.f32"),
            *[dict(type="sample", id=c["id"], sample=i, event_batch_ms=0.1,
                   host_enqueue_batch_us=50.0) for i in range(10)],
            dict(type="complete", cases=1, cpu_correctness_checked=False),
        ]

    def test_complete_records_accepted(self):
        self.assertEqual(len(probe.validate_records(self.records(), [self.case], 160)[self.case["id"]]), 10)

    def test_missing_and_duplicate_samples_refused(self):
        for mutation in ("missing", "duplicate"):
            records = self.records()
            if mutation == "missing":
                records.pop(3)
            else:
                records[3]["sample"] = 1
            with self.assertRaisesRegex(ValueError, "Missing or duplicated samples"):
                probe.validate_records(records, [self.case], 160)

    def test_nonfinite_timing_refused(self):
        records = self.records()
        records[3]["event_batch_ms"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid event_batch_ms"):
            probe.validate_records(records, [self.case], 160)

    def test_partial_run_refused(self):
        with self.assertRaisesRegex(ValueError, "complete record"):
            probe.validate_records(self.records()[:-1], [self.case], 160)

    def test_metadata_change_refused(self):
        records = self.records()
        records[2]["stride"] = 8
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            probe.validate_records(records, [self.case], 160)


if __name__ == "__main__":
    unittest.main()
