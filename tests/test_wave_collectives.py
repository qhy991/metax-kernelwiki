"""Host tests for complete physical-wave outputs; no GPU intrinsic execution."""
import array
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "wave_collectives", Path(__file__).resolve().parents[1] / "experiments/wave_collectives/experiment.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def case(block, n):
    return dict(id=f"wave_b{block}_n{n}", block=block, n=n, warmups=10, samples=10, launches=10)


def guarded(values):
    guard = array.array("I", [probe.GUARD_WORD] * probe.GUARD_ELEMENTS)
    return guard + array.array("I", values) + guard


class WaveCollectivesTest(unittest.TestCase):
    def setUp(self):
        self.source = array.array("I", range(1, 129))

    def test_full_wave_and_upper_half_shuffle_direction_and_source_wrapping(self):
        expected = probe.reference_output(case(128, 128), self.source)
        self.assertEqual(expected[0:9], [1, 32, 33, 64, 1, 32, 1, 32, 2080])
        self.assertEqual(expected[32*9:33*9], [1, 32, 33, 64, 33, 64, 33, 64, 2080])
        self.assertEqual(expected[64*9:65*9], [65, 96, 97, 128, 65, 96, 65, 96, 6176])
        self.assertEqual(expected[96*9:97*9], [65, 96, 97, 128, 97, 128, 97, 128, 6176])
        self.assertEqual({expected[thread*9+8] for thread in range(128)}, {2080, 6176})

    def test_zero_logical_length_still_has_all_physical_thread_outputs(self):
        for block in (64, 128):
            c = case(block, 0)
            expected = probe.reference_output(c, self.source)
            self.assertEqual(expected, [0] * (block * 9))
            checked = probe.validate_output(c, self.source, guarded(expected))
            self.assertEqual(checked["physical_threads_checked"], block)
            self.assertEqual(checked["payload_elements_checked"], block * 9)
            self.assertEqual(checked["zero_outputs_checked"], block * 9)

    def test_padded_thread_can_receive_nonzero_collective_results(self):
        expected = probe.reference_output(case(128, 65), self.source)
        self.assertEqual(expected[65*9:66*9], [65, 0, 0, 0, 65, 0, 65, 0, 65])
        self.assertEqual(expected[127*9:128*9], [65, 0, 0, 0, 0, 0, 0, 0, 65])

    def test_every_admitted_case_has_complete_exact_output_coverage(self):
        cases = probe.default_cases()
        self.assertEqual(len(cases), 20)
        self.assertEqual([c["n"] for c in cases if c["block"] == 64], [0,1,31,32,33,63,64])
        self.assertEqual([c["n"] for c in cases if c["block"] == 128], [0,1,31,32,33,63,64,65,95,96,97,127,128])
        for c in cases:
            with self.subTest(block=c["block"], n=c["n"]):
                checked = probe.validate_output(c, self.source, guarded(probe.reference_output(c, self.source)))
                self.assertEqual(checked["payload_elements_checked"], c["block"] * 9)
                self.assertEqual(checked["mismatches"], 0)
                self.assertEqual(probe.case_metadata(c)["total_launches"], 110)

    def test_wrong_32_lane_and_whole_block_reductions_are_refused(self):
        c = case(128, 128)
        for mutation in ("four_groups", "whole_block"):
            output = guarded(probe.reference_output(c, self.source))
            for thread in range(128):
                total = sum(self.source[(thread // 32)*32:(thread // 32+1)*32]) if mutation == "four_groups" else sum(self.source)
                output[32 + thread * 9 + 8] = total
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "channel=8"):
                probe.validate_output(c, self.source, output)

    def test_upper_32_broadcast_and_second_wave_sources_are_not_first_subgroup(self):
        c = case(128, 128)
        for thread, channel, wrong in ((32, 4, 1), (32, 6, 1), (96, 5, 96), (64, 0, 1), (64, 2, 33)):
            output = guarded(probe.reference_output(c, self.source))
            output[32 + thread * 9 + channel] = wrong
            with self.subTest(thread=thread, channel=channel), self.assertRaisesRegex(ValueError, "mismatches"):
                probe.validate_output(c, self.source, output)

    def test_nonzero_padding_and_unwritten_physical_outputs_are_refused(self):
        for c, thread, channel, wrong in ((case(64, 0), 63, 8, 1), (case(64, 32), 40, 2, 33),
                                          (case(128, 65), 127, 8, probe.GUARD_WORD)):
            output = guarded(probe.reference_output(c, self.source))
            output[32 + thread * 9 + channel] = wrong
            with self.subTest(block=c["block"], n=c["n"]), self.assertRaisesRegex(ValueError, "mismatches"):
                probe.validate_output(c, self.source, output)
        with self.assertRaisesRegex(ValueError, "extent"):
            probe.validate_output(case(128, 65), self.source, guarded([0] * (65 * 9)))

    def test_all_guards_input_and_output_extent_are_checked(self):
        c = case(64, 1)
        for index in (0, 31, 32+64*9, 32+64*9+31):
            output = guarded(probe.reference_output(c, self.source))
            output[index] = 0
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "guard overwritten"):
                probe.validate_output(c, self.source, output)
        self.source[127] = 127
        with self.assertRaisesRegex(ValueError, "128 exact"):
            probe.validate_input(self.source)

    def write_plan(self, directory, cases):
        path = Path(directory) / "cases.tsv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=probe.COLUMNS, delimiter="\t")
            writer.writeheader()
            writer.writerows(cases)
        return path

    def test_plan_roundtrip_reordered_subset_and_invalid_boundaries(self):
        cases = probe.default_cases()
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(probe.read_plan(self.write_plan(directory, cases)), cases)
            self.assertEqual(probe.read_plan(self.write_plan(directory, cases[-2:][::-1])), cases[-2:][::-1])
        for block,n in ((32,31),(96,64),(256,128),(64,-1),(64,65),(128,129),(64,2),(128,94)):
            with self.subTest(block=block,n=n), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    probe.read_plan(self.write_plan(directory, [case(block,n)]))

    def records(self, c):
        return [
            dict(type="device", name="MetaX C550", visible_device_count=1, wave_size_api=64,
                 pci_bus_id="0000:01:00.0", runtime_version_api=1, driver_version_api=1),
            dict(type="protocol", schema_version=1, experiment=probe.EXPERIMENT, input_elements=128,
                 dtype="int32", guard_elements_each_side=32, timer="mcEventElapsedTime", required_wave_size=64,
                 mask_hex=probe.MASK_HEX, mask_bits=64, channels=list(probe.CHANNELS), participation="all physical threads",
                 **{field: None for field in probe.ENVIRONMENT}),
            dict(type="case", id=c["id"], **probe.case_metadata(c), function_attributes_before_timing=dict(
                 maxThreadsPerBlock=512,numRegs=16,sharedSizeBytes=0,localSizeBytes=0)),
            *[dict(type="sample", id=c["id"], sample=i, event_batch_ms=0.1,host_enqueue_batch_us=50.0) for i in range(10)],
            dict(type="complete",cases=1,cpu_correctness_checked=False),
        ]

    def test_correct_records_accepted_and_wave32_device_refused(self):
        c = case(128, 97)
        records = self.records(c)
        self.assertEqual(len(probe.validate_records(records, [c])[c["id"]]),10)
        records[0]["wave_size_api"] = 32
        with self.assertRaisesRegex(ValueError, "observed wave size 64"):
            probe.validate_records(records, [c])

    def test_wrong_mask_participation_and_boundary_metadata_are_refused(self):
        c = case(128, 97)
        for field,wrong in (("mask_hex","0xffffffff"),("mask_bits",32),("n",96),("physical_threads",97),
                            ("logical_threads",128),("zero_padded_threads",0),("wave_count",4),
                            ("wave_size",32),("participation","logical threads only"),("output_elements",97*9),
                            ("channels_per_thread",8),("output_layout","channel-major")):
            for missing in (False,True):
                records = self.records(c)
                if missing:
                    records[2].pop(field)
                else:
                    records[2][field] = wrong
                with self.subTest(field=field,missing=missing),self.assertRaisesRegex(ValueError,"Case metadata mismatch"):
                    probe.validate_records(records,[c])
        for field,wrong in (("mask_hex","0xffffffff"),("mask_bits",32),("required_wave_size",32),
                            ("participation","logical threads only"),("channels",list(reversed(probe.CHANNELS)))):
            records = self.records(c)
            records[1][field] = wrong
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"Protocol metadata mismatch"):
                probe.validate_records(records,[c])

    def test_incomplete_samples_and_invalid_resources_are_refused(self):
        c = case(64, 33)
        for mutation in ("missing_sample","duplicate_sample","nonfinite","no_complete","no_resource","boolean_resource"):
            records = self.records(c)
            if mutation == "missing_sample": records.pop(3)
            elif mutation == "duplicate_sample": records[3]["sample"] = 1
            elif mutation == "nonfinite": records[3]["event_batch_ms"] = float("nan")
            elif mutation == "no_complete": records.pop()
            elif mutation == "no_resource": records[2]["function_attributes_before_timing"].pop("numRegs")
            else: records[2]["function_attributes_before_timing"]["numRegs"] = True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                probe.validate_records(records,[c])

    def test_prepare_and_complete_cpu_file_check(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs,outputs = Path(directory)/"input",Path(directory)/"output"
            prepared = probe.prepare(inputs)
            self.assertEqual(prepared["cases"],20)
            self.assertEqual(prepared["input_bytes"],512)
            c = case(128,0)
            self.write_plan(inputs,[c])
            outputs.mkdir()
            with (outputs/(c["id"]+".i32")).open("wb") as handle:
                guarded([0]*1152).tofile(handle)
            (outputs/"raw.jsonl").write_text("".join(json.dumps(record)+"\n" for record in self.records(c)))
            checked = probe.check(inputs,outputs)
            self.assertEqual(checked["status"],"pass")
            self.assertEqual(checked["cases"][0]["physical_threads_checked"],128)
            self.assertEqual(checked["cases"][0]["zero_outputs_checked"],1152)


if __name__ == "__main__":
    unittest.main()
