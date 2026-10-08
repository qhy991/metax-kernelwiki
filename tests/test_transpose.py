"""CPU oracle and transpose-contract checks; no MXCC or GPU execution coverage."""
import array
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "transpose", Path(__file__).resolve().parents[1] / "experiments/transpose/experiment.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def words(values):
    result = array.array("I")
    result.frombytes(array.array("f", values).tobytes())
    return result


def guarded(payload):
    guard = array.array("I", [probe.GUARD_WORD] * probe.GUARD_ELEMENTS)
    return guard + payload + guard


class TransposeTest(unittest.TestCase):
    def test_non_square_transpose_direction(self):
        self.assertEqual(list(probe.reference_pairs(2, 3)),
                         [(0, 0), (1, 2), (2, 4), (3, 1), (4, 3), (5, 5)])
        source = words(range(6))
        output = guarded(words([0, 3, 1, 4, 2, 5]))
        result = probe.validate_output(dict(id="nonsquare", rows=2, cols=3), source, output)
        self.assertEqual(result["payload_elements_checked"], 6)
        self.assertEqual(result["mismatches"], 0)
        with self.assertRaisesRegex(ValueError, "mismatches"):
            probe.validate_output(dict(id="wrong_direction", rows=3, cols=2), source, output)

    def test_matrix_oracle_on_all_small_shapes_and_variants(self):
        for rows, cols in probe.SHAPES:
            if rows * cols > 65 * 65:
                continue
            source = words(range(rows * cols))
            matrix = [source[start:start + cols] for start in range(0, len(source), cols)]
            output = guarded(array.array("I", (value for column in zip(*matrix) for value in column)))
            for variant in (*probe.VARIANTS, *probe.SHARED_PITCH_VARIANTS):
                case = dict(id="matrix", rows=rows, cols=cols, variant=variant)
                with self.subTest(rows=rows, cols=cols, variant=variant):
                    result = probe.validate_output(case, source, output)
                    self.assertEqual(result["finite_count"], rows * cols)
                    self.assertEqual(result["guard_elements_checked"], 64)

    def test_wrong_index_unwritten_nonfinite_and_guard_writes_are_refused(self):
        case = dict(id="negative", rows=31, cols=33)
        source = words(range(31 * 33))
        payload = array.array("I", [probe.GUARD_WORD] * len(source))
        for src, dst in probe.reference_pairs(31, 33):
            payload[dst] = source[src]
        for mutation in ("identity", "swapped", "unwritten", "infinity", "prefix", "suffix", "truncated"):
            output = guarded(payload)
            if mutation == "identity":
                output[32:-32] = source
            elif mutation == "swapped":
                output[33], output[34] = output[34], output[33]
            elif mutation == "unwritten":
                output[32 + len(payload) - 1] = probe.GUARD_WORD
            elif mutation == "infinity":
                output[32] = 0x7F800000
            elif mutation == "prefix":
                output[31] = 0
            elif mutation == "suffix":
                output[-32] = 0
            else:
                output.pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                probe.validate_output(case, source, output)

    def test_tile_loads_initialize_every_valid_store_once_across_all_edge_classes(self):
        # Every tile in an admitted shape belongs to one of these local extent pairs.
        edge_classes = set()
        for rows, cols in probe.SHAPES:
            heights = {min(64, rows)} | ({rows % 64} if rows % 64 else set())
            widths = {min(64, cols)} | ({cols % 64} if cols % 64 else set())
            edge_classes.update((height, width) for height in heights for width in widths)
        for height, width in sorted(edge_classes):
            for padding in (0, 1):
                loaded = {}
                barrier_participants = set()
                for ty in range(4):
                    for tx in range(64):
                        for offset in range(0, 64, 4):
                            if tx < width and ty + offset < height:
                                slot = (ty + offset) * (64 + padding) + tx
                                self.assertNotIn(slot, loaded)
                                loaded[slot] = (ty + offset, tx)
                        barrier_participants.add((tx, ty))
                self.assertEqual(len(barrier_participants), 256)
                stores = {}
                # The read phase starts only after the modeled full-block barrier.
                for ty in range(4):
                    for tx in range(64):
                        for offset in range(0, 64, 4):
                            if tx < height and ty + offset < width:
                                slot = tx * (64 + padding) + ty + offset
                                self.assertIn(slot, loaded, (height, width, padding, slot))
                                self.assertEqual(loaded[slot], (tx, ty + offset))
                                output_coordinate = (ty + offset, tx)
                                self.assertNotIn(output_coordinate, stores)
                                stores[output_coordinate] = loaded[slot]
                self.assertEqual(set(stores), {(col, row) for row in range(height) for col in range(width)})

    def test_small_multi_tile_write_coverage_including_tails(self):
        for rows, cols in ((1, 65), (65, 1), (63, 65), (65, 63), (65, 65), (31, 33), (33, 31)):
            outputs = []
            inputs = []
            for by in range((rows + 63) // 64):
                for bx in range((cols + 63) // 64):
                    for local_row in range(64):
                        for local_col in range(64):
                            row, col = by * 64 + local_row, bx * 64 + local_col
                            if row < rows and col < cols:
                                inputs.append(row * cols + col)
                                outputs.append(col * rows + row)
            with self.subTest(rows=rows, cols=cols):
                self.assertEqual(sorted(inputs), list(range(rows * cols)))
                self.assertEqual(sorted(outputs), list(range(rows * cols)))

    def write_plan(self, directory, cases):
        path = Path(directory) / "cases.tsv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=probe.COLUMNS, delimiter="\t")
            writer.writeheader()
            writer.writerows(cases)
        return path

    def test_57_case_plan_roundtrip_and_geometry(self):
        cases = probe.default_cases()
        self.assertEqual(len(probe.SHAPES), 19)
        self.assertEqual(len(cases), 57)
        for shape_index, shape in enumerate(probe.SHAPES):
            group = cases[shape_index * 3:(shape_index + 1) * 3]
            self.assertEqual([c["variant"] for c in group], list(probe.VARIANTS))
            self.assertTrue(all((c["rows"], c["cols"]) == shape for c in group))
        for case in cases:
            meta = probe.launch_metadata(case)
            self.assertLessEqual(meta["n"], 1 << 24)
            self.assertEqual(meta["block_x"] * meta["block_y"] * meta["block_z"], 256)
            self.assertEqual(case["warmups"] + case["samples"] * case["launches"], 110)
            self.assertEqual(meta["intended_static_shared_bytes"],
                             {"direct": 0, "tile64": 16384, "tile64_pad1": 16640}[case["variant"]])
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(probe.read_plan(self.write_plan(directory, cases)), cases)
            subset = [cases[-1], cases[0]]
            self.assertEqual(probe.read_plan(self.write_plan(directory, subset)), subset)

    def test_shared_pitch_suite_has_38_cases_and_equal_source_capacity(self):
        cases = probe.shared_pitch_cases()
        self.assertEqual(len(cases), 38)
        self.assertEqual(probe.VARIANTS, ("direct", "tile64", "tile64_pad1"))
        self.assertEqual(len(probe.default_cases()), 57)
        for shape_index, shape in enumerate(probe.SHAPES):
            group = cases[shape_index * 2:(shape_index + 1) * 2]
            self.assertEqual([case["variant"] for case in group], ["runtime_pitch64", "runtime_pitch65"])
            for case, pitch in zip(group, (64, 65)):
                self.assertEqual((case["rows"], case["cols"]), shape)
                meta = probe.launch_metadata(case)
                self.assertEqual(meta["shared_pitch_elements"], pitch)
                self.assertEqual(meta["padding"], pitch - 64)
                self.assertEqual(meta["allocated_shared_elements"], 4160)
                self.assertEqual(meta["intended_static_shared_bytes"], 16640)
                self.assertEqual((meta["block_x"], meta["block_y"], meta["block_z"]), (64, 4, 1))
                self.assertEqual(case["warmups"] + case["samples"] * case["launches"], 110)
        for case in probe.default_cases():
            meta = probe.launch_metadata(case)
            self.assertNotIn("shared_pitch_elements", meta)
            self.assertNotIn("allocated_shared_elements", meta)
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(probe.read_plan(self.write_plan(directory, cases)), cases)
            subset = [cases[-1], cases[0]]
            self.assertEqual(probe.read_plan(self.write_plan(directory, subset)), subset)

    def test_runtime_pitches_share_4160_slots_and_preserve_unique_initialized_stores(self):
        edge_classes = set()
        for rows, cols in probe.SHAPES:
            heights = {min(64, rows)} | ({rows % 64} if rows % 64 else set())
            widths = {min(64, cols)} | ({cols % 64} if cols % 64 else set())
            edge_classes.update((height, width) for height in heights for width in widths)
        for height, width in sorted(edge_classes):
            for pitch in (64, 65):
                storage = [None] * 4160
                loaded = set()
                for ty in range(4):
                    for tx in range(64):
                        for offset in range(0, 64, 4):
                            if tx < width and ty + offset < height:
                                slot = (ty + offset) * pitch + tx
                                self.assertLess(slot, len(storage))
                                self.assertNotIn(slot, loaded)
                                loaded.add(slot)
                                storage[slot] = (ty + offset) * width + tx
                # All 256 modeled threads complete the load phase before any shared read.
                output = [None] * (height * width)
                consumed = set()
                for ty in range(4):
                    for tx in range(64):
                        for offset in range(0, 64, 4):
                            if tx < height and ty + offset < width:
                                slot = tx * pitch + ty + offset
                                self.assertLess(slot, len(storage))
                                self.assertIn(slot, loaded)
                                self.assertNotIn(slot, consumed)
                                consumed.add(slot)
                                output_index = (ty + offset) * height + tx
                                self.assertIsNone(output[output_index])
                                output[output_index] = storage[slot]
                with self.subTest(height=height, width=width, pitch=pitch):
                    self.assertEqual(loaded, consumed)
                    self.assertEqual(output, [row * width + col for col in range(width) for row in range(height)])

    def test_invalid_shapes_variants_and_timing_are_refused(self):
        base = probe.default_cases()[0]
        for field, value in (("rows", 0), ("cols", -1), ("rows", 1 << 25), ("cols", 66),
                             ("variant", "tile32"), ("variant", "runtime_pitch63"),
                             ("variant", "runtime_pitch66"), ("warmups", 20), ("launches", 100)):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    probe.read_plan(self.write_plan(directory, [dict(base, **{field: value})]))

    def records(self, variant="tile64_pad1"):
        case = dict(id="transpose_test", rows=65, cols=63, variant=variant, warmups=10, samples=10, launches=10)
        metadata = probe.launch_metadata(case)
        return case, [
            dict(type="device", name="MetaX C550", visible_device_count=1, wave_size_api=64,
                 pci_bus_id="0000:01:00.0", runtime_version_api=1, driver_version_api=1),
            dict(type="protocol", schema_version=1, experiment=probe.EXPERIMENT,
                 input_elements=probe.INPUT_ELEMENTS, guard_elements_each_side=32,
                 timer="mcEventElapsedTime", **{key: None for key in probe.ENVIRONMENT}),
            dict(type="case", **case, **metadata, launches_per_sample=10, total_launches=110,
                 unique_input_elements=4095, input_span_bytes=16380, logical_bytes_per_launch=32760,
                 output_file="transpose_test.f32", function_attributes_before_timing=dict(
                     maxThreadsPerBlock=512, numRegs=16, sharedSizeBytes=17000, localSizeBytes=0)),
            *[dict(type="sample", id=case["id"], sample=i, event_batch_ms=0.1,
                   host_enqueue_batch_us=50.0) for i in range(10)],
            dict(type="complete", cases=1, cpu_correctness_checked=False),
        ]

    def test_actual_shared_size_is_observed_not_forced_to_intended_size(self):
        for variant in (*probe.VARIANTS, *probe.SHARED_PITCH_VARIANTS):
            case, records = self.records(variant)
            self.assertEqual(len(probe.validate_records(records, [case])[case["id"]]), 10)

    def test_shared_pitch_metadata_is_complete_exact_and_separate_from_actual_allocation(self):
        for variant in probe.SHARED_PITCH_VARIANTS:
            for field in ("shared_pitch_elements", "allocated_shared_elements", "intended_static_shared_bytes"):
                for value in (None, True, 0, -1, 63, 66, 4096, 16384, "64", 64.0):
                    case, records = self.records(variant)
                    if value is None:
                        records[2].pop(field)
                    else:
                        records[2][field] = value
                    with self.subTest(variant=variant, field=field, value=value), self.assertRaisesRegex(ValueError, "metadata mismatch"):
                        probe.validate_records(records, [case])
            case, records = self.records(variant)
            records[2]["shared_pitch_elements"] = 65 if variant == "runtime_pitch64" else 64
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                probe.validate_records(records, [case])
        first_case, first_records = self.records("runtime_pitch64")
        second_case, second_records = self.records("runtime_pitch65")
        first_records[2]["function_attributes_before_timing"]["sharedSizeBytes"] = 16384
        second_records[2]["function_attributes_before_timing"]["sharedSizeBytes"] = 17000
        # Runtime attributes remain independent observations even for one source allocation.
        probe.validate_records(first_records, [first_case])
        probe.validate_records(second_records, [second_case])

    def test_missing_or_invalid_geometry_and_function_metadata_are_refused(self):
        for field in ("rows", "cols", "variant", "grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z",
                      "tile_rows", "tile_cols", "padding", "intended_static_shared_bytes", "total_launches"):
            for mutation in ("missing", "wrong"):
                case, records = self.records()
                if mutation == "missing":
                    records[2].pop(field)
                else:
                    records[2][field] = "incorrect" if field == "variant" else -1
                with self.subTest(field=field, mutation=mutation), self.assertRaisesRegex(ValueError, "metadata mismatch"):
                    probe.validate_records(records, [case])
        for field in ("maxThreadsPerBlock", "numRegs", "sharedSizeBytes", "localSizeBytes"):
            for value in (None, -1, True, 1.5):
                case, records = self.records()
                attributes = records[2]["function_attributes_before_timing"]
                if value is None:
                    attributes.pop(field)
                else:
                    attributes[field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "function attributes"):
                    probe.validate_records(records, [case])

    def test_missing_duplicate_misordered_and_invalid_samples_are_refused(self):
        for mutation in ("missing", "duplicate", "misordered", "before_case", "nan", "boolean"):
            case, records = self.records()
            if mutation == "missing":
                records.pop(3)
            elif mutation == "duplicate":
                records[3]["sample"] = 1
            elif mutation == "misordered":
                records[3], records[4] = records[4], records[3]
            elif mutation == "before_case":
                records[2], records[3] = records[3], records[2]
            elif mutation == "nan":
                records[3]["event_batch_ms"] = float("nan")
            else:
                records[3]["host_enqueue_batch_us"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                probe.validate_records(records, [case])

    def test_incomplete_run_wrong_protocol_or_missing_runtime_metadata_is_refused(self):
        for mutation in ("no_complete", "protocol", "version", "environment"):
            case, records = self.records()
            if mutation == "no_complete":
                records.pop()
            elif mutation == "protocol":
                records[1]["experiment"] = "fixed-footprint-memory-order"
            elif mutation == "version":
                records[0].pop("runtime_version_api")
            else:
                records[1].pop("MACA_CACHE_PATH")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                probe.validate_records(records, [case])


if __name__ == "__main__":
    unittest.main()
