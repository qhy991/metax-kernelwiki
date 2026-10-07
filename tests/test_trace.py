"""CPU-only checks for accepting actual kernel traces and public projections."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_trace.py"
SPEC = importlib.util.spec_from_file_location("summarize_trace", SCRIPT)
trace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace)


def kernel(timestamp, duration):
    return {"ph": "X", "pid": 2, "cat": "0", "ts": timestamp, "dur": duration,
            "name": "/private/source/path/kernel",
            "args": {"process_id": 9999, "block": {"x": 256, "y": 1, "z": 1},
                     "grid": {"x": 16385, "y": 1, "z": 1},
                     "mem": {"registers_per_thread": 6, "static_shared": 0}}}


class TraceAcceptanceTest(unittest.TestCase):
    def test_public_summary_retains_raw_units_and_metadata_coverage(self):
        earlier = kernel(1791359468814913280, 49000)
        earlier["args"]["max_block_size"] = 512
        later = kernel(1791359468815013280, 48000)
        result = trace.summarize_trace({"traceEvents": [later, earlier]}, 2, 1)
        self.assertEqual(result["raw_durations"], [49000, 48000])
        self.assertEqual(result["raw_durations_after_warmups"], [48000])
        self.assertEqual(result["time_unit"], "units_unverified")
        self.assertFalse(result["duration_conversion_applied"])
        self.assertEqual(result["resources"]["max_block_size"], {
            "reported_count": 1, "missing_count": 1,
            "values": [{"value": 512, "count": 1}]})
        encoded = json.dumps(result)
        for private in ("1791359468814913280", "9999", "/private/source", '"pid"', '"ts"'):
            self.assertNotIn(private, encoded)

    def test_runtime_api_events_cannot_pass_as_gpu_kernels(self):
        event = kernel(10, 5)
        event.update(pid=1, name="mcLaunchKernel")
        with self.assertRaisesRegex(ValueError, "No supported GPU kernel events"):
            trace.summarize_trace({"traceEvents": [event]}, 1, 0)

    def test_process_success_without_gpu_trace_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No supported GPU kernel events"):
            trace.summarize_trace({"returncode": 0, "traceEvents": []}, 1, 0)

    def test_expected_kernel_count_must_match(self):
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            trace.summarize_trace({"traceEvents": [kernel(10, 5)]}, 2, 0)

    def test_negative_nonfinite_and_boolean_durations_are_rejected(self):
        for duration in (-1, float("nan"), float("inf"), True):
            with self.subTest(duration=duration):
                with self.assertRaisesRegex(ValueError, "kernel duration"):
                    trace.summarize_trace({"traceEvents": [kernel(10, duration)]}, 1, 0)

    def test_malformed_trace_structure_is_rejected(self):
        for document in ([], {}, {"traceEvents": {}}, {"traceEvents": [None]}):
            with self.subTest(document=document):
                with self.assertRaisesRegex(ValueError, "Malformed trace"):
                    trace.summarize_trace(document, 1, 0)

    def test_cli_missing_or_malformed_input_is_nonzero_without_public_path(self):
        with tempfile.TemporaryDirectory(prefix="trace-input-test-") as directory:
            path = Path(directory) / "missing.json"
            for content in (None, "not JSON"):
                with self.subTest(content=content):
                    if content is not None:
                        path.write_text(content)
                    result = subprocess.run(
                        [sys.executable, str(SCRIPT), str(path), "--expected-launches", "1"],
                        text=True, capture_output=True, timeout=15)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(json.loads(result.stderr)["status"], "error")
                    self.assertNotIn(str(path), result.stderr)


if __name__ == "__main__":
    unittest.main()
