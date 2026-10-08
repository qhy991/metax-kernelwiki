"""Small CPU receipt fixtures, not SDK execution or device evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "experiments/mcrtc_format/check.py"
SPEC = importlib.util.spec_from_file_location("mcrtc_receipt_check", MODULE)
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def api(name, code=0):
    return dict(type="api", api=name, status_code=code,
                status_name="MCRTC_ERROR_COMPILATION" if code == 6 else "MCRTC_SUCCESS",
                error_string="synthetic compiler failure" if code == 6 else "synthetic success")


class McrtcFormatTest(unittest.TestCase):
    def fixture(self, directory, case="valid", output=b"\x00opaque\xff\x00bytes\x00", log=None):
        if log is None:
            log = b"synthetic warning\n\x00" if case == "valid" else b"error: MCRTC_FORMAT_NEGATIVE_CONTROL\n\x00"
        rows = [check.request_metadata(case), api("mcrtcVersion"), dict(type="version", major=7, minor=9),
                api("mcrtcCreateProgram"), api("mcrtcCompileProgram", 0 if case == "valid" else 6),
                api("mcrtcGetProgramLogSize"), dict(type="size", buffer="compile-log", bytes=len(log)),
                api("mcrtcGetProgramLog"), dict(type="file", buffer="compile-log", file="compile.log", bytes=len(log))]
        (directory / "compile.log").write_bytes(log)
        if case == "valid":
            rows += [api("mcrtcGetBitcodeSize"), dict(type="size", buffer="producer-output", bytes=len(output)),
                     api("mcrtcGetBitcode"), dict(type="file", buffer="producer-output", file="producer-output.bin", bytes=len(output))]
            (directory / "producer-output.bin").write_bytes(output)
        rows += [api("mcrtcDestroyProgram"), dict(type="complete", case=case,
                    status="producer-success" if case == "valid" else "compile-failed",
                    exit_code=0 if case == "valid" else 1, program_created=True, program_destroyed=True,
                    log_retained=True, output_retained=case == "valid")]
        self.save(directory, rows)
        return rows

    def save(self, directory, rows):
        (directory / "raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def cli(self, directory, case):
        return subprocess.run(["python3", str(MODULE), str(directory), "--case", case], capture_output=True, text=True)

    def assert_rejected(self, directory, case="valid"):
        result = self.cli(directory, case)
        self.assertEqual((result.returncode, result.stdout), (2, ""))
        self.assertEqual(json.loads(result.stderr)["error_kind"], "receipt")

    def test_success_preserves_opaque_bytes_and_observed_version(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            opaque = b"\x00not a prescribed format\xff\x00\x00"
            self.fixture(directory, output=opaque)
            result = check.check(directory, "valid")
            self.assertEqual(result["status"], "producer-success")
            self.assertEqual(result["recorded_producer_exit"], 0)
            self.assertEqual(result["observed_mcrtc_version"], {"major": 7, "minor": 9})
            self.assertEqual(result["output_bytes_checked"], len(opaque))
            self.assertEqual((directory / "producer-output.bin").read_bytes(), opaque)
            self.assertFalse(result["output_format_interpreted"])
            self.assertEqual((result["loader_acceptance"], result["numerical_acceptance"]), ("not_tested", "not_applicable"))
            self.assertEqual(self.cli(directory, "valid").returncode, 0)

    def test_reported_zero_extents_remain_zero_not_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.fixture(directory, output=b"", log=b"")
            result = check.check(directory, "valid")
            self.assertEqual((result["log_bytes_checked"], result["output_bytes_checked"]), (0, 0))
            self.assertTrue(result["output_retained"])
            (directory / "producer-output.bin").unlink()
            self.assert_rejected(directory)

    def test_declared_negative_is_valid_control_but_stays_compile_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.fixture(directory, "compile-error")
            result = check.check(directory, "compile-error")
            self.assertEqual((result["status"], result["recorded_producer_exit"]), ("compile-failed", 1))
            self.assertTrue(result["receipt_valid"] and result["expected_compile_failure"] and result["program_destroyed"])
            self.assertFalse(result["output_retained"])
            self.assertIsNone(result["output_bytes_checked"])
            self.assertNotIn("passed", result)
            self.assertEqual(self.cli(directory, "compile-error").returncode, 0)
            (directory / "producer-output.bin").write_bytes(b"stale")
            self.assert_rejected(directory, "compile-error")

    def test_case_binding_and_negative_marker_prevent_unrelated_failure_acceptance(self):
        for case in ("valid", "compile-error"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                self.fixture(directory, case)
                self.assert_rejected(directory, "compile-error" if case == "valid" else "valid")
        for log in (b"unrelated compiler failure\0", b""):
            with self.subTest(log=log), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                self.fixture(directory, "compile-error", log=log)
                self.assert_rejected(directory, "compile-error")

    def test_nonzero_api_wrong_status_name_and_null_error_string_refuse(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            records = self.fixture(directory)
            for index, row in enumerate(records):
                if row["type"] != "api":
                    continue
                for mutation in ("failure", "bool", "float", "wrong-name", "null-string"):
                    changed = copy.deepcopy(records)
                    if mutation == "failure": changed[index].update(status_code=11, status_name="MCRTC_ERROR_INTERNAL_ERROR")
                    elif mutation == "bool": changed[index]["status_code"] = False
                    elif mutation == "float": changed[index]["status_code"] = 0.0
                    elif mutation == "wrong-name": changed[index]["status_name"] = "MCRTC_ERROR_COMPILATION"
                    else: changed[index]["error_string"] = None
                    with self.subTest(api=row["api"], mutation=mutation):
                        self.save(directory, changed)
                        with self.assertRaises(check.ReceiptError): check.check(directory, "valid")
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            records = self.fixture(directory, "compile-error")
            for code, name in ((0, "MCRTC_SUCCESS"), (5, "MCRTC_ERROR_INVALID_OPTION"), (6, "MCRTC_SUCCESS")):
                changed = copy.deepcopy(records)
                changed[4].update(status_code=code, status_name=name)
                self.save(directory, changed)
                with self.subTest(code=code, name=name), self.assertRaises(check.ReceiptError): check.check(directory, "compile-error")

    def test_cleanup_and_retention_completion_flags_are_required(self):
        for case in ("valid", "compile-error"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                records = self.fixture(directory, case)
                for key in ("program_created", "program_destroyed", "log_retained", "output_retained"):
                    for value in (not records[-1][key], int(records[-1][key])):
                        changed = copy.deepcopy(records)
                        changed[-1][key] = value
                        self.save(directory, changed)
                        with self.subTest(field=key, value=value), self.assertRaises(check.ReceiptError): check.check(directory, case)
                changed = copy.deepcopy(records)
                changed[-1].update(status="error", exit_code=2)
                self.save(directory, changed)
                self.assert_rejected(directory, case)

    def test_missing_truncated_oversized_files_and_log_terminator_refuse(self):
        for filename in ("compile.log", "producer-output.bin"):
            for mutation in ("missing", "truncated", "extended", "log-nul"):
                if mutation == "log-nul" and filename != "compile.log": continue
                with self.subTest(filename=filename, mutation=mutation), tempfile.TemporaryDirectory() as temp:
                    directory = Path(temp)
                    self.fixture(directory)
                    path = directory / filename
                    data = path.read_bytes()
                    if mutation == "missing": path.unlink()
                    elif mutation == "truncated": path.write_bytes(data[:-1])
                    elif mutation == "extended": path.write_bytes(data + b"\0")
                    else: path.write_bytes(data[:-1] + b"X")
                    self.assert_rejected(directory)

    def test_size_records_file_roles_and_buffer_cap_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            records = self.fixture(directory)
            for index in (6, 10):
                for value in (-1, True, 1.0, 67108865, records[index]["bytes"] + 1):
                    changed = copy.deepcopy(records)
                    changed[index]["bytes"] = value
                    self.save(directory, changed)
                    with self.subTest(index=index, value=value), self.assertRaises(check.ReceiptError): check.check(directory, "valid")
            for index, key, value in ((8, "file", "../compile.log"), (12, "file", "compile.log"),
                                      (6, "buffer", "producer-output"), (8, "bytes", 0.0), (12, "bytes", False)):
                changed = copy.deepcopy(records)
                changed[index][key] = value
                self.save(directory, changed)
                with self.subTest(index=index, field=key), self.assertRaises(check.ReceiptError): check.check(directory, "valid")

    def test_every_visibility_mask_must_be_present_and_empty(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            records = self.fixture(directory)
            for key in check.MASKS:
                for mutation in ("absent", "nonempty", "null", "bool"):
                    changed = copy.deepcopy(records)
                    if mutation == "absent": del changed[0]["visibility"][key]
                    else: changed[0]["visibility"][key] = {"nonempty": "0", "null": None, "bool": False}[mutation]
                    self.save(directory, changed)
                    with self.subTest(key=key, mutation=mutation), self.assertRaises(check.ReceiptError): check.check(directory, "valid")

    def test_source_options_pointer_flags_and_version_types_are_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            records = self.fixture(directory)
            for key, value in (("source", "extern C void other(){}"), ("program_name", "other.mc"),
                    ("schema", "unknown"), ("num_options", False), ("options", ["-x maca"]),
                    ("options_pointer_is_null", 1), ("num_headers", 0.0), ("headers_pointer_is_null", False),
                    ("include_names_pointer_is_null", False), ("maximum_buffer_bytes", 67108864.0)):
                changed = copy.deepcopy(records)
                changed[0][key] = value
                self.save(directory, changed)
                with self.subTest(field=key), self.assertRaises(check.ReceiptError): check.check(directory, "valid")
            for key in ("major", "minor"):
                for value in (-1, True, 1.0, 1 << 31):
                    changed = copy.deepcopy(records)
                    changed[2][key] = value
                    self.save(directory, changed)
                    with self.subTest(field=key, value=value), self.assertRaises(check.ReceiptError): check.check(directory, "valid")

    def test_truncated_reordered_error_and_malformed_records_refuse(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            records = self.fixture(directory)
            cases = [records[:-1], records + [records[-1]], records[:-2] + [records[-1]],
                     records[:9] + [dict(type="error", message="retention failed")] + records[9:]]
            swapped = copy.deepcopy(records)
            swapped[3], swapped[4] = swapped[4], swapped[3]
            cases.append(swapped)
            for changed in cases:
                self.save(directory, changed)
                self.assert_rejected(directory)
            for text in ('{"type":"request","type":"request"}\n', 'NaN\n', '{"unfinished":', '\n'):
                (directory / "raw.jsonl").write_text(text)
                self.assert_rejected(directory)

    def test_cli_requires_explicit_case(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.fixture(directory)
            for args in ([str(directory)], [str(directory), "--case", "other"]):
                result = subprocess.run(["python3", str(MODULE), *args], capture_output=True, text=True)
                self.assertEqual((result.returncode, result.stdout), (2, ""))
                self.assertIn("--case", result.stderr)


if __name__ == "__main__":
    unittest.main()
