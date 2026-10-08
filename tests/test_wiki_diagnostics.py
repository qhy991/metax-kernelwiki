"""Evidence-scope admission tests; fixtures describe policy, not device observations."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("wiki_diagnostics", Path(__file__).resolve().parents[1] / "scripts/wiki.py")
wiki = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wiki)


class WikiDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        (self.root / "page.md").write_text("Fixture evidence page.\n")
        self.row = dict(id="diagnostic-fixture", title="Fixture", tags=["test"], confidence="locally-measured",
                        evidence_scope="device-correctness", path="page.md", sources=["fixture"],
                        limitations=["CPU schema test only"], result="result.json")
        self.data = dict(run_id="fixture", source_commit="fixture-commit", device={"name": "fixture"},
                         environment={"kind": "fixture"}, evidence_scope="device-correctness",
                         correctness={"passed": False, "tested_contract": "All finite outputs equal the fixed oracle."},
                         measurement={"purpose": "correctness_diagnostic", "performance_accepted": False},
                         limitations=["No device evidence in this fixture"])

    def validate(self, data=None, row=None, *, write_result=True):
        if write_result:
            (self.root / "result.json").write_text(json.dumps(self.data if data is None else data))
        catalog = dict(schema="metax-kernelwiki.catalog.v1", entries=[self.row if row is None else row])
        with patch.object(wiki, "ROOT", self.root):
            return wiki.validate(catalog)

    def test_explicit_diagnostic_admits_boolean_failure_and_success(self):
        for passed in (False, True):
            data = copy.deepcopy(self.data)
            data["correctness"]["passed"] = passed
            with self.subTest(passed=passed):
                self.assertEqual(self.validate(data), [])

    def test_performance_scope_still_requires_pass_even_with_diagnostic_markers(self):
        row = dict(self.row, evidence_scope="local-measurement")
        data = copy.deepcopy(self.data)
        data["evidence_scope"] = "local-measurement"
        self.assertTrue(any("passing full-output" in error for error in self.validate(data, row)))
        data["correctness"]["passed"] = True
        self.assertEqual(self.validate(data, row), [])
        for not_bool_true in (1, "true", False, None):
            data["correctness"]["passed"] = not_bool_true
            with self.subTest(value=not_bool_true):
                self.assertTrue(any("passing full-output" in error for error in self.validate(data, row)))

    def test_old_performance_result_needs_no_diagnostic_markers(self):
        row = dict(self.row, evidence_scope="local-measurement")
        data = copy.deepcopy(self.data)
        data["correctness"] = {"passed": True}
        data["measurement"] = {"timer": "fixture timer"}
        data["evidence_scope"] = "local-measurement"
        self.assertEqual(self.validate(data, row), [])

    def test_diagnostic_scope_must_match_in_the_result(self):
        for scope in (None, "local-measurement", "protocol", ""):
            data = copy.deepcopy(self.data)
            if scope is None:
                data.pop("evidence_scope")
            else:
                data["evidence_scope"] = scope
            with self.subTest(scope=scope):
                self.assertTrue(any("scope must match" in error for error in self.validate(data)))

    def test_diagnostic_passed_must_be_an_actual_boolean(self):
        for value in (None, 0, 1, "false", "true", [], {}):
            data = copy.deepcopy(self.data)
            if value is None:
                data["correctness"].pop("passed")
            else:
                data["correctness"]["passed"] = value
            with self.subTest(value=value):
                self.assertTrue(any("must be a boolean" in error for error in self.validate(data)))

    def test_diagnostic_requires_a_nonempty_text_contract(self):
        for contract in (None, "", " \n\t", False, 42, ["contract"]):
            data = copy.deepcopy(self.data)
            if contract is None:
                data["correctness"].pop("tested_contract")
            else:
                data["correctness"]["tested_contract"] = contract
            with self.subTest(contract=contract):
                self.assertTrue(any("tested contract" in error for error in self.validate(data)))

    def test_diagnostic_purpose_is_mandatory_and_exact(self):
        for purpose in (None, "", "performance", "correctness", True):
            data = copy.deepcopy(self.data)
            if purpose is None:
                data["measurement"].pop("purpose")
            else:
                data["measurement"]["purpose"] = purpose
            with self.subTest(purpose=purpose):
                self.assertTrue(any("purpose must be correctness_diagnostic" in error for error in self.validate(data)))

    def test_diagnostic_explicitly_refuses_performance_acceptance(self):
        for passed in (False, True):
            for accepted in (None, True, 0, "false", "no", []):
                data = copy.deepcopy(self.data)
                data["correctness"]["passed"] = passed
                if accepted is None:
                    data["measurement"].pop("performance_accepted")
                else:
                    data["measurement"]["performance_accepted"] = accepted
                with self.subTest(passed=passed, accepted=accepted):
                    self.assertTrue(any("performance_accepted must be false" in error for error in self.validate(data)))

    def test_result_file_and_common_fields_remain_required(self):
        self.assertTrue(any("requires result" in error for error in self.validate(write_result=False)))
        for field in ("run_id", "source_commit", "device", "environment", "correctness", "measurement", "limitations"):
            data = copy.deepcopy(self.data)
            data.pop(field)
            with self.subTest(field=field):
                self.assertTrue(any(f"result missing {field}" in error for error in self.validate(data)))

    def test_no_other_scope_uses_the_diagnostic_exception(self):
        for scope in ("protocol", "compile-only", "external-observation", "upstream-source"):
            with self.subTest(scope=scope):
                self.assertTrue(any("requires result and scope" in error
                                    for error in self.validate(row=dict(self.row, evidence_scope=scope))))

    def test_nonobject_result_sections_are_rejected(self):
        self.assertTrue(any("result must be an object" in error for error in self.validate(data=[])))
        for field in ("correctness", "measurement"):
            data = copy.deepcopy(self.data)
            data[field] = "not a result section"
            with self.subTest(field=field):
                self.assertTrue(self.validate(data))


if __name__ == "__main__":
    unittest.main()
