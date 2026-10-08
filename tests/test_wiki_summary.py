"""CLI retrieval tests use temporary document fixtures, not device evidence."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/wiki.py"
START = "<!-- kernelwiki:summary:start -->"
END = "<!-- kernelwiki:summary:end -->"


class WikiSummaryTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "scripts").mkdir()
        (self.root / "data").mkdir()
        self.script = self.root / "scripts/wiki.py"
        shutil.copy2(SCRIPT, self.script)
        self.page = self.root / "fixture.md"
        row = dict(id="fixture-note", title="Fixture note", tags=["fixture"],
                   confidence="documented", evidence_scope="protocol", path="fixture.md",
                   sources=["fixture"], limitations="Temporary CLI fixture; no device evidence.")
        (self.root / "data/catalog.json").write_text(json.dumps(
            dict(schema="metax-kernelwiki.catalog.v1", entries=[row])))
        self.summary = "## Current findings\n\nFixture finding with [details](#details)."
        self.full_page = f"# Before summary\n\n{START}\n{self.summary}\n{END}\n\n## Details\nRetained full-page detail.\n"
        self.page.write_text(self.full_page)

    def run_cli(self, *arguments):
        return subprocess.run([sys.executable, str(self.script), *arguments],
                              capture_output=True, text=True, check=False)

    def test_full_show_preserves_the_entire_page_by_default(self):
        result = self.run_cli("show", "fixture-note")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(self.full_page, result.stdout)
        self.assertIn("documented / protocol", result.stdout)

    def test_summary_shows_only_the_canonical_block_and_entry_context(self):
        result = self.run_cli("show", "fixture-note", "--summary")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(self.summary, result.stdout)
        self.assertIn("fixture-note | Fixture note | documented / protocol", result.stdout)
        self.assertNotIn(START, result.stdout)
        self.assertNotIn(END, result.stdout)
        self.assertNotIn("Before summary", result.stdout)
        self.assertNotIn("Retained full-page detail", result.stdout)
        self.assertEqual(self.page.read_text(), self.full_page)

    def test_absent_summary_fails_without_dumping_the_page(self):
        self.page.write_text("Full-page content with no summary markers.\n")
        full = self.run_cli("show", "fixture-note")
        self.assertEqual(full.returncode, 0, full.stderr)
        self.assertIn(self.page.read_text(), full.stdout)
        result = self.run_cli("show", "fixture-note", "--summary")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No summary block is defined", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_malformed_markers_fail_without_falling_back_to_full_content(self):
        cases = {
            "missing end": f"{START}\nContent",
            "missing start": f"Content\n{END}",
            "duplicate start": f"{START}\n{START}\nContent\n{END}",
            "duplicate end": f"{START}\nContent\n{END}\n{END}",
            "two blocks": f"{START}\nOne\n{END}\n{START}\nTwo\n{END}",
            "reversed": f"{END}\nContent\n{START}",
            "inexact marker": "<!-- kernelwiki:summary:start-->\nContent\n" + END,
        }
        for label, page in cases.items():
            with self.subTest(label=label):
                self.page.write_text(page)
                result = self.run_cli("show", "fixture-note", "--summary")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Summary", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_whitespace_only_summary_is_rejected(self):
        self.page.write_text(f"{START}\n \t\n{END}")
        result = self.run_cli("show", "fixture-note", "--summary")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Summary block is empty", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_summary_flag_is_rejected_for_other_commands(self):
        for command in ("list", "search", "validate"):
            with self.subTest(command=command):
                result = self.run_cli(command, "--summary")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--summary is only valid with show", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_summary_requires_an_entry_id(self):
        result = self.run_cli("show", "--summary")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires an entry ID", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
