"""CPU-only admission controls; the fixture broker never opens a GPU or lock."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest


ADAPTER = Path(__file__).resolve().parents[1] / "scripts" / "cake_local_exec.py"
MOCK_BROKER = '''\
import os
from pathlib import Path

def admit_local_job(kind):
    Path(os.environ["TEST_ADMISSION_MARKER"]).write_text("admit:" + kind)
    return "maca-test"

def observe_local_job(kind):
    return "maca-test"
'''


class AdmissionBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="metax-admission-test-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.source = self.root / "cake"
        package = self.source / "src" / "open_cake_ir" / "evaluation"
        package.mkdir(parents=True)
        (package.parent / "__init__.py").write_text("")
        (package / "__init__.py").write_text("")
        self.broker = package / "local_broker.py"
        self.broker.write_text(MOCK_BROKER)
        self.marker = self.root / "admitted.txt"
        self.receipt = self.root / "receipt.json"
        self.environment = dict(os.environ)
        for key in ("TMPDIR", "TEMP", "TMP", "GPUQ_JOB_ID", "METAL_JOB_ID",
                    "METAL_BROKER_LOCK_FD", "PYTHONPATH", "GIT_DIR", "GIT_WORK_TREE",
                    "GIT_INDEX_FILE"):
            self.environment.pop(key, None)
        self.environment.update(PYTHONDONTWRITEBYTECODE="1",
                                TEST_ADMISSION_MARKER=str(self.marker))
        self.git("init", "--quiet")
        self.git("add", "src")
        self.git("commit", "--quiet", "-m", "CPU admission fixture")
        self.commit = self.git("rev-parse", "HEAD").stdout.strip()

    def git(self, *arguments):
        return subprocess.run(
            ["git", "-C", str(self.source), "-c", "core.hooksPath=/dev/null",
             "-c", "commit.gpgSign=false", "-c", "user.name=Admission Test",
             "-c", "user.email=admission-test@example.invalid", *arguments],
            env=self.environment, text=True, capture_output=True, check=True)

    def invoke(self, *, commit=None, environment=None):
        return subprocess.run(
            [sys.executable, str(ADAPTER), "--cake-source", str(self.source),
             "--cake-commit", self.commit if commit is None else commit,
             "--device", "0", "--receipt", str(self.receipt), "--",
             sys.executable, "-c", "pass"],
            env=self.environment if environment is None else environment,
            text=True, capture_output=True, timeout=15)

    def assert_refused_before_admission(self, result, reason):
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(reason, result.stderr)
        self.assertFalse(self.marker.exists(), "broker.admit_local_job was called")
        self.assertFalse(self.receipt.exists(), "refusal created an admission receipt")

    def test_private_temp_without_tmpdir_cannot_split_lock_namespace(self):
        private_temp = self.root / "private-temp"
        private_temp.mkdir()
        environment = dict(self.environment, TEMP=str(private_temp))
        self.assertNotIn("TMPDIR", environment)
        self.assert_refused_before_admission(
            self.invoke(environment=environment), "split the existing broker namespace")

    def test_commit_mismatch_is_refused_before_admission(self):
        self.assert_refused_before_admission(
            self.invoke(commit="0" * 40), "differs from pinned commit")

    def test_dirty_owner_source_is_refused_before_admission(self):
        self.broker.write_text(MOCK_BROKER + "\n# uncommitted change\n")
        self.assert_refused_before_admission(self.invoke(), "tracked changes")

    def test_each_nested_allocation_marker_is_refused_before_admission(self):
        for key in ("GPUQ_JOB_ID", "METAL_JOB_ID", "METAL_BROKER_LOCK_FD"):
            with self.subTest(marker=key):
                environment = dict(self.environment, **{key: "already-allocated"})
                self.assert_refused_before_admission(
                    self.invoke(environment=environment), "nested allocation is refused")

    def test_clean_canonical_namespace_reaches_mock_broker(self):
        result = self.invoke(environment=dict(self.environment, TMPDIR="/tmp"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.marker.read_text(), "admit:maca")
        self.assertTrue(self.receipt.exists())


if __name__ == "__main__":
    unittest.main()
