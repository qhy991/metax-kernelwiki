"""CPU-only admission controls; the fixture broker never opens a GPU or lock."""
from pathlib import Path
import json
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
DEVICE_MASKS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "MACA_VISIBLE_DEVICES")
MOCK_DEVICE_BROKER = '''\
import json
import os
from pathlib import Path

def admit_local_job(kind, *, device=None, lock_scope="user", queue_seconds=0):
    masks = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "MACA_VISIBLE_DEVICES")
    Path(os.environ["TEST_ADMISSION_MARKER"]).write_text(json.dumps({
        "kind": kind, "device": device, "lock_scope": lock_scope,
        "queue_seconds": queue_seconds,
        "masks_before_owner": {key: os.environ.get(key) for key in masks},
    }))
    for key in masks:
        os.environ.pop(key, None)
    os.environ["MACA_VISIBLE_DEVICES"] = str(device)
    return "maca-device-test"

def observe_local_job(kind):
    return "maca-device-test"
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

    def use_broker(self, contents):
        self.broker.write_text(contents)
        self.git("add", "src")
        self.git("commit", "--quiet", "-m", "CPU broker API fixture")
        self.commit = self.git("rev-parse", "HEAD").stdout.strip()

    def invoke(self, *, commit=None, environment=None, lock_scope=None, device=0, child="pass"):
        scope_args = [] if lock_scope is None else ["--lock-scope", lock_scope]
        return subprocess.run(
            [sys.executable, str(ADAPTER), "--cake-source", str(self.source),
             "--cake-commit", self.commit if commit is None else commit,
             "--device", str(device), *scope_args, "--receipt", str(self.receipt), "--",
             sys.executable, "-c", child],
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
        for scope in ("user", "device"):
            with self.subTest(scope=scope):
                self.assert_refused_before_admission(
                    self.invoke(environment=environment, lock_scope=scope),
                    "split the existing broker namespace")

    def test_commit_mismatch_is_refused_before_admission(self):
        for scope in ("user", "device"):
            with self.subTest(scope=scope):
                self.assert_refused_before_admission(
                    self.invoke(commit="0" * 40, lock_scope=scope), "differs from pinned commit")

    def test_dirty_owner_source_is_refused_before_admission(self):
        self.broker.write_text(MOCK_BROKER + "\n# uncommitted change\n")
        for scope in ("user", "device"):
            with self.subTest(scope=scope):
                self.assert_refused_before_admission(self.invoke(lock_scope=scope), "tracked changes")

    def test_each_nested_allocation_marker_is_refused_before_admission(self):
        for key in ("GPUQ_JOB_ID", "METAL_JOB_ID", "METAL_BROKER_LOCK_FD"):
            for scope in ("user", "device"):
                with self.subTest(marker=key, scope=scope):
                    environment = dict(self.environment, **{key: "already-allocated"})
                    self.assert_refused_before_admission(
                        self.invoke(environment=environment, lock_scope=scope), "nested allocation is refused")

    def test_clean_canonical_namespace_reaches_mock_broker(self):
        result = self.invoke(environment=dict(self.environment, TMPDIR="/tmp"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.marker.read_text(), "admit:maca")
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(receipt["lock_scope"], "legacy user scope, shared with existing device-scope workers")

    def test_explicit_user_scope_preserves_legacy_api_and_mask(self):
        environment = dict(self.environment, TMPDIR="/tmp", **{key: "9" for key in DEVICE_MASKS})
        child = "import json, os; print(json.dumps({key: os.environ.get(key) for key in " + repr(DEVICE_MASKS) + "}))"
        result = self.invoke(environment=environment, lock_scope="user", device=3, child=child)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.marker.read_text(), "admit:maca")
        self.assertEqual(json.loads(result.stdout), {key: "3" if key == "MACA_VISIBLE_DEVICES" else None
                                                   for key in DEVICE_MASKS})

    def test_device_scope_delegates_kwargs_and_mask_to_owner(self):
        self.use_broker(MOCK_DEVICE_BROKER)
        original_masks = {key: "9" for key in DEVICE_MASKS}
        environment = dict(self.environment, TMPDIR="/tmp", **original_masks)
        child = "import json, os; print(json.dumps({key: os.environ.get(key) for key in " + repr(DEVICE_MASKS) + "}))"
        result = self.invoke(environment=environment, lock_scope="device", device=3, child=child)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.marker.read_text()), {
            "kind": "maca", "device": 3, "lock_scope": "device", "queue_seconds": 0,
            "masks_before_owner": original_masks,
        })
        self.assertEqual(json.loads(result.stdout), {key: "3" if key == "MACA_VISIBLE_DEVICES" else None
                                                   for key in DEVICE_MASKS})
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(receipt["lock_scope"], "device")
        self.assertEqual(receipt["physical_device"], 3)
        self.assertEqual(receipt["logical_device"], 0)
        self.assertEqual(receipt["broker_job_id"], "maca-device-test")

    def test_device_scope_requires_owner_api_without_fallback(self):
        for signature in ("kind", "kind, *, device=None, queue_seconds=0"):
            with self.subTest(signature=signature):
                if signature != "kind":
                    self.use_broker(MOCK_BROKER.replace("admit_local_job(kind)", f"admit_local_job({signature})"))
                result = self.invoke(environment=dict(self.environment, TMPDIR="/tmp"), lock_scope="device")
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertIn("unexpected keyword argument", result.stderr)
                self.assertFalse(self.marker.exists(), "unsupported API must not fall back to legacy admission")
                self.assertFalse(self.receipt.exists(), "unsupported API created an admission receipt")

    def test_device_scope_observation_mismatch_does_not_write_receipt(self):
        self.use_broker(MOCK_DEVICE_BROKER.replace('return "maca-device-test"\n',
                                                  'return "maca-other-job"\n', 1))
        result = self.invoke(environment=dict(self.environment, TMPDIR="/tmp"), lock_scope="device")
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertIn("broker admission changed", result.stderr)
        self.assertTrue(self.marker.exists())
        self.assertFalse(self.receipt.exists())


if __name__ == "__main__":
    unittest.main()
