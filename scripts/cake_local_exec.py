#!/usr/bin/env python3
"""Execute a short native probe through an existing Cake MACA broker.

This adapter imports allocation from its existing owner. It implements no lock,
queue, GPU availability inference or automatic fallback. Use an outer process
timeout. A successful exec retains the owner's lock until the probe exits.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cake-source", required=True, type=Path)
    p.add_argument("--cake-commit", required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.device < 0 or not command or not Path(command[0]).is_absolute():
        p.error("select a nonnegative physical device and an absolute executable")
    source = args.cake_source.resolve()
    if git(source, "rev-parse", "HEAD") != args.cake_commit:
        p.error("allocation source differs from pinned commit")
    if git(source, "status", "--porcelain", "--untracked-files=no"):
        p.error("allocation source has tracked changes")
    if args.receipt.exists():
        p.error("receipt exists; use a fresh run directory")
    if any(os.environ.get(k) for k in ("GPUQ_JOB_ID", "METAL_JOB_ID", "METAL_BROKER_LOCK_FD")):
        p.error("nested allocation is refused")
    sys.path.insert(0, str(source / "src"))
    from open_cake_ir.evaluation.local_broker import admit_local_job, observe_local_job

    # Legacy user-scope admission also excludes newer device-scoped workers,
    # which share the legacy lock. Do not substitute a private lock directory.
    if Path(tempfile.gettempdir()).resolve() != Path("/tmp").resolve():
        p.error("a private TMPDIR would split the existing broker namespace")
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "MACA_VISIBLE_DEVICES"):
        os.environ.pop(key, None)
    os.environ["MACA_VISIBLE_DEVICES"] = str(args.device)
    job = admit_local_job("maca")
    if observe_local_job("maca") != job:
        raise RuntimeError("broker admission changed")
    receipt = {
        "schema": "metax-kernelwiki.local-admission.v1",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "broker_job_id": job,
        "broker_source_commit": args.cake_commit,
        "allocation": "local_serialized",
        "lock_scope": "legacy user scope, shared with existing device-scope workers",
        "external_gpu_activity": "not_excluded",
        "physical_device": args.device,
        "logical_device": 0,
        "pid": os.getpid(),
        "command": command,
        "release_boundary": "probe process exit after device synchronization and output retention",
        "correctness": "pending host verification after device release",
    }
    with args.receipt.open("x") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")
    os.execvpe(command[0], command, dict(os.environ))


if __name__ == "__main__":
    main()
