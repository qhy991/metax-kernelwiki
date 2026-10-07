#!/usr/bin/env python3
"""Summarize mcTracer GPU kernel events without converting timestamp units.

Supports the observed mcTracer 3.5.3 export: complete events on pid 2/category 0
with kernel block/grid/memory metadata. Process ids and timestamps are used only
to select/order events; they are never emitted in the public summary.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys


RESOURCE_FIELDS = (
    "block.x", "block.y", "block.z", "grid.x", "grid.y", "grid.z",
    "mem.dynamic_shared", "mem.private_per_thread", "mem.private_total",
    "mem.registers_per_thread", "mem.static_shared", "max_block_size",
    "mtreg_occupancy(%)", "shared_memeory_occupancy(%)",
)


def number(value, label):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"Invalid {label}: expected a finite nonnegative number")
    return value


def resource_value(arguments, field):
    value = arguments
    for key in field.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return number(value, f"kernel resource {field}")


def duration_summary(values):
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values), "mean": statistics.mean(values)}


def summarize_trace(document, expected_launches, warmups):
    if (type(expected_launches) is not int or expected_launches <= 0
            or type(warmups) is not int or not 0 <= warmups < expected_launches):
        raise ValueError("Expected launches must be positive and exceed the warmup count")
    if not isinstance(document, dict) or not isinstance(document.get("traceEvents"), list):
        raise ValueError("Malformed trace: expected a traceEvents array")
    events = document["traceEvents"]
    if any(not isinstance(event, dict) for event in events):
        raise ValueError("Malformed trace event: expected an object")
    kernels = []
    for event in events:
        arguments = event.get("args")
        # PID/category alone also select memset. Kernel launch geometry and
        # resource metadata distinguish kernels from runtime APIs and transfers.
        if (event.get("ph") == "X" and event.get("pid") == 2
                and event.get("cat") in ("0", 0) and isinstance(arguments, dict)
                and all(isinstance(arguments.get(key), dict) for key in ("block", "grid", "mem"))):
            number(event.get("ts"), "kernel timestamp")
            number(event.get("dur"), "kernel duration")
            for geometry in ("block", "grid"):
                for axis in ("x", "y", "z"):
                    value = arguments[geometry].get(axis)
                    if type(value) is not int or value <= 0:
                        raise ValueError("Invalid kernel launch geometry")
            kernels.append(event)
    if not kernels:
        raise ValueError("No supported GPU kernel events; API records or process success are insufficient")
    if len(kernels) != expected_launches:
        raise ValueError(f"GPU kernel count mismatch: expected {expected_launches}, observed {len(kernels)}")
    kernels.sort(key=lambda event: event["ts"])
    durations = [event["dur"] for event in kernels]
    timed = durations[warmups:]
    resources = {}
    for field in RESOURCE_FIELDS:
        observed = [resource_value(event["args"], field) for event in kernels]
        counts = Counter(value for value in observed if value is not None)
        resources[field] = {
            "reported_count": sum(counts.values()),
            "missing_count": observed.count(None),
            "values": [{"value": value, "count": count} for value, count in sorted(counts.items())],
        }
    return {
        "schema": "metax-kernelwiki.mcTracer-summary.v1",
        "scope": "observed mcTracer 3.5.3 export; trace-captured execution only",
        "time_unit": "units_unverified",
        "duration_conversion_applied": False,
        "event_count": len(events),
        "gpu_kernel_count": len(kernels),
        "warmup_kernel_count": warmups,
        "timed_kernel_count": len(timed),
        "warmup_assignment": "caller-specified first kernels ordered by raw timestamp",
        "raw_durations": durations,
        "raw_durations_after_warmups": timed,
        "raw_duration_summary": duration_summary(durations),
        "raw_duration_summary_after_warmups": duration_summary(timed),
        "resources": resources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--expected-launches", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=0)
    args = parser.parse_args()
    try:
        try:
            contents = args.input.read_text()
        except (OSError, UnicodeError) as error:
            raise ValueError("Cannot read trace input as UTF-8") from error
        try:
            document = json.loads(contents)
        except json.JSONDecodeError as error:
            raise ValueError("Malformed trace JSON") from error
        result = summarize_trace(document, args.expected_launches, args.warmups)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except ValueError as error:
        print(json.dumps({"status": "error", "message": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
