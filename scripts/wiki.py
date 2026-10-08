#!/usr/bin/env python3
"""Offline C550 knowledge retrieval; the catalog is the single index."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIDENCE = {"documented", "source-reported", "inferred", "experimental", "locally-measured"}
SCOPES = {"external-observation", "upstream-source", "protocol", "compile-only", "device-correctness", "local-measurement"}
SUMMARY_START = "<!-- kernelwiki:summary:start -->"
SUMMARY_END = "<!-- kernelwiki:summary:end -->"


def extract_summary(page):
    if SUMMARY_START not in page and SUMMARY_END not in page:
        raise ValueError("No summary block is defined for this page.")
    if page.count(SUMMARY_START) != 1 or page.count(SUMMARY_END) != 1:
        raise ValueError("Summary requires exactly one start marker and one end marker.")
    start = page.index(SUMMARY_START) + len(SUMMARY_START)
    end = page.index(SUMMARY_END)
    if end < start:
        raise ValueError("Summary end marker must follow the start marker.")
    summary = page[start:end].strip()
    if not summary:
        raise ValueError("Summary block is empty.")
    return summary


def validate(catalog):
    errors, seen = [], set()
    if catalog.get("schema") != "metax-kernelwiki.catalog.v1":
        errors.append("unsupported catalog schema")
    for row in catalog.get("entries", []):
        ident = row.get("id", "<missing>")
        if ident in seen:
            errors.append(f"duplicate id: {ident}")
        seen.add(ident)
        for key in ("id", "title", "tags", "confidence", "evidence_scope", "path", "sources", "limitations"):
            if not row.get(key):
                errors.append(f"{ident}: missing {key}")
        if row.get("confidence") not in CONFIDENCE or row.get("evidence_scope") not in SCOPES:
            errors.append(f"{ident}: invalid evidence label")
        path = (ROOT / row.get("path", "")).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            errors.append(f"{ident}: invalid page path")
        if row.get("confidence") == "locally-measured":
            result = ROOT / row.get("result", "")
            scope = row.get("evidence_scope")
            if scope not in {"local-measurement", "device-correctness"} or not result.is_file():
                errors.append(f"{ident}: local measurement requires result and scope")
            else:
                data = json.loads(result.read_text())
                if not isinstance(data, dict):
                    errors.append(f"{ident}: result must be an object")
                    continue
                for key in ("run_id", "source_commit", "device", "environment", "correctness", "measurement", "limitations"):
                    if not data.get(key):
                        errors.append(f"{ident}: result missing {key}")
                correctness = data.get("correctness")
                correctness = correctness if isinstance(correctness, dict) else {}
                if scope == "local-measurement" and correctness.get("passed") is not True:
                    errors.append(f"{ident}: measured entry needs passing full-output check")
                elif scope == "device-correctness":
                    if data.get("evidence_scope") != "device-correctness":
                        errors.append(f"{ident}: diagnostic result scope must match device-correctness")
                    if type(correctness.get("passed")) is not bool:
                        errors.append(f"{ident}: diagnostic correctness.passed must be a boolean")
                    tested_contract = correctness.get("tested_contract")
                    if not isinstance(tested_contract, str) or not tested_contract.strip():
                        errors.append(f"{ident}: diagnostic result needs a tested contract")
                    measurement = data.get("measurement")
                    measurement = measurement if isinstance(measurement, dict) else {}
                    if measurement.get("purpose") != "correctness_diagnostic":
                        errors.append(f"{ident}: diagnostic measurement purpose must be correctness_diagnostic")
                    if measurement.get("performance_accepted") is not False:
                        errors.append(f"{ident}: diagnostic performance_accepted must be false")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "search", "show", "validate"))
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--summary", action="store_true", help="show only the page's marked summary block")
    args = parser.parse_args()
    if args.summary and args.command != "show":
        parser.error("--summary is only valid with show.")
    if args.summary and not args.query:
        parser.error("show --summary requires an entry ID.")
    catalog = json.loads((ROOT / "data/catalog.json").read_text())
    if args.command == "validate":
        errors = validate(catalog)
        print("\n".join(errors) if errors else f"valid: {len(catalog['entries'])} entries")
        return bool(errors)
    matches = []
    for row in catalog["entries"]:
        if args.command == "show":
            match = row["id"] == args.query
        else:
            match = args.query.casefold() in json.dumps(row, ensure_ascii=False).casefold()
        if match:
            matches.append(row)
            content = None
            if args.command == "show":
                content = (ROOT / row["path"]).read_text()
                if args.summary:
                    try:
                        content = extract_summary(content)
                    except ValueError as error:
                        parser.error(f"{row['id']}: {error}")
            print(f"{row['id']} | {row['title']} | {row['confidence']} / {row['evidence_scope']}")
            print(f"  {row['path']}\n  {row['limitations']}")
            if args.command == "show":
                print(content)
    return not bool(matches)


if __name__ == "__main__":
    raise SystemExit(main())
