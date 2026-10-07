#!/usr/bin/env python3
"""Offline C550 knowledge retrieval; the catalog is the single index."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIDENCE = {"documented", "source-reported", "inferred", "experimental", "locally-measured"}
SCOPES = {"external-observation", "upstream-source", "protocol", "compile-only", "device-correctness", "local-measurement"}


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
            if row.get("evidence_scope") != "local-measurement" or not result.is_file():
                errors.append(f"{ident}: local measurement requires result and scope")
            else:
                data = json.loads(result.read_text())
                for key in ("run_id", "source_commit", "device", "environment", "correctness", "measurement", "limitations"):
                    if not data.get(key):
                        errors.append(f"{ident}: result missing {key}")
                if data.get("correctness", {}).get("passed") is not True:
                    errors.append(f"{ident}: measured entry needs passing full-output check")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "search", "show", "validate"))
    parser.add_argument("query", nargs="?", default="")
    args = parser.parse_args()
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
            print(f"{row['id']} | {row['title']} | {row['confidence']} / {row['evidence_scope']}")
            print(f"  {row['path']}\n  {row['limitations']}")
            if args.command == "show":
                print((ROOT / row["path"]).read_text())
    return not bool(matches)


if __name__ == "__main__":
    raise SystemExit(main())
