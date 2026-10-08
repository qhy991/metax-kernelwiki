#!/usr/bin/env python3
"""CPU preparation and exact dyadic oracle for one native MACA WMMA tile."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import struct
import sys

EXPERIMENT = "native-wmma-fp16-fp32-16x16"
CONTROL_EXPERIMENT = "wmma-scalar-fp32-input-control"
PREFIX_EXPERIMENT = "wmma-scalar-fp32-prefix-control"
PREFIX_SUITE = "prefix-control"
PREFIX_FAMILIES = {"singleton": 1, "dense": 16}
WITNESS_EXPERIMENT = "wmma-scalar-fp32-witness-control"
WITNESS_SUITE = "witness-control"
WITNESS_PATTERNS = {
    "dense-origin": (13, 2, "dense-formulas"),
    "isolated-origin": (13, 2, "isolated-fixed-pairs"),
    "isolated-c00": (0, 0, "isolated-fixed-pairs"),
}
PRODUCT_EXPERIMENT = "wmma-scalar-fp32-product-control"
PRODUCT_SUITE = "product-control"
PRODUCT_PATTERNS = {
    "positive-k0": ((-12,0),(-1,0)),
    "positive-k1": ((0,-12),(0,-1)),
    "negative-k0": ((1,0),(-13,0)),
    "negative-k1": ((0,1),(0,-13)),
    "pair-forward": ((-12,1),(-1,-13)),
    "pair-reversed": ((1,-12),(-13,-1)),
}
SIGN_EXPERIMENT = "wmma-scalar-fp32-sign-control"
SIGN_SUITE = "sign-control"
SIGN_PATTERNS = {
    "positive12-k0": ((-12,0),(-1,0)),
    "negative12-k0": ((12,0),(-1,0)),
    "positive13-k1": ((0,-1),(0,-13)),
    "negative13-k1": ((0,1),(0,-13)),
    "pair-pp": ((-12,-1),(-1,-13)),
    "pair-pn": ((-12,1),(-1,-13)),
    "pair-np": ((12,-1),(-1,-13)),
    "pair-nn": ((12,1),(-1,-13)),
}
SIGN_PRODUCT_BASELINES = {"positive12-k0":"positive-k0", "negative13-k1":"negative-k1", "pair-pn":"pair-forward"}
# Only the two row0/column0 ordered-pair suites share this implementation.
ORDERED_PATTERN_SUITES = {
    PRODUCT_SUITE: ("product",PRODUCT_PATTERNS,"isolated-ordered-products",2),
    SIGN_SUITE: ("sign",SIGN_PATTERNS,"isolated-signed-products",3),
}
SHAPES = ((16,16,0),(1,1,1),(16,16,16),(15,16,16),(16,15,16),(15,15,15),
          (7,9,17),(15,16,31),(16,15,32),(16,16,33),(9,7,63),(16,16,64))
COLUMNS = ("id","m","n","k","warmups","samples","launches")
CONTROL_COLUMNS = ("id","m","n","k","order","warmups","samples","launches")
PREFIX_COLUMNS = ("id","m","n","k","suite","family","order","warmups","samples","launches")
WITNESS_COLUMNS = ("id","m","n","k","suite","pattern","target_row","target_col","input_rule","order","warmups","samples","launches")
PRODUCT_COLUMNS = WITNESS_COLUMNS
SIGN_COLUMNS = WITNESS_COLUMNS
ORDERS = {"wmma-first": ("wmma","scalar"), "scalar-first": ("scalar","wmma")}
SNAPSHOT_PHASES = ("before","between","after")
TILE = 16
PACKED_CHUNKS = 4
OPERAND_HALFWORDS = 1024
OUTPUT_WORDS = 256
GUARD_WORDS = 64
GUARD_VALUE = 0xFFFFFFFF
ENVIRONMENT = ("MACA_LAUNCH_MODE","MACA_LAUNCH_BLOCKING","MACA_DIRECT_DISPATCH","MACA_CACHE_PATH","MACA_CACHE_DISABLE")


def a_numerator(i: int, k: int) -> int:
    return (67 * i + 13 * k) % 31 - 15


def b_numerator(k: int, j: int) -> int:
    return (17 * k + 5 * j + 3) % 29 - 14


def validate_case(case: dict, *, prefix: bool = False) -> None:
    shape = tuple(case[field] for field in ("m","n","k"))
    if prefix:
        side=PREFIX_FAMILIES.get(case.get("family"))
        if (case.get("suite")!=PREFIX_SUITE or side is None
                or any(type(value) is not int for value in shape)
                or shape[:2]!=(side,side) or not 0<=shape[2]<=16):
            raise ValueError("Shape, suite or family outside the fixed WMMA prefix cases")
        return
    if any(type(value) is not int for value in shape) or shape not in SHAPES:
        raise ValueError("Shape outside the fixed WMMA boundary cases")


def default_cases() -> list[dict]:
    return [dict(id=f"wmma_m{m}_n{n}_k{k}",m=m,n=n,k=k,warmups=10,samples=10,launches=10) for m,n,k in SHAPES]


def read_plan(path: Path, control: bool = False, *, prefix: bool = False) -> list[dict]:
    if prefix and not control:
        raise ValueError("WMMA prefix cases require paired scalar control")
    columns = PREFIX_COLUMNS if prefix else CONTROL_COLUMNS if control else COLUMNS
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle,delimiter="\t")
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError("Unsupported cases.tsv header")
        rows = list(reader)
    maximum = 34 if prefix else 12
    if not 1 <= len(rows) <= maximum:
        raise ValueError(f"A plan must contain 1 to {maximum} cases")
    result, ids = [], set()
    for row in rows:
        if None in row or any(row[field] is None for field in columns):
            raise ValueError("Malformed case row")
        case = {field:row[field] if field in ("id","suite","family","order") else int(row[field]) for field in columns}
        if (not case["id"] or case["id"] in ids
                or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for ch in case["id"])):
            raise ValueError("Invalid or duplicate case id")
        validate_case(case,prefix=prefix)
        if control and case["order"] not in ORDERS:
            raise ValueError("Unknown WMMA/scalar variant order")
        if (case["warmups"],case["samples"],case["launches"]) != (10,10,10):
            raise ValueError("Unsupported timing protocol")
        result.append(case)
        ids.add(case["id"])
    return result


def validate_witness_case(case: dict) -> None:
    expected=WITNESS_PATTERNS.get(case.get("pattern"))
    if (case.get("suite")!=WITNESS_SUITE or expected is None
            or any(type(case.get(field)) is not int for field in ("m","n","k","target_row","target_col"))
            or (case["m"],case["n"],case["k"])!=(16,16,2)
            or (case["target_row"],case["target_col"],case.get("input_rule"))!=expected):
        raise ValueError("Shape, pattern, target or input rule outside the fixed WMMA witness cases")


def witness_cases(order: str = "wmma-first") -> list[dict]:
    if order not in ORDERS:
        raise ValueError("Unknown WMMA/scalar variant order")
    return [dict(id="witness_"+pattern.replace("-","_"),m=16,n=16,k=2,suite=WITNESS_SUITE,
                 pattern=pattern,target_row=row,target_col=col,input_rule=rule,order=order,
                 warmups=10,samples=10,launches=10)
            for pattern,(row,col,rule) in WITNESS_PATTERNS.items()]


def read_witness_plan(path: Path) -> list[dict]:
    return _read_pattern_plan(path,WITNESS_SUITE)


def read_product_plan(path: Path) -> list[dict]:
    return _read_pattern_plan(path,PRODUCT_SUITE)


def read_sign_plan(path: Path) -> list[dict]:
    return _read_pattern_plan(path,SIGN_SUITE)


def _read_pattern_plan(path: Path, suite: str) -> list[dict]:
    label,maximum,validate={WITNESS_SUITE:("witness",3,validate_witness_case),
                            PRODUCT_SUITE:("product",6,validate_product_case),
                            SIGN_SUITE:("sign",8,validate_sign_case)}[suite]
    with path.open(newline="") as handle:
        reader=csv.DictReader(handle,delimiter="\t")
        if tuple(reader.fieldnames or ())!=WITNESS_COLUMNS:
            raise ValueError(f"Unsupported {label} cases.tsv header")
        rows=list(reader)
    if not 1<=len(rows)<=maximum:
        raise ValueError(f"A {label} plan must contain 1 to {maximum} cases")
    cases,ids,patterns=[],set(),set()
    text_fields={"id","suite","pattern","input_rule","order"}
    for row in rows:
        if None in row or any(row[field] is None for field in WITNESS_COLUMNS):
            raise ValueError(f"Malformed {label} case row")
        case={field:row[field] if field in text_fields else int(row[field]) for field in WITNESS_COLUMNS}
        if (not case["id"] or case["id"] in ids
                or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for ch in case["id"])):
            raise ValueError("Invalid or duplicate case id")
        validate(case)
        if case["pattern"] in patterns:
            raise ValueError(f"Duplicate {label} pattern")
        if case["order"] not in ORDERS:
            raise ValueError("Unknown WMMA/scalar variant order")
        if (case["warmups"],case["samples"],case["launches"])!=(10,10,10):
            raise ValueError("Unsupported timing protocol")
        ids.add(case["id"]);patterns.add(case["pattern"]);cases.append(case)
    return cases


def witness_packed_inputs(case: dict) -> tuple[list[int],list[int]]:
    validate_witness_case(case)
    a,b=[0]*OPERAND_HALFWORDS,[0]*OPERAND_HALFWORDS
    for outer in range(16):
        for k in range(2):
            if case["pattern"]=="dense-origin":
                an,bn=a_numerator(outer,k),b_numerator(k,outer)
            else:
                an=(-12,1)[k] if outer==case["target_row"] else 0
                bn=(-1,-13)[k] if outer==case["target_col"] else 0
            a[outer*16+k]=halfword(an)
            b[outer*16+k]=halfword(bn)
    return a,b


def validate_witness_inputs(case: dict, a: list[int], b: list[int]) -> dict:
    expected=witness_packed_inputs(case)
    for label,actual,want in (("A",a,expected[0]),("B",b,expected[1])):
        if len(actual)!=1024 or actual!=want:
            raise ValueError(f"{case['id']}: packed witness {label} input words differ from the contract")
    masked=0 if case["pattern"]=="dense-origin" else 30
    return dict(input_halfwords_checked=2048,a_padding_halfwords_checked=992,b_padding_halfwords_checked=992,
                a_pattern_masked_halfwords_checked=masked,b_pattern_masked_halfwords_checked=masked)


def witness_reference_output(case: dict) -> list[float]:
    """Logical integer oracle, independent of packed slots and either device kernel."""
    validate_witness_case(case)
    result=[0.0]*256
    if case["pattern"]=="dense-origin":
        for i in range(16):
            for j in range(16):
                result[i*16+j]=sum(a_numerator(i,k)*b_numerator(k,j) for k in range(2))/256
    else:
        result[case["target_row"]*16+case["target_col"]]=((-12)*(-1)+1*(-13))/256
    return result


def witness_pattern_contracts() -> list[dict]:
    return [dict(pattern=pattern,target_row=row,target_col=col,input_rule=rule)
            for pattern,(row,col,rule) in WITNESS_PATTERNS.items()]


def validate_product_case(case: dict) -> None:
    _validate_ordered_case(case,PRODUCT_SUITE)


def validate_sign_case(case: dict) -> None:
    _validate_ordered_case(case,SIGN_SUITE)


def _validate_ordered_case(case: dict, suite: str) -> None:
    label,patterns,rule,_=ORDERED_PATTERN_SUITES[suite]
    if (case.get("suite")!=suite or case.get("pattern") not in patterns
            or any(type(case.get(field)) is not int for field in ("m","n","k","target_row","target_col"))
            or (case["m"],case["n"],case["k"],case["target_row"],case["target_col"])!=(16,16,2,0,0)
            or case.get("input_rule")!=rule):
        raise ValueError(f"Shape, pattern, target or input rule outside the fixed WMMA {label} cases")


def product_cases(order: str = "wmma-first") -> list[dict]:
    return _ordered_cases(PRODUCT_SUITE,order)


def sign_cases(order: str = "wmma-first") -> list[dict]:
    return _ordered_cases(SIGN_SUITE,order)


def _ordered_cases(suite: str, order: str) -> list[dict]:
    if order not in ORDERS:
        raise ValueError("Unknown WMMA/scalar variant order")
    label,patterns,rule,_=ORDERED_PATTERN_SUITES[suite]
    return [dict(id=label+"_"+pattern.replace("-","_"),m=16,n=16,k=2,suite=suite,
                 pattern=pattern,target_row=0,target_col=0,input_rule=rule,order=order,
                 warmups=10,samples=10,launches=10) for pattern in patterns]


def product_packed_inputs(case: dict) -> tuple[list[int],list[int]]:
    return _ordered_packed_inputs(case,PRODUCT_SUITE)


def sign_packed_inputs(case: dict) -> tuple[list[int],list[int]]:
    return _ordered_packed_inputs(case,SIGN_SUITE)


def _ordered_packed_inputs(case: dict, suite: str) -> tuple[list[int],list[int]]:
    _validate_ordered_case(case,suite)
    a,b=[0]*1024,[0]*1024
    left,right=ORDERED_PATTERN_SUITES[suite][1][case["pattern"]]
    for k in range(2):
        a[k],b[k]=halfword(left[k]),halfword(right[k])
    return a,b


def validate_product_inputs(case: dict, a: list[int], b: list[int]) -> dict:
    return _validate_ordered_inputs(case,a,b,PRODUCT_SUITE)


def validate_sign_inputs(case: dict, a: list[int], b: list[int]) -> dict:
    return _validate_ordered_inputs(case,a,b,SIGN_SUITE)


def _validate_ordered_inputs(case: dict, a: list[int], b: list[int], suite: str) -> dict:
    expected=_ordered_packed_inputs(case,suite)
    suite_label=ORDERED_PATTERN_SUITES[suite][0]
    for label,actual,want in (("A",a,expected[0]),("B",b,expected[1])):
        if len(actual)!=1024 or actual!=want:
            raise ValueError(f"{case['id']}: packed {suite_label} {label} input words differ from the contract")
    return dict(input_halfwords_checked=2048,a_padding_halfwords_checked=992,b_padding_halfwords_checked=992,
                a_pattern_masked_halfwords_checked=30,b_pattern_masked_halfwords_checked=30)


def product_reference_output(case: dict) -> list[float]:
    return _ordered_reference_output(case,PRODUCT_SUITE)


def sign_reference_output(case: dict) -> list[float]:
    return _ordered_reference_output(case,SIGN_SUITE)


def _ordered_reference_output(case: dict, suite: str) -> list[float]:
    """Two logical integer products; no packed-array or kernel indexing is reused."""
    _validate_ordered_case(case,suite)
    left,right=ORDERED_PATTERN_SUITES[suite][1][case["pattern"]]
    result=[0.0]*256
    result[0]=sum(a*b for a,b in zip(left,right))/256
    return result


def product_pattern_contracts() -> list[dict]:
    return _ordered_pattern_contracts(PRODUCT_SUITE)


def sign_pattern_contracts() -> list[dict]:
    return _ordered_pattern_contracts(SIGN_SUITE)


def _ordered_pattern_contracts(suite: str) -> list[dict]:
    _,patterns,rule,_=ORDERED_PATTERN_SUITES[suite]
    result=[]
    for pattern,(left,right) in patterns.items():
        products=[a*b for a,b in zip(left,right)]
        row=dict(pattern=pattern,target_row=0,target_col=0,input_rule=rule,
                           k_slots=[0,1],a_numerators=list(left),b_numerators=list(right),product_numerators=products,
                           nonzero_product_k_slots=[k for k,value in enumerate(products) if value!=0],
                 target_reference_numerator=sum(products))
        if suite==SIGN_SUITE:
            row.update(product_signs=[(value>0)-(value<0) for value in products],
                       product_magnitudes=[abs(value) for value in products],
                       matching_product_pattern=SIGN_PRODUCT_BASELINES.get(pattern))
        result.append(row)
    return result


def product_case_fields(case: dict) -> dict:
    return _ordered_case_fields(case,PRODUCT_SUITE)


def sign_case_fields(case: dict) -> dict:
    return _ordered_case_fields(case,SIGN_SUITE)


def _ordered_case_fields(case: dict, suite: str) -> dict:
    _validate_ordered_case(case,suite)
    contract=next(row for row in _ordered_pattern_contracts(suite) if row["pattern"]==case["pattern"])
    return dict(suite=suite,**contract)


def product_protocol_metadata() -> dict:
    return dict(witness_mode=2,suite=PRODUCT_SUITE,product_patterns=product_pattern_contracts(),
                logical_shape=[16,16,2],maximum_cases=6,input_scale_denominator=16,target_reference_denominator=256,
                product_rule="only A row0 and B column0 contain the declared ordered pairs at k0,k1; all other input words are positive zero")


def sign_protocol_metadata() -> dict:
    return dict(witness_mode=3,suite=SIGN_SUITE,sign_patterns=sign_pattern_contracts(),
                logical_shape=[16,16,2],maximum_cases=8,input_scale_denominator=16,target_reference_denominator=256,
                sign_rule="only A row0 and B column0 contain the declared ordered pairs at k0,k1; all other input words are positive zero",
                paired_b_numerators=[-1,-13],
                standalone_inactive_slot_policy="both A and B are positive zero at the inactive K slot",
                baseline_scope="matching_product_pattern identifies a declared equal operand contract, not a new device comparison")


def halfword(numerator: int) -> int:
    return struct.unpack("<H",struct.pack("<e",numerator / 16))[0]


def packed_inputs(case: dict, *, prefix: bool = False) -> tuple[list[int],list[int]]:
    validate_case(case,prefix=prefix)
    a, b = [0]*OPERAND_HALFWORDS, [0]*OPERAND_HALFWORDS
    for chunk in range(PACKED_CHUNKS):
        for i in range(TILE):
            for local_k in range(TILE):
                k = chunk * TILE + local_k
                if i < case["m"] and k < case["k"]:
                    a[chunk*256+i*16+local_k] = halfword(a_numerator(i,k))
        for j in range(TILE):
            for local_k in range(TILE):
                k = chunk * TILE + local_k
                if j < case["n"] and k < case["k"]:
                    b[chunk*256+j*16+local_k] = halfword(b_numerator(k,j))
    return a,b


def validate_inputs(case: dict, a: list[int], b: list[int], *, prefix: bool = False) -> dict:
    expected_a, expected_b = packed_inputs(case,prefix=prefix)
    for label, actual, expected in (("A",a,expected_a),("B",b,expected_b)):
        if len(actual) != OPERAND_HALFWORDS or actual != expected:
            raise ValueError(f"{case['id']}: packed {label} input words differ from the contract")
    return dict(input_halfwords_checked=2*OPERAND_HALFWORDS,
                a_padding_halfwords_checked=OPERAND_HALFWORDS-case["m"]*case["k"],
                b_padding_halfwords_checked=OPERAND_HALFWORDS-case["n"]*case["k"])


def reference_output(case: dict, *, prefix: bool = False) -> list[float]:
    """Independent logical coordinates; does not consume packed tiles or WMMA indices."""
    validate_case(case,prefix=prefix)
    result = [0.0]*OUTPUT_WORDS
    for i in range(case["m"]):
        for j in range(case["n"]):
            numerator = sum(a_numerator(i,k)*b_numerator(k,j) for k in range(case["k"]))
            result[i*16+j] = numerator / 256
    return result


def read_words(path: Path, count: int, width: int) -> list[int]:
    data = path.read_bytes()
    if len(data) != count*width:
        raise ValueError(f"{path.name}: incorrect packed file extent")
    return list(struct.unpack("<"+("H" if width==2 else "I")*count,data))


def validate_output(case: dict, words: list[int], *, prefix: bool = False) -> dict:
    if len(words) != OUTPUT_WORDS + 2*GUARD_WORDS:
        raise ValueError(f"{case['id']}: incorrect C output extent")
    for side, guard in (("prefix",words[:GUARD_WORDS]),("suffix",words[-GUARD_WORDS:])):
        if guard != [GUARD_VALUE]*GUARD_WORDS:
            raise ValueError(f"{case['id']}: {side} guard overwritten")
    values = struct.unpack("<256f",struct.pack("<256I",*words[GUARD_WORDS:-GUARD_WORDS]))
    expected = reference_output(case,prefix=prefix)
    for index,(actual,want) in enumerate(zip(values,expected)):
        if not math.isfinite(actual) or actual != want:
            raise ValueError(f"{case['id']}: nonfinite or unequal output at row={index//16}, col={index%16}")
    return dict(payload_elements_checked=256,finite_count=256,logical_outputs_checked=case["m"]*case["n"],
                padded_outputs_checked=256-case["m"]*case["n"],zero_outputs_checked=expected.count(0.0),
                guard_elements_checked=128,mismatches=0)


def oracle_metadata() -> dict:
    return dict(schema_version=1,experiment=EXPERIMENT,operand_dtype="little-endian IEEE-754 binary16",
                output_dtype="little-endian IEEE-754 binary32",tile=[16,16,16],packed_chunks=4,
                operand_halfwords_each=1024,a_layout="row_major",b_layout="col_major",c_layout="row_major",
                leading_dimension=16,input_scale_denominator=16,output_scale_denominator=256,
                a_rule="A_num(i,k)=((67*i+13*k)%31)-15",b_rule="B_num(k,j)=((17*k+5*j+3)%29)-14",
                padding="logical M/N/K padding and unused packed chunks are positive zero",
                reference="independent logical i,j,k integer numerator sum divided by 256; no packed-input multiplication",
                comparison="all 256 FP32 values finite and numerically exact; signed zero equivalent; no tolerance",
                accumulator_prefix_numerator_bound=13440,guard_elements_each_side=64,guard_uint32=GUARD_VALUE)


def control_oracle_metadata(*, prefix: bool = False, suite: str | None = None) -> dict:
    if suite is not None and (suite not in (WITNESS_SUITE,PRODUCT_SUITE,SIGN_SUITE) or prefix):
        raise ValueError("Unknown or incompatible paired control suite")
    result = {**oracle_metadata(), "experiment": CONTROL_EXPERIMENT,
            "variants": ["wmma","scalar"], "input_snapshots": list(SNAPSHOT_PHASES),
            "input_policy": "one A/B H2D per logical case; same allocations and no rewrite between variants",
            "output_policy": "separate guarded C allocations; each variant stores all 256 values",
            "scalar_policy": "one thread per C element; float operands and accumulator; same full padded K chunks",
            "analysis_policy": "complete per-variant exact diagnostics and snapshot equality; no tolerance or performance acceptance"}
    if prefix:
        result.update(experiment=PREFIX_EXPERIMENT,suite=PREFIX_SUITE,
                      families={family:dict(m=side,n=side,k_inclusive=[0,16]) for family,side in PREFIX_FAMILIES.items()},
                      maximum_cases=34,
                      prefix_scope="logical M=N=1 or16 and K0..16; unchanged physical16x16 tile and full participation")
    if suite==WITNESS_SUITE:
        result.pop("a_rule");result.pop("b_rule")
        result.update(experiment=WITNESS_EXPERIMENT,suite=WITNESS_SUITE,patterns=witness_pattern_contracts(),
                      logical_shape=[16,16,2],maximum_cases=3,
                      dense_a_rule="A_num(i,k)=((67*i+13*k)%31)-15",
                      dense_b_rule="B_num(k,j)=((17*k+5*j+3)%29)-14",
                      isolated_a_numerators=[-12,1],isolated_b_numerators=[-1,-13],
                      isolated_rule="only declared A target_row and B target_col retain the ordered pairs at k0,k1; all other input words are positive zero",
                      target_reference_numerator=-1,target_reference_denominator=256,
                      reference="dense: independent logical integer dot/256; isolated: only declared target cell is -1/256",
                      target_coordinate_scope="logical matrix coordinates, not a hardware lane or fragment mapping")
    if suite in ORDERED_PATTERN_SUITES:
        result.pop("a_rule");result.pop("b_rule")
        protocol=sign_protocol_metadata() if suite==SIGN_SUITE else product_protocol_metadata()
        result.update(experiment=SIGN_EXPERIMENT if suite==SIGN_SUITE else PRODUCT_EXPERIMENT,**protocol,
                      reference="only C00 is the sum of the two declared integer products/256; all other outputs are zero",
                      target_coordinate_scope="logical matrix coordinates, not a hardware lane or fragment mapping",
                      k_slot_scope="declared logical K positions; no native accumulation-order assertion")
    return result


def control_cases(order: str = "wmma-first") -> list[dict]:
    if order not in ORDERS:
        raise ValueError("Unknown WMMA/scalar variant order")
    return [dict(case, order=order) for case in default_cases()]


def prefix_cases(order: str = "wmma-first") -> list[dict]:
    if order not in ORDERS:
        raise ValueError("Unknown WMMA/scalar variant order")
    return [dict(id=f"prefix_{family}_k{k}",m=side,n=side,k=k,suite=PREFIX_SUITE,family=family,
                 order=order,warmups=10,samples=10,launches=10)
            for family,side in PREFIX_FAMILIES.items() for k in range(17)]


def prepare(destination: Path, suite: str = "default", order: str | None = None) -> dict:
    if suite not in ("default","scalar-control",PREFIX_SUITE,WITNESS_SUITE,PRODUCT_SUITE,SIGN_SUITE) or (suite=="default" and order is not None):
        raise ValueError("Order is only supported by paired control suites")
    prefix = suite==PREFIX_SUITE
    witness = suite==WITNESS_SUITE
    control = suite!="default"
    pattern_suite=suite if suite==WITNESS_SUITE or suite in ORDERED_PATTERN_SUITES else None
    if suite in ORDERED_PATTERN_SUITES:
        cases=_ordered_cases(suite,order or "wmma-first")
    else:
        cases = witness_cases(order or "wmma-first") if witness else prefix_cases(order or "wmma-first") if prefix else control_cases(order or "wmma-first") if control else default_cases()
    destination.mkdir(parents=True,exist_ok=False)
    for case in cases:
        a,b = _pattern_packed_inputs(case,pattern_suite) if pattern_suite else packed_inputs(case,prefix=prefix)
        for suffix,words in (("a",a),("b",b)):
            (destination/f"{case['id']}.{suffix}.f16").write_bytes(struct.pack("<1024H",*words))
    with (destination/"cases.tsv").open("w",newline="") as handle:
        columns=WITNESS_COLUMNS if pattern_suite else PREFIX_COLUMNS if prefix else CONTROL_COLUMNS if control else COLUMNS
        writer=csv.DictWriter(handle,fieldnames=columns,delimiter="\t",lineterminator="\n")
        writer.writeheader()
        writer.writerows(cases)
    metadata=control_oracle_metadata(prefix=prefix,suite=pattern_suite) if control else oracle_metadata()
    (destination/"oracle.json").write_text(json.dumps(metadata,indent=2)+"\n")
    result = dict(prepared=str(destination),cases=len(cases),input_bytes_per_case=4096)
    if prefix:
        result.update(suite=PREFIX_SUITE,families=list(PREFIX_FAMILIES))
    if witness:
        result.update(suite=WITNESS_SUITE,patterns=list(WITNESS_PATTERNS))
    if suite in ORDERED_PATTERN_SUITES:
        result.update(suite=suite,patterns=list(ORDERED_PATTERN_SUITES[suite][1]))
    return result


def validate_control_case(case: dict, *, prefix: bool = False, suite: str | None = None) -> None:
    if suite is not None:
        if suite not in (WITNESS_SUITE,PRODUCT_SUITE,SIGN_SUITE) or prefix:
            raise ValueError("Unknown or incompatible paired control suite")
        if suite in ORDERED_PATTERN_SUITES:
            _validate_ordered_case(case,suite)
        else:
            validate_witness_case(case)
    else:
        validate_case(case,prefix=prefix)


def witness_case_fields(case: dict) -> dict:
    validate_witness_case(case)
    return {field:case[field] for field in ("suite","pattern","target_row","target_col","input_rule")}


def _pattern_case_fields(case: dict, suite: str) -> dict:
    if suite in ORDERED_PATTERN_SUITES:
        return _ordered_case_fields(case,suite)
    if suite==WITNESS_SUITE:
        return witness_case_fields(case)
    raise ValueError("Unknown pattern control suite")


def _pattern_packed_inputs(case: dict, suite: str) -> tuple[list[int],list[int]]:
    if suite in ORDERED_PATTERN_SUITES:
        return _ordered_packed_inputs(case,suite)
    if suite==WITNESS_SUITE:
        return witness_packed_inputs(case)
    raise ValueError("Unknown pattern control suite")


def witness_protocol_metadata() -> dict:
    return dict(witness_mode=1,suite=WITNESS_SUITE,witness_patterns=witness_pattern_contracts(),
                logical_shape=[16,16,2],maximum_cases=3,
                dense_a_rule="A_num(i,k)=((67*i+13*k)%31)-15",
                dense_b_rule="B_num(k,j)=((17*k+5*j+3)%29)-14",
                isolated_a_numerators=[-12,1],isolated_b_numerators=[-1,-13],
                input_scale_denominator=16,
                isolated_rule="only declared A target_row and B target_col retain the ordered pairs at k0,k1; all other input words are positive zero",
                target_reference_numerator=-1,target_reference_denominator=256)


def case_metadata(case: dict, variant: str | None = None, *, prefix: bool = False, suite: str | None = None) -> dict:
    validate_control_case(case,prefix=prefix,suite=suite)
    if prefix and variant is None:
        raise ValueError("Prefix metadata requires an explicit paired variant")
    if suite is not None and variant is None:
        label=ORDERED_PATTERN_SUITES[suite][0].capitalize() if suite in ORDERED_PATTERN_SUITES else "Witness"
        raise ValueError(f"{label} metadata requires an explicit paired variant")
    result = dict(m=case["m"],n=case["n"],k=case["k"],tile_m=16,tile_n=16,tile_k=16,
                k_chunks=(case["k"]+15)//16,packed_chunks=4,a_layout="row_major",b_layout="col_major",c_layout="row_major",
                leading_dimension=16,operand_dtype="float16",accumulator_dtype="float32",output_elements=256,
                operand_halfwords_each=1024,physical_threads=64,block_x=64,block_y=1,block_z=1,grid_x=1,grid_y=1,grid_z=1,
                required_wave_size=64,participation="all 64 physical threads; uniform K-chunk loop",
                a_file=case["id"]+".a.f16",b_file=case["id"]+".b.f16",output_file=case["id"]+".f32",
                guard_elements_each_side=64,warmups=10,samples=10,launches_per_sample=10,total_launches=110)
    if variant is not None:
        if variant not in ("wmma","scalar") or case.get("order") not in ORDERS:
            raise ValueError("Unknown control variant or order")
        result.update(variant=variant,order=case["order"],output_file=case["id"]+f".{variant}.f32")
        if variant=="scalar":
            result.update(physical_threads=256,block_x=256,
                          participation="one thread per C element; full padded K chunks")
    if prefix:
        result.update(suite=PREFIX_SUITE,family=case["family"])
    if suite is not None:
        result.update(_pattern_case_fields(case,suite))
    return result


def validate_record_header(records: list[dict], cases: list[dict], experiment: str = EXPERIMENT) -> None:
    for kind in ("device","protocol","complete"):
        if sum(row.get("type")==kind for row in records) != 1:
            raise ValueError(f"Expected exactly one {kind} record")
    if records[-1].get("type") != "complete":
        raise ValueError("Incomplete device run")
    device=next(row for row in records if row["type"]=="device")
    protocol=next(row for row in records if row["type"]=="protocol")
    for field,value in dict(name="MetaX C550",visible_device_count=1,wave_size_api=64).items():
        if type(device.get(field)) is not type(value) or device[field] != value:
            raise ValueError(f"Device metadata mismatch: {field}")
    if not isinstance(device.get("pci_bus_id"),str) or not device["pci_bus_id"]:
        raise ValueError("Missing PCI identity")
    for field in ("runtime_version_api","driver_version_api"):
        if type(device.get(field)) is not int or device[field]<0:
            raise ValueError(f"Invalid device version field: {field}")
    for field,value in dict(schema_version=1,experiment=experiment,operand_dtype="float16",accumulator_dtype="float32",
                            output_elements=256,guard_elements_each_side=64,required_wave_size=64,
                            timer="mcEventElapsedTime",comparison="finite exact numeric equality; signed zero equivalent; no tolerance").items():
        if type(protocol.get(field)) is not type(value) or protocol[field] != value:
            raise ValueError(f"Protocol metadata mismatch: {field}")
    for field in ENVIRONMENT:
        if field not in protocol or (protocol[field] is not None and not isinstance(protocol[field],str)):
            raise ValueError(f"Missing or invalid environment field: {field}")
    if (type(records[-1].get("cases")) is not int or records[-1]["cases"] != len(cases)
            or records[-1].get("cpu_correctness_checked") is not False):
        raise ValueError("Completion metadata mismatch")


def validate_resources(record: dict) -> None:
    attributes=record.get("function_attributes_before_timing")
    if (not isinstance(attributes,dict) or any(type(attributes.get(field)) is not int or attributes[field]<minimum
            for field,minimum in (("maxThreadsPerBlock",1),("numRegs",0),("sharedSizeBytes",0),("localSizeBytes",0)))):
        raise ValueError("Missing or invalid function attributes")
    alignments=record.get("pointer_alignment_observed_bytes")
    if (not isinstance(alignments,dict) or set(alignments)!={"a","b","c_payload"}
            or any(type(value) is not int or value<=0 or value&(value-1) for value in alignments.values())):
        raise ValueError("Missing or invalid pointer-alignment observations")


def validate_records(records: list[dict], cases: list[dict]) -> dict[str,list[dict]]:
    validate_record_header(records,cases)
    by_id={case["id"]:case for case in cases}
    declarations,samples=[],{key:[] for key in by_id}
    for record in records:
        kind=record.get("type")
        if kind in ("device","protocol","complete"):
            continue
        key=record.get("id")
        if key not in by_id:
            raise ValueError(f"Unknown case: {key}")
        if kind=="case":
            if key in declarations:
                raise ValueError("Duplicate case declaration")
            for field,value in case_metadata(by_id[key]).items():
                if type(record.get(field)) is not type(value) or record[field] != value:
                    raise ValueError(f"Case metadata mismatch: {key}: {field}")
            validate_resources(record)
            declarations.append(key)
        elif kind=="sample":
            if not declarations or key!=declarations[-1]:
                raise ValueError("Sample outside its declared case")
            for field in ("event_batch_ms","host_enqueue_batch_us"):
                value=record.get(field)
                if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                    raise ValueError(f"Invalid {field}")
            samples[key].append(record)
        else:
            raise ValueError("Unknown record type")
    if declarations!=list(by_id):
        raise ValueError("Missing or misordered case declarations")
    for key in by_id:
        indices=[sample.get("sample") for sample in samples[key]]
        if any(type(index) is not int for index in indices) or indices!=list(range(10)):
            raise ValueError("Missing, duplicate or misordered samples")
    return samples


def snapshot_metadata(case: dict, phase: str, *, prefix: bool = False, suite: str | None = None) -> dict:
    if phase not in SNAPSHOT_PHASES:
        raise ValueError("Unknown input snapshot phase")
    result = dict(phase=phase,order=case["order"],operand_halfwords_each=1024,
                  a_file=f"{case['id']}.{phase}.a.f16",b_file=f"{case['id']}.{phase}.b.f16")
    if prefix:
        validate_case(case,prefix=True)
        result.update(suite=PREFIX_SUITE,family=case["family"])
    if suite is not None:
        validate_control_case(case,prefix=prefix,suite=suite)
        result.update(_pattern_case_fields(case,suite))
    return result


def validate_control_records(records: list[dict], cases: list[dict], *, prefix: bool = False, suite: str | None = None) -> dict:
    experiment=control_oracle_metadata(prefix=prefix,suite=suite)["experiment"]
    validate_record_header(records,cases,experiment)
    if records[0].get("type")!="device" or records[1].get("type")!="protocol":
        raise ValueError("Control device/protocol records must precede logical cases")
    protocol=records[1]
    if prefix:
        for field,value in dict(prefix_mode=1,suite=PREFIX_SUITE,prefix_families=list(PREFIX_FAMILIES),
                                prefix_k_min=0,prefix_k_max=16,maximum_cases=34).items():
            if type(protocol.get(field)) is not type(value) or protocol[field]!=value:
                raise ValueError(f"Prefix protocol metadata mismatch: {field}")
    if suite==WITNESS_SUITE:
        for field,value in witness_protocol_metadata().items():
            if json.dumps(protocol.get(field),sort_keys=True,allow_nan=False)!=json.dumps(value,sort_keys=True):
                raise ValueError(f"Witness protocol metadata mismatch: {field}")
    if suite in ORDERED_PATTERN_SUITES:
        expected=sign_protocol_metadata() if suite==SIGN_SUITE else product_protocol_metadata()
        for field,value in expected.items():
            if json.dumps(protocol.get(field),sort_keys=True,allow_nan=False)!=json.dumps(value,sort_keys=True):
                raise ValueError(f"{ORDERED_PATTERN_SUITES[suite][0].capitalize()} protocol metadata mismatch: {field}")
    for field,value in dict(control_mode=1,variants_per_case=2,input_snapshots=list(SNAPSHOT_PHASES),
                            input_rewrite_between_variants=False,purpose="correctness_diagnostic",performance_accepted=False).items():
        if type(protocol.get(field)) is not type(value) or protocol[field]!=value:
            raise ValueError(f"Control protocol metadata mismatch: {field}")
    if (type(records[-1].get("variant_executions")) is not int
            or records[-1]["variant_executions"]!=2*len(cases)):
        raise ValueError("Control completion variant count mismatch")
    position=2
    def take(kind,case,expected):
        nonlocal position
        if position>=len(records)-1:
            raise ValueError(f"Missing control record: {kind}")
        row=records[position]
        position+=1
        for field,value in dict(type=kind,id=case["id"],**expected).items():
            if (field not in row or type(row.get(field)) is not type(value) or row[field]!=value
                    or (suite in ORDERED_PATTERN_SUITES and isinstance(value,list)
                        and json.dumps(row[field],allow_nan=False)!=json.dumps(value))):
                raise ValueError(f"Control record order or metadata mismatch: {case['id']}: {kind}: {field}")
        return row
    result={}
    for case in cases:
        validate_control_case(case,prefix=prefix,suite=suite)
        if case.get("order") not in ORDERS:
            raise ValueError("Unknown WMMA/scalar variant order")
        fields=("m","n","k","suite","family","order") if prefix else ("m","n","k","order")
        logical={field:case[field] for field in fields}
        if suite is not None:
            logical.update(_pattern_case_fields(case,suite))
        take("logical_case",case,logical)
        snapshots={"before":take("input_snapshot",case,snapshot_metadata(case,"before",prefix=prefix,suite=suite))}
        variants={}
        for index,variant in enumerate(ORDERS[case["order"]]):
            if index:
                snapshots["between"]=take("input_snapshot",case,snapshot_metadata(case,"between",prefix=prefix,suite=suite))
            declaration=take("case",case,case_metadata(case,variant,prefix=prefix,suite=suite))
            validate_resources(declaration)
            samples=[]
            for sample in range(10):
                row=take("sample",case,dict(variant=variant,sample=sample))
                for field in ("event_batch_ms","host_enqueue_batch_us"):
                    value=row.get(field)
                    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                        raise ValueError(f"Invalid control timing: {field}")
                samples.append(row)
            variants[variant]=dict(declaration=declaration,samples=samples)
        snapshots["after"]=take("input_snapshot",case,snapshot_metadata(case,"after",prefix=prefix,suite=suite))
        result[case["id"]]=dict(variants=variants,snapshots=snapshots)
    if position!=len(records)-1:
        raise ValueError("Unexpected trailing control records")
    return result


def analyze_control_output(case: dict, words: list[int], *, prefix: bool = False, suite: str | None = None) -> dict:
    if len(words)!=384:
        raise ValueError("Incorrect control output extent")
    validate_control_case(case,prefix=prefix,suite=suite)
    expected=_ordered_reference_output(case,suite) if suite in ORDERED_PATTERN_SUITES else witness_reference_output(case) if suite==WITNESS_SUITE else reference_output(case,prefix=prefix)
    values=struct.unpack("<256f",struct.pack("<256I",*words[64:-64]))
    mismatches=[]
    for index,(observed,want) in enumerate(zip(values,expected)):
        if not math.isfinite(observed) or observed!=want:
            mismatches.append(dict(row=index//16,col=index%16,observed_word_uint32=words[64+index],
                                   observed_value=observed if math.isfinite(observed) else None,expected_value=want,
                                   kind="finite_unequal" if math.isfinite(observed) else "nonfinite"))
    guard_mismatches=[dict(side=side,index=index,observed_word_uint32=value)
                      for side,guard in (("prefix",words[:64]),("suffix",words[-64:]))
                      for index,value in enumerate(guard) if value!=GUARD_VALUE]
    return dict(exact_passed=not mismatches,payload_elements_checked=256,
                finite_count=sum(math.isfinite(value) for value in values),
                logical_outputs_checked=case["m"]*case["n"],padded_outputs_checked=256-case["m"]*case["n"],
                mismatch_count=len(mismatches),mismatches=mismatches,guards_intact=not guard_mismatches,
                guard_elements_checked=128,guard_mismatches=guard_mismatches)


def check_control(input_directory: Path, output_directory: Path, *, prefix: bool = False, suite: str | None = None) -> dict:
    metadata=control_oracle_metadata(prefix=prefix,suite=suite)
    if json.dumps(json.loads((input_directory/"oracle.json").read_text()),sort_keys=True,allow_nan=False) != json.dumps(metadata,sort_keys=True):
        raise ValueError("Unsupported control oracle metadata")
    witness=suite==WITNESS_SUITE
    cases=_read_pattern_plan(input_directory/"cases.tsv",suite) if suite is not None else read_plan(input_directory/"cases.tsv",control=True,prefix=prefix)
    prepared={}
    for case in cases:
        a=read_words(input_directory/f"{case['id']}.a.f16",1024,2)
        b=read_words(input_directory/f"{case['id']}.b.f16",1024,2)
        if suite in ORDERED_PATTERN_SUITES:
            _validate_ordered_inputs(case,a,b,suite)
        elif witness:
            validate_witness_inputs(case,a,b)
        else:
            validate_inputs(case,a,b,prefix=prefix)
        prepared[case["id"]]=(a,b)
    records=[json.loads(line) for line in (output_directory/"raw.jsonl").read_text().splitlines()]
    indexed=validate_control_records(records,cases,prefix=prefix,suite=suite)
    summaries=[]
    for case in cases:
        retained=indexed[case["id"]]
        fixed=_pattern_packed_inputs(case,suite) if suite is not None else packed_inputs(case,prefix=prefix)
        snapshots={}
        for phase,snapshot in retained["snapshots"].items():
            operands={}
            for operand,want,contract_words in zip(("a","b"),prepared[case["id"]],fixed):
                observed=read_words(output_directory/snapshot[operand+"_file"],1024,2)
                differences=[index for index,(actual,expected) in enumerate(zip(observed,want)) if actual!=expected]
                contract_differences=[index for index,(actual,expected) in enumerate(zip(observed,contract_words)) if actual!=expected]
                operands[operand]=dict(equal_to_prepared=not differences,halfwords_checked=1024,
                                       mismatch_count=len(differences),mismatch_indices=differences,
                                       equal_to_fixed_packing=not contract_differences,
                                       fixed_packing_mismatch_indices=contract_differences,
                                       file=snapshot[operand+"_file"])
            snapshots[phase]=dict(equal_to_prepared=all(row["equal_to_prepared"] for row in operands.values()),operands=operands)
            if suite is not None:
                snapshots[phase].update(_pattern_case_fields(case,suite),phase=phase,order=case["order"])
        variants={}
        for variant,data in retained["variants"].items():
            declaration=data["declaration"]
            words=read_words(output_directory/declaration["output_file"],384,4)
            row={**case_metadata(case,variant,prefix=prefix,suite=suite),**analyze_control_output(case,words,prefix=prefix,suite=suite)}
            for field in ("function_attributes_before_timing","pointer_alignment_observed_bytes"):
                row[field]=declaration[field]
            timings=[sample["event_batch_ms"]*1000/10 for sample in data["samples"]]
            row.update(event_mean_per_launch_us_median=statistics.median(timings),
                       event_mean_per_launch_us_min=min(timings),event_mean_per_launch_us_max=max(timings),sample_count=10)
            variants[variant]=row
        inputs_equal=all(snapshot["equal_to_prepared"] for snapshot in snapshots.values())
        passed=inputs_equal and all(row["exact_passed"] and row["guards_intact"] for row in variants.values())
        summary=dict(id=case["id"],m=case["m"],n=case["n"],k=case["k"],order=case["order"],
                     input_snapshots=snapshots,inputs_unchanged=inputs_equal,variants=variants,passed=passed)
        if prefix:
            summary.update(suite=PREFIX_SUITE,family=case["family"])
        if suite is not None:
            summary.update(_pattern_case_fields(case,suite))
        summaries.append(summary)
    passed=all(row["passed"] for row in summaries)
    result = dict(status="pass" if passed else "diagnostic_failed",experiment=metadata["experiment"],
                structural_valid=True,passed=passed,purpose="correctness_diagnostic",performance_accepted=False,
                logical_cases_checked=len(cases),variant_outputs_checked=2*len(cases),
                prepared_input_halfwords_checked=2*1024*len(cases),input_snapshot_halfwords_checked=6*1024*len(cases),
                payload_elements_checked=2*256*len(cases),guard_elements_checked=2*128*len(cases),cases=summaries,
                input_observation_scope="before/between/after capture boundaries only; no assertion about transient values inside a kernel",
                tested_contract="same packed operands at before/between/after snapshots; both full256 FP32 outputs finite and exact to integer-dot/256, signed zeros equivalent; all guards intact",
                timing_scope="descriptive per-variant batches; no speedup or performance acceptance")
    if prefix:
        result.update(suite=PREFIX_SUITE,families=list(PREFIX_FAMILIES),maximum_cases=34)
    if witness:
        result.update(suite=WITNESS_SUITE,patterns=witness_pattern_contracts(),maximum_cases=3)
    if suite in ORDERED_PATTERN_SUITES:
        result.update(suite=suite,patterns=_ordered_pattern_contracts(suite),maximum_cases=len(ORDERED_PATTERN_SUITES[suite][1]))
    return result


def check(input_directory: Path, output_directory: Path) -> dict:
    oracle=json.loads((input_directory/"oracle.json").read_text())
    if oracle==control_oracle_metadata(suite=SIGN_SUITE):
        return check_control(input_directory,output_directory,suite=SIGN_SUITE)
    if oracle==control_oracle_metadata(suite=PRODUCT_SUITE):
        return check_control(input_directory,output_directory,suite=PRODUCT_SUITE)
    if oracle==control_oracle_metadata(suite=WITNESS_SUITE):
        return check_control(input_directory,output_directory,suite=WITNESS_SUITE)
    if oracle==control_oracle_metadata(prefix=True):
        return check_control(input_directory,output_directory,prefix=True)
    if oracle==control_oracle_metadata():
        return check_control(input_directory,output_directory)
    if oracle != oracle_metadata():
        raise ValueError("Unsupported oracle metadata")
    cases=read_plan(input_directory/"cases.tsv")
    inputs={}
    for case in cases:
        a=read_words(input_directory/f"{case['id']}.a.f16",1024,2)
        b=read_words(input_directory/f"{case['id']}.b.f16",1024,2)
        inputs[case["id"]]=validate_inputs(case,a,b)
    records=[json.loads(line) for line in (output_directory/"raw.jsonl").read_text().splitlines()]
    samples=validate_records(records,cases)
    declarations={row["id"]:row for row in records if row["type"]=="case"}
    summaries=[]
    for case in cases:
        row=dict(id=case["id"],**case_metadata(case),**inputs[case["id"]])
        for field in ("function_attributes_before_timing","pointer_alignment_observed_bytes"):
            row[field]=declarations[case["id"]][field]
        row.update(validate_output(case,read_words(output_directory/row["output_file"],384,4)))
        timings=[sample["event_batch_ms"]*1000/10 for sample in samples[case["id"]]]
        row.update(event_mean_per_launch_us_median=statistics.median(timings),event_mean_per_launch_us_min=min(timings),
                   event_mean_per_launch_us_max=max(timings),sample_count=10)
        summaries.append(row)
    return dict(status="pass",experiment=EXPERIMENT,cases_checked=len(cases),
                checker="all packed input words validated; independent logical integer-numerator oracle; all256 finite FP32 outputs exact with signed-zero equivalence and128 guards",
                timing_scope="descriptive full tile kernel; host packing and transfers excluded; no native-instruction throughput or end-to-end GEMM claim",
                cases=summaries)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    preparation=commands.add_parser("prepare")
    preparation.add_argument("directory",type=Path)
    preparation.add_argument("--suite",choices=("default","scalar-control",PREFIX_SUITE,WITNESS_SUITE,PRODUCT_SUITE,SIGN_SUITE),default="default")
    preparation.add_argument("--order",choices=tuple(ORDERS))
    checker=commands.add_parser("check")
    checker.add_argument("input_directory",type=Path)
    checker.add_argument("output_directory",type=Path)
    args=parser.parse_args()
    try:
        result=prepare(args.directory,args.suite,args.order) if args.command=="prepare" else check(args.input_directory,args.output_directory)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 1 if result.get("status")=="diagnostic_failed" else 0
    except (OSError,ValueError,KeyError,TypeError) as error:
        print(json.dumps(dict(status="error",message=str(error))),file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
