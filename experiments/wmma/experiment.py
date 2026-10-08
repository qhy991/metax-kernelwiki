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
SHAPES = ((16,16,0),(1,1,1),(16,16,16),(15,16,16),(16,15,16),(15,15,15),
          (7,9,17),(15,16,31),(16,15,32),(16,16,33),(9,7,63),(16,16,64))
COLUMNS = ("id","m","n","k","warmups","samples","launches")
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


def validate_case(case: dict) -> None:
    shape = tuple(case[field] for field in ("m","n","k"))
    if any(type(value) is not int for value in shape) or shape not in SHAPES:
        raise ValueError("Shape outside the fixed WMMA boundary cases")


def default_cases() -> list[dict]:
    return [dict(id=f"wmma_m{m}_n{n}_k{k}",m=m,n=n,k=k,warmups=10,samples=10,launches=10) for m,n,k in SHAPES]


def read_plan(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle,delimiter="\t")
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("Unsupported cases.tsv header")
        rows = list(reader)
    if not 1 <= len(rows) <= 12:
        raise ValueError("A plan must contain 1 to 12 cases")
    result, ids = [], set()
    for row in rows:
        if None in row or any(row[field] is None for field in COLUMNS):
            raise ValueError("Malformed case row")
        case = {field:row[field] if field=="id" else int(row[field]) for field in COLUMNS}
        if (not case["id"] or case["id"] in ids
                or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for ch in case["id"])):
            raise ValueError("Invalid or duplicate case id")
        validate_case(case)
        if (case["warmups"],case["samples"],case["launches"]) != (10,10,10):
            raise ValueError("Unsupported timing protocol")
        result.append(case)
        ids.add(case["id"])
    return result


def halfword(numerator: int) -> int:
    return struct.unpack("<H",struct.pack("<e",numerator / 16))[0]


def packed_inputs(case: dict) -> tuple[list[int],list[int]]:
    validate_case(case)
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


def validate_inputs(case: dict, a: list[int], b: list[int]) -> dict:
    expected_a, expected_b = packed_inputs(case)
    for label, actual, expected in (("A",a,expected_a),("B",b,expected_b)):
        if len(actual) != OPERAND_HALFWORDS or actual != expected:
            raise ValueError(f"{case['id']}: packed {label} input words differ from the contract")
    return dict(input_halfwords_checked=2*OPERAND_HALFWORDS,
                a_padding_halfwords_checked=OPERAND_HALFWORDS-case["m"]*case["k"],
                b_padding_halfwords_checked=OPERAND_HALFWORDS-case["n"]*case["k"])


def reference_output(case: dict) -> list[float]:
    """Independent logical coordinates; does not consume packed tiles or WMMA indices."""
    validate_case(case)
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


def validate_output(case: dict, words: list[int]) -> dict:
    if len(words) != OUTPUT_WORDS + 2*GUARD_WORDS:
        raise ValueError(f"{case['id']}: incorrect C output extent")
    for side, guard in (("prefix",words[:GUARD_WORDS]),("suffix",words[-GUARD_WORDS:])):
        if guard != [GUARD_VALUE]*GUARD_WORDS:
            raise ValueError(f"{case['id']}: {side} guard overwritten")
    values = struct.unpack("<256f",struct.pack("<256I",*words[GUARD_WORDS:-GUARD_WORDS]))
    expected = reference_output(case)
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


def prepare(destination: Path) -> dict:
    destination.mkdir(parents=True,exist_ok=False)
    cases = default_cases()
    for case in cases:
        a,b = packed_inputs(case)
        for suffix,words in (("a",a),("b",b)):
            (destination/f"{case['id']}.{suffix}.f16").write_bytes(struct.pack("<1024H",*words))
    with (destination/"cases.tsv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=COLUMNS,delimiter="\t",lineterminator="\n")
        writer.writeheader()
        writer.writerows(cases)
    (destination/"oracle.json").write_text(json.dumps(oracle_metadata(),indent=2)+"\n")
    return dict(prepared=str(destination),cases=len(cases),input_bytes_per_case=4096)


def case_metadata(case: dict) -> dict:
    validate_case(case)
    return dict(m=case["m"],n=case["n"],k=case["k"],tile_m=16,tile_n=16,tile_k=16,
                k_chunks=(case["k"]+15)//16,packed_chunks=4,a_layout="row_major",b_layout="col_major",c_layout="row_major",
                leading_dimension=16,operand_dtype="float16",accumulator_dtype="float32",output_elements=256,
                operand_halfwords_each=1024,physical_threads=64,block_x=64,block_y=1,block_z=1,grid_x=1,grid_y=1,grid_z=1,
                required_wave_size=64,participation="all 64 physical threads; uniform K-chunk loop",
                a_file=case["id"]+".a.f16",b_file=case["id"]+".b.f16",output_file=case["id"]+".f32",
                guard_elements_each_side=64,warmups=10,samples=10,launches_per_sample=10,total_launches=110)


def validate_records(records: list[dict], cases: list[dict]) -> dict[str,list[dict]]:
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
    for field,value in dict(schema_version=1,experiment=EXPERIMENT,operand_dtype="float16",accumulator_dtype="float32",
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
            attributes=record.get("function_attributes_before_timing")
            if (not isinstance(attributes,dict) or any(type(attributes.get(field)) is not int or attributes[field]<minimum
                    for field,minimum in (("maxThreadsPerBlock",1),("numRegs",0),("sharedSizeBytes",0),("localSizeBytes",0)))):
                raise ValueError("Missing or invalid function attributes")
            alignments=record.get("pointer_alignment_observed_bytes")
            if (not isinstance(alignments,dict) or set(alignments)!={"a","b","c_payload"}
                    or any(type(value) is not int or value<=0 or value&(value-1) for value in alignments.values())):
                raise ValueError("Missing or invalid pointer-alignment observations")
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


def check(input_directory: Path, output_directory: Path) -> dict:
    if json.loads((input_directory/"oracle.json").read_text()) != oracle_metadata():
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
    commands.add_parser("prepare").add_argument("directory",type=Path)
    checker=commands.add_parser("check")
    checker.add_argument("input_directory",type=Path)
    checker.add_argument("output_directory",type=Path)
    args=parser.parse_args()
    try:
        result=prepare(args.directory) if args.command=="prepare" else check(args.input_directory,args.output_directory)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0
    except (OSError,ValueError,KeyError,TypeError) as error:
        print(json.dumps(dict(status="error",message=str(error))),file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
