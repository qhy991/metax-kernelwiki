"""Reciprocal input contracts, exact CPU oracles and negative controls; no GPU evidence."""
import copy
import csv
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

MODULE=Path(__file__).resolve().parents[1]/'experiments/wmma/experiment.py'
SPEC=importlib.util.spec_from_file_location('wmma_reciprocal',MODULE)
probe=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(probe)
SUITE='reciprocal-control'


def operands(q,ae,be,role):
    a=[Fraction(-q,16)*Fraction(2)**ae,Fraction(1,16)*Fraction(2)**ae]
    b=[Fraction(-1,16)*Fraction(2)**be,Fraction(-(q+1),16)*Fraction(2)**be]
    if role=='positive':a[1]=b[1]=Fraction(0)
    if role=='negative':a[0]=b[0]=Fraction(0)
    return a,b


def half_words(values):
    return list(struct.unpack('<2H',struct.pack('<2e',*(float(x) for x in values))))+[0]*1022


def reference(case):
    a,b=operands(case['q'],case['a_scale_exp'],case['b_scale_exp'],case['role'])
    return [float(sum(x*y for x,y in zip(a,b)))]+[0.0]*255


def output_words(values):
    return [0xffffffff]*64+list(struct.unpack('<256I',struct.pack('<256f',*values)))+[0xffffffff]*64


class WmmaReciprocalTest(unittest.TestCase):
    def write_plan(self,path,cases,columns=probe.RECIPROCAL_COLUMNS):
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=columns,delimiter='\t',lineterminator='\n')
            writer.writeheader();writer.writerows(cases)

    def test_closed45_conditions_both_orders_and_distinct_headers(self):
        expected=[(q,e,-e,r) for q in (6,7,12) for e in (-2,-1,0,1,2) for r in ('positive','negative','pair')]
        for order in probe.ORDERS:
            cases=probe.reciprocal_cases(order)
            self.assertEqual([(c['q'],c['a_scale_exp'],c['b_scale_exp'],c['role']) for c in cases],expected)
            self.assertEqual(cases[0]['pattern'],'q06-am2-bp2-positive');self.assertEqual(cases[-1]['id'],'reciprocal_q12_ap2_bm2_pair')
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'cases.tsv'
                for selected in (cases,list(reversed(cases)),cases[:1]):
                    self.write_plan(path,selected);self.assertEqual(probe.read_reciprocal_plan(path),selected)
                    for reader in (probe.read_witness_plan,probe.read_product_plan,probe.read_sign_plan,probe.read_magnitude_plan,probe.read_scale_plan):
                        with self.assertRaisesRegex(ValueError,'header'):reader(path)
                for old,columns in ((probe.default_cases(),probe.COLUMNS),(probe.control_cases(),probe.CONTROL_COLUMNS),
                    (probe.prefix_cases(),probe.PREFIX_COLUMNS),(probe.witness_cases(),probe.WITNESS_COLUMNS),
                    (probe.product_cases(),probe.PRODUCT_COLUMNS),(probe.sign_cases(),probe.SIGN_COLUMNS),
                    (probe.magnitude_cases(),probe.MAGNITUDE_COLUMNS),(probe.scale_cases(),probe.SCALE_COLUMNS)):
                    self.write_plan(path,old[:1],columns)
                    with self.assertRaisesRegex(ValueError,'header'):probe.read_reciprocal_plan(path)

    def test_all_words_exact_invariant_products_and_rational_metadata(self):
        for case in probe.reciprocal_cases():
            q,ae,be,r=case['q'],case['a_scale_exp'],case['b_scale_exp'],case['role']
            a,b=operands(q,ae,be,r);actual=probe.reciprocal_packed_inputs(case)
            self.assertEqual(actual,(half_words(a),half_words(b)))
            products=[x*y for x,y in zip(a,b)]
            want=[Fraction(q,256) if r!='negative' else Fraction(0),Fraction(-(q+1),256) if r!='positive' else Fraction(0)]
            self.assertEqual(products,want)
            self.assertEqual(probe.reciprocal_reference_output(case),reference(case));self.assertEqual(Fraction(reference(case)[0]),sum(want))
            meta=probe.reciprocal_case_fields(case)
            self.assertEqual([Fraction(x,meta['a_denominator']) for x in meta['a_numerators']],a)
            self.assertEqual([Fraction(x,meta['b_denominator']) for x in meta['b_numerators']],b)
            self.assertEqual([Fraction(x,meta['product_denominator']) for x in meta['product_numerators']],want)
            self.assertEqual(Fraction(meta['target_reference_numerator'],meta['target_reference_denominator']),sum(want))
            for op,e in (('a',ae),('b',be)):
                self.assertEqual(Fraction(meta[op+'_scale_numerator'],meta[op+'_scale_denominator']),Fraction(2)**e)
                self.assertEqual(meta[op+'_denominator'],16*meta[op+'_scale_denominator'])
            self.assertEqual(meta['product_denominator'],meta['a_denominator']*meta['b_denominator'])
            for data,values in zip(actual,(a,b)):
                self.assertEqual([i for i,w in enumerate(data) if w],[i for i,x in enumerate(values) if x])
                self.assertEqual([Fraction(v) for v in struct.unpack('<1024e',struct.pack('<1024H',*data))[:2]],values)
                self.assertEqual(data[2:],[0]*1022)
            self.assertEqual(probe.validate_reciprocal_inputs(case,*actual)['input_halfwords_checked'],2048)
            self.assertNotIn('scale_exp',meta);self.assertNotIn('matching_magnitude_pattern',meta)
        largest=next(c for c in probe.reciprocal_cases() if c['pattern']=='q12-am2-bp2-pair')
        self.assertEqual(operands(12,-2,2,'pair')[1][1],Fraction(-13,4))
        self.assertEqual(probe.reciprocal_reference_output(largest)[0],-1/256)

    def test_actual_full_buffer_uniqueness_and_plural_prior_matches(self):
        old={c['pattern']:tuple(tuple(x) for x in probe.scale_packed_inputs(c)) for c in probe.scale_cases()}
        cases=probe.reciprocal_cases();new={c['pattern']:tuple(tuple(x) for x in probe.reciprocal_packed_inputs(c)) for c in cases}
        self.assertEqual(len(new),45);self.assertEqual(len(set(new.values())),45)
        matching={}
        for case in cases:
            matches=[name for name,data in old.items() if data==new[case['pattern']]]
            meta=probe.reciprocal_case_fields(case);self.assertIn('matching_scale_patterns',meta);self.assertEqual(meta['matching_scale_patterns'],matches)
            if matches:matching[case['pattern']]=matches
        self.assertEqual(len(matching),9);self.assertEqual(sum(len(x) for x in matching.values()),11)
        self.assertEqual(matching['q06-a0-b0-positive'],['q06-e0-positive','q12-em1-positive'])
        self.assertEqual(matching['q12-a0-b0-positive'],['q06-ep1-positive','q12-e0-positive'])
        for name,matches in matching.items():
            if name not in ('q06-a0-b0-positive','q12-a0-b0-positive'):self.assertEqual(len(matches),1)
        for row in probe.scale_pattern_contracts():self.assertNotIn('matching_scale_patterns',row)

    def test_closed_exponents_reciprocal_binding_and46_row_limit(self):
        case=probe.reciprocal_cases()[17]  # q7, A=-2, B=+2, pair
        for fields in ({'a_scale_exp':-3},{'b_scale_exp':3},{'a_scale_exp':2},{'b_scale_exp':-2},
                       {'a_scale_exp':-1,'b_scale_exp':-1},{'a_scale_exp':-2.0},{'b_scale_exp':True},
                       {'a_scale_exp':'-1.5'},{'b_scale_exp':'true'},{'q':8},{'q':False},{'role':'positive'},
                       {'role':'other'},{'suite':'scale-control'},{'input_rule':'isolated-a-power-of-two-scale'},
                       {'target_row':1},{'target_col':1},{'m':1},{'n':1},{'k':1},{'order':'unknown'},
                       {'warmups':0},{'samples':9},{'launches':11}):
            with self.subTest(fields=fields),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'cases.tsv';self.write_plan(path,[dict(case,**fields)])
                with self.assertRaises(ValueError):probe.read_reciprocal_plan(path)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cases.tsv';self.write_plan(path,probe.reciprocal_cases()+[dict(case,id='forty_sixth')])
            with self.assertRaisesRegex(ValueError,'1 to 45'):probe.read_reciprocal_plan(path)
            self.write_plan(path,[case,dict(case,id='duplicate')])
            with self.assertRaisesRegex(ValueError,'Duplicate reciprocal pattern'):probe.read_reciprocal_plan(path)

    def test_wrong_one_sided_same_sign_and_product_preserving_input_changes(self):
        for case in (probe.reciprocal_cases()[17],probe.reciprocal_cases()[26]):
            for mutation in ('unscaled_b','unscaled_a','same_sign','opposite_assignment','sign_transfer','swap_slots','other_row','other_col','negative_zero'):
                a,b=probe.reciprocal_packed_inputs(case);meta=probe.reciprocal_case_fields(case)
                if mutation=='unscaled_b':b[:2]=[probe.halfword(x) for x in meta['b_base_numerators']]
                elif mutation=='unscaled_a':a[:2]=[probe.halfword(x) for x in meta['a_base_numerators']]
                elif mutation=='same_sign':b[:2]=[probe.scaled_halfword(x,case['a_scale_exp']) for x in meta['b_base_numerators']]
                elif mutation=='opposite_assignment':
                    a[:2]=[probe.scaled_halfword(x,case['b_scale_exp']) for x in meta['a_base_numerators']]
                    b[:2]=[probe.scaled_halfword(x,case['a_scale_exp']) for x in meta['b_base_numerators']]
                    decode=lambda w:Fraction(struct.unpack('<e',struct.pack('<H',w))[0])
                    self.assertEqual(sum(decode(x)*decode(y) for x,y in zip(a[:2],b[:2])),-Fraction(1,256))
                elif mutation=='sign_transfer':a[0]^=0x8000;b[0]^=0x8000
                elif mutation=='swap_slots':a[:2]=a[:2][::-1];b[:2]=b[:2][::-1]
                elif mutation=='other_row':a[16]=probe.halfword(1)
                elif mutation=='other_col':b[16]=probe.halfword(1)
                else:b[2]=0x8000
                with self.subTest(pattern=case['pattern'],mutation=mutation),self.assertRaises(ValueError):probe.validate_reciprocal_inputs(case,a,b)

    def fixture(self,root,cases):
        inputs,outputs=root/'input',root/'output';probe.prepare(inputs,SUITE,cases[0]['order']);outputs.mkdir();self.write_plan(inputs/'cases.tsv',cases)
        records=[dict(type='device',name='MetaX C550',visible_device_count=1,wave_size_api=64,pci_bus_id='0000:01:00.0',runtime_version_api=1,driver_version_api=1),
                 dict(type='protocol',schema_version=1,experiment=probe.RECIPROCAL_EXPERIMENT,operand_dtype='float16',accumulator_dtype='float32',output_elements=256,guard_elements_each_side=64,required_wave_size=64,timer='mcEventElapsedTime',comparison='finite exact numeric equality; signed zero equivalent; no tolerance',control_mode=1,variants_per_case=2,input_snapshots=list(probe.SNAPSHOT_PHASES),input_rewrite_between_variants=False,purpose='correctness_diagnostic',performance_accepted=False,**probe.reciprocal_protocol_metadata(),**{k:None for k in probe.ENVIRONMENT})]
        for case in cases:
            a,b=probe.reciprocal_packed_inputs(case)
            records.append(dict(type='logical_case',id=case['id'],**{k:case[k] for k in ('m','n','k','order')},**probe.reciprocal_case_fields(case)))
            def snapshot(phase):
                meta=probe.snapshot_metadata(case,phase,suite=SUITE);records.append(dict(type='input_snapshot',id=case['id'],**meta))
                for op,data in (('a',a),('b',b)):(outputs/meta[op+'_file']).write_bytes(struct.pack('<1024H',*data))
            snapshot('before')
            for i,variant in enumerate(probe.ORDERS[case['order']]):
                if i:snapshot('between')
                meta=probe.case_metadata(case,variant,suite=SUITE)
                records.append(dict(type='case',id=case['id'],**meta,function_attributes_before_timing=dict(maxThreadsPerBlock=512,numRegs=16,sharedSizeBytes=0,localSizeBytes=0),pointer_alignment_observed_bytes=dict(a=512,b=256,c_payload=256)))
                records.extend(dict(type='sample',id=case['id'],variant=variant,sample=j,event_batch_ms=0.1,host_enqueue_batch_us=50.0) for j in range(10))
                (outputs/meta['output_file']).write_bytes(struct.pack('<384I',*output_words(reference(case))))
            snapshot('after')
        records.append(dict(type='complete',cases=len(cases),variant_executions=2*len(cases),cpu_correctness_checked=False))
        self.save(outputs,records);return inputs,outputs,records

    def save(self,outputs,records):(outputs/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))

    def test_complete45_both_orders_and_exact_failures(self):
        for order in probe.ORDERS:
            cases=probe.reciprocal_cases(order)
            if order=='scalar-first':cases.reverse()
            with self.subTest(order=order),tempfile.TemporaryDirectory() as directory:
                inputs,outputs,_=self.fixture(Path(directory),cases);checked=probe.check(inputs,outputs)
                self.assertEqual(checked['status'],'pass');self.assertIs(checked['performance_accepted'],False)
                self.assertEqual((checked['logical_cases_checked'],checked['variant_outputs_checked'],checked['payload_elements_checked'],checked['guard_elements_checked']),(45,90,23040,11520))
                self.assertEqual((checked['prepared_input_halfwords_checked'],checked['input_snapshot_halfwords_checked']),(92160,276480))
                for index,variant in ((0,'wmma'),(1,'scalar')):
                    path=outputs/(cases[index]['id']+'.'+variant+'.f32');data=probe.read_words(path,384,4);data[64]^=1;path.write_bytes(struct.pack('<384I',*data))
                checked=probe.check(inputs,outputs);self.assertEqual(checked['status'],'diagnostic_failed')
                self.assertFalse(checked['cases'][0]['variants']['wmma']['exact_passed']);self.assertTrue(checked['cases'][0]['variants']['scalar']['exact_passed'])
                self.assertFalse(checked['cases'][1]['variants']['scalar']['exact_passed']);self.assertTrue(all(c['passed'] for c in checked['cases'][2:]))
                cli=subprocess.run(['python3',str(MODULE),'check',str(inputs),str(outputs)],capture_output=True,text=True)
                self.assertEqual(cli.returncode,1);self.assertEqual(json.loads(cli.stdout)['variant_outputs_checked'],90)

    def test_deep_exponent_rational_and_required_plural_list_metadata(self):
        for case in (probe.reciprocal_cases()[2],probe.reciprocal_cases()[6]):
            with tempfile.TemporaryDirectory() as directory:
                inputs,outputs,records=self.fixture(Path(directory),[case]);self.assertTrue(probe.check(inputs,outputs)['passed'])
                for kind in ('logical_case','case','input_snapshot'):
                    for field in probe.reciprocal_case_fields(case):
                        changed=copy.deepcopy(records);next(r for r in changed if r['type']==kind).pop(field)
                        with self.subTest(kind=kind,missing=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
                    for field,value in (('a_scale_exp',float(case['a_scale_exp'])),('b_scale_exp',False),('b_scale_numerator',True),('a_scale_denominator',1.0),('b_base_numerators',[-1,True]),('b_numerators',[-1,1.0]),('product_denominator',256.0),('matching_scale_patterns',None),('matching_scale_patterns',['q06-e0-positive'])):
                        changed=copy.deepcopy(records);next(r for r in changed if r['type']==kind)[field]=value
                        with self.subTest(kind=kind,field=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
                for field in probe.reciprocal_protocol_metadata():
                    changed=copy.deepcopy(records);changed[1].pop(field)
                    with self.subTest(protocol_missing=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
                for field,value in (('b_scale_exp',2.0),('b_scale_numerator',4.0),('matching_scale_patterns',None)):
                    changed=copy.deepcopy(records);changed[1]['reciprocal_patterns'][0][field]=value
                    with self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
                oracle=probe.control_oracle_metadata(suite=SUITE);oracle['reciprocal_patterns'][0].pop('matching_scale_patterns')
                (inputs/'oracle.json').write_text(json.dumps(oracle))
                with self.assertRaises(ValueError):probe.check(inputs,outputs)

    def test_snapshots_full_outputs_guards_and_signed_zero(self):
        case=probe.reciprocal_cases()[-1]
        for phase in probe.SNAPSHOT_PHASES:
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as directory:
                inputs,outputs,_=self.fixture(Path(directory),[case]);path=outputs/(case['id']+'.'+phase+'.b.f16')
                data=probe.read_words(path,1024,2);data[16]=probe.halfword(1);path.write_bytes(struct.pack('<1024H',*data))
                checked=probe.check(inputs,outputs);self.assertFalse(checked['passed']);self.assertTrue(checked['structural_valid'])
                row=checked['cases'][0]['input_snapshots'][phase];self.assertEqual((row['a_scale_exp'],row['b_scale_exp']),(2,-2))
                self.assertFalse(row['operands']['b']['equal_to_prepared']);self.assertFalse(row['operands']['b']['equal_to_fixed_packing'])
                self.assertTrue(all(v['exact_passed'] for v in checked['cases'][0]['variants'].values()))
                path.unlink()
                with self.assertRaises(OSError):probe.check(inputs,outputs)
        correct=output_words(reference(case))
        for index,value in ((64,0),(319,correct[64]),(65,0xffffffff),(64,0x7f800000),(0,0),(383,0)):
            actual=correct.copy();actual[index]=value;checked=probe.analyze_control_output(case,actual,suite=SUITE)
            self.assertFalse(checked['exact_passed'] and checked['guards_intact'])
        correct[65:320]=[0x80000000]*255;self.assertTrue(probe.analyze_control_output(case,correct,suite=SUITE)['exact_passed'])
        with self.assertRaises(ValueError):probe.analyze_control_output(case,correct[:-1],suite=SUITE)

    def test_mode6_is_explicit_and_seven_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stub=root/'mxcc-stub';stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n');stub.chmod(0o700)
            base=dict(os.environ,MXCC=str(stub),C550_ARCH='xcore1000')
            for control,prefix,mode,accepted in ((1,0,6,True),(0,0,6,False),(1,1,6,False),(1,0,7,False),(1,0,-1,False)):
                env=dict(base,C550_WMMA_CONTROL=str(control),C550_WMMA_PREFIX=str(prefix),C550_WMMA_WITNESS=str(mode))
                result=subprocess.run(['bash',str(MODULE.with_name('compile.sh')),str(root/'probe')],env=env,text=True,capture_output=True)
                with self.subTest(control=control,prefix=prefix,mode=mode):
                    if accepted:self.assertEqual(result.returncode,0,result.stderr);self.assertIn('-DC550_WMMA_WITNESS=6',result.stdout.splitlines())
                    else:self.assertEqual((result.returncode,result.stdout),(2,''))


if __name__=='__main__':unittest.main()
