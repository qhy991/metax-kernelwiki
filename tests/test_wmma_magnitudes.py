"""Bounded magnitude host checks; synthetic outputs provide no device evidence."""
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

MODULE=Path(__file__).resolve().parents[1]/'experiments/wmma/experiment.py'
SPEC=importlib.util.spec_from_file_location('wmma_magnitudes',MODULE)
probe=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(probe)
SUITE='magnitude-control'
BASELINES={'positive':'positive12-k0','negative':'negative13-k1','pair':'pair-pn'}


def logical_terms(q,role):
    if role=='positive':return [-q,0],[-1,0],q
    if role=='negative':return [0,1],[0,-(q+1)],-(q+1)
    return [-q,1],[-1,-(q+1)],-1


def output_words(values):
    return [0xffffffff]*64+list(struct.unpack('<256I',struct.pack('<256f',*values)))+[0xffffffff]*64


class WmmaMagnitudesTest(unittest.TestCase):
    def write_plan(self,path,cases,columns=probe.MAGNITUDE_COLUMNS):
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=columns,delimiter='\t',lineterminator='\n')
            writer.writeheader();writer.writerows(cases)

    def test_closed42_cases_explicit_q_role_and_old_header_refusal(self):
        expected=[(q,role) for q in range(1,15) for role in ('positive','negative','pair')]
        for order in probe.ORDERS:
            cases=probe.magnitude_cases(order)
            self.assertEqual([(c['q'],c['role']) for c in cases],expected)
            self.assertEqual([c['pattern'] for c in cases],[f'q{q:02d}-{role}' for q,role in expected])
            self.assertEqual([c['id'] for c in cases],[f'magnitude_q{q:02d}_{role}' for q,role in expected])
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'cases.tsv'
                for selected in (cases,list(reversed(cases)),cases[35:36]):
                    self.write_plan(path,selected);self.assertEqual(probe.read_magnitude_plan(path),selected)
                    for reader in (probe.read_witness_plan,probe.read_product_plan,probe.read_sign_plan):
                        with self.assertRaisesRegex(ValueError,'header'):reader(path)
                old_suites=((probe.default_cases(),probe.COLUMNS),(probe.control_cases(),probe.CONTROL_COLUMNS),
                            (probe.prefix_cases(),probe.PREFIX_COLUMNS),(probe.witness_cases(),probe.WITNESS_COLUMNS),
                            (probe.product_cases(),probe.PRODUCT_COLUMNS),(probe.sign_cases(),probe.SIGN_COLUMNS))
                for old,columns in old_suites:
                    self.write_plan(path,old[:1],columns)
                    with self.assertRaisesRegex(ValueError,'header'):probe.read_magnitude_plan(path)
        self.assertEqual([len(probe.default_cases()),len(probe.control_cases()),len(probe.prefix_cases()),len(probe.witness_cases()),len(probe.product_cases()),len(probe.sign_cases())],[12,12,34,3,6,8])

    def test_range_role_pattern_binding_and43rd_row_refusals(self):
        base=probe.magnitude_cases()[35]
        for fields in ({'q':0,'pattern':'q00-pair'},{'q':15,'pattern':'q15-pair'},
                       {'q':11},{'q':True},{'q':12.0},{'role':'other'},{'role':'positive'},
                       {'pattern':'q12-positive'},{'suite':'sign-control'},{'input_rule':'isolated-signed-products'},
                       {'m':1},{'n':1},{'k':1},{'target_row':1},{'target_col':1},{'order':'unknown'},
                       {'warmups':9},{'samples':9},{'launches':9}):
            with self.subTest(fields=fields),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'cases.tsv';self.write_plan(path,[dict(base,**fields)])
                with self.assertRaises(ValueError):probe.read_magnitude_plan(path)
        for q in (False,12.0):
            with self.assertRaises(ValueError):probe.validate_magnitude_case(dict(base,q=q))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cases.tsv';self.write_plan(path,probe.magnitude_cases()+[dict(base,id='forty_third')])
            with self.assertRaisesRegex(ValueError,'1 to 42'):probe.read_magnitude_plan(path)
            self.write_plan(path,[base,dict(base,id='duplicate')])
            with self.assertRaisesRegex(ValueError,'Duplicate magnitude pattern'):probe.read_magnitude_plan(path)
        with self.assertRaises(ValueError):probe.control_oracle_metadata(prefix=True,suite=SUITE)

    def test_all_words_exact_integer_oracle_and_only_q12_baselines(self):
        previous={c['pattern']:c for c in probe.sign_cases()};matching=0
        for case in probe.magnitude_cases():
            q,role=case['q'],case['role'];left,right,numerator=logical_terms(q,role)
            a,b=probe.magnitude_packed_inputs(case)
            expected=lambda values:list(struct.unpack('<2H',struct.pack('<2e',*(x/16 for x in values))))+[0]*1022
            self.assertEqual(a,expected(left));self.assertEqual(b,expected(right))
            self.assertLessEqual(max(abs(v) for v in left+right),15)
            self.assertEqual(sum(x*y for x,y in zip(left,right)),numerator)
            self.assertEqual(probe.magnitude_reference_output(case),[numerator/256]+[0.0]*255)
            self.assertEqual(probe.validate_magnitude_inputs(case,a,b)['input_halfwords_checked'],2048)
            meta=probe.magnitude_case_fields(case);products=[x*y for x,y in zip(left,right)]
            self.assertEqual((meta['q'],meta['role']),(q,role));self.assertEqual(meta['product_numerators'],products)
            self.assertEqual(meta['product_signs'],[(v>0)-(v<0) for v in products]);self.assertEqual(meta['product_magnitudes'],[abs(v) for v in products])
            self.assertEqual(meta['nonzero_product_k_slots'],[k for k,v in enumerate(products) if v])
            if role!='pair':
                inactive=1 if role=='positive' else 0;self.assertEqual((a[inactive],b[inactive]),(0,0))
            if q==12:
                self.assertEqual(meta['matching_sign_pattern'],BASELINES[role])
                self.assertEqual((a,b),probe.sign_packed_inputs(previous[BASELINES[role]]));matching+=1
            else:self.assertIn('matching_sign_pattern',meta);self.assertIsNone(meta['matching_sign_pattern'])
        self.assertEqual(matching,3)
        self.assertTrue(all('matching_sign_pattern' not in c and 'q' not in c for c in probe.sign_pattern_contracts()))

    def test_factorization_and_inactive_operands_remain_exact(self):
        for case in probe.magnitude_cases():
            source_a,source_b=probe.magnitude_packed_inputs(case)
            active=0 if case['role']!='negative' else 1
            for mutation in ('sign_transfer','reversed_terms','other_row','other_col','negative_zero','unused_chunk'):
                a,b=source_a.copy(),source_b.copy()
                if mutation=='sign_transfer':
                    a[active]^=0x8000;b[active]^=0x8000
                    decode=lambda w:struct.unpack('<e',struct.pack('<H',w))[0]
                    self.assertEqual(decode(a[active])*decode(b[active]),decode(source_a[active])*decode(source_b[active]))
                elif mutation=='reversed_terms':a[:2]=a[:2][::-1];b[:2]=b[:2][::-1]
                elif mutation=='other_row':a[16]=probe.halfword(1)
                elif mutation=='other_col':b[16]=probe.halfword(1)
                elif mutation=='negative_zero':a[2]=0x8000
                else:b[256]=probe.halfword(1)
                with self.subTest(q=case['q'],role=case['role'],mutation=mutation),self.assertRaises(ValueError):probe.validate_magnitude_inputs(case,a,b)
            if case['role']!='pair':
                a,b=source_a.copy(),source_b.copy();inactive=1-active;b[inactive]=probe.halfword(-1)
                self.assertEqual(a[inactive],0)
                with self.assertRaises(ValueError):probe.validate_magnitude_inputs(case,a,b)

    def fixture(self,root,cases):
        inputs,outputs=root/'input',root/'output';probe.prepare(inputs,SUITE,cases[0]['order']);outputs.mkdir();self.write_plan(inputs/'cases.tsv',cases)
        records=[dict(type='device',name='MetaX C550',visible_device_count=1,wave_size_api=64,pci_bus_id='0000:01:00.0',runtime_version_api=1,driver_version_api=1),
                 dict(type='protocol',schema_version=1,experiment=probe.MAGNITUDE_EXPERIMENT,operand_dtype='float16',accumulator_dtype='float32',output_elements=256,guard_elements_each_side=64,required_wave_size=64,timer='mcEventElapsedTime',comparison='finite exact numeric equality; signed zero equivalent; no tolerance',control_mode=1,variants_per_case=2,input_snapshots=list(probe.SNAPSHOT_PHASES),input_rewrite_between_variants=False,purpose='correctness_diagnostic',performance_accepted=False,**probe.magnitude_protocol_metadata(),**{k:None for k in probe.ENVIRONMENT})]
        for case in cases:
            a,b=probe.magnitude_packed_inputs(case)
            records.append(dict(type='logical_case',id=case['id'],**{k:case[k] for k in ('m','n','k','order')},**probe.magnitude_case_fields(case)))
            def snapshot(phase):
                meta=probe.snapshot_metadata(case,phase,suite=SUITE);records.append(dict(type='input_snapshot',id=case['id'],**meta))
                for op,data in (('a',a),('b',b)):(outputs/meta[op+'_file']).write_bytes(struct.pack('<1024H',*data))
            snapshot('before')
            for i,variant in enumerate(probe.ORDERS[case['order']]):
                if i:snapshot('between')
                meta=probe.case_metadata(case,variant,suite=SUITE)
                records.append(dict(type='case',id=case['id'],**meta,function_attributes_before_timing=dict(maxThreadsPerBlock=512,numRegs=16,sharedSizeBytes=0,localSizeBytes=0),pointer_alignment_observed_bytes=dict(a=512,b=256,c_payload=256)))
                records.extend(dict(type='sample',id=case['id'],variant=variant,sample=j,event_batch_ms=0.1,host_enqueue_batch_us=50.0) for j in range(10))
                (outputs/meta['output_file']).write_bytes(struct.pack('<384I',*output_words(probe.magnitude_reference_output(case))))
            snapshot('after')
        records.append(dict(type='complete',cases=len(cases),variant_executions=2*len(cases),cpu_correctness_checked=False))
        self.save(outputs,records);return inputs,outputs,records

    def save(self,outputs,records):(outputs/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))

    def test_complete42_cases_both_orders_and_mixed_failures(self):
        for order in probe.ORDERS:
            cases=probe.magnitude_cases(order)
            if order=='scalar-first':cases.reverse()
            with self.subTest(order=order),tempfile.TemporaryDirectory() as directory:
                inputs,outputs,_=self.fixture(Path(directory),cases);checked=probe.check(inputs,outputs)
                self.assertEqual(checked['status'],'pass');self.assertIs(checked['performance_accepted'],False)
                self.assertEqual((checked['logical_cases_checked'],checked['variant_outputs_checked'],checked['payload_elements_checked'],checked['guard_elements_checked']),(42,84,21504,10752))
                self.assertEqual((checked['prepared_input_halfwords_checked'],checked['input_snapshot_halfwords_checked']),(86016,258048))
                for index,variant in ((0,'wmma'),(1,'scalar')):
                    path=outputs/(cases[index]['id']+'.'+variant+'.f32');data=probe.read_words(path,384,4);data[64]^=1;path.write_bytes(struct.pack('<384I',*data))
                checked=probe.check(inputs,outputs);self.assertEqual(checked['status'],'diagnostic_failed')
                self.assertFalse(checked['cases'][0]['variants']['wmma']['exact_passed']);self.assertTrue(checked['cases'][0]['variants']['scalar']['exact_passed'])
                self.assertFalse(checked['cases'][1]['variants']['scalar']['exact_passed']);self.assertTrue(all(c['passed'] for c in checked['cases'][2:]))
                cli=subprocess.run(['python3',str(MODULE),'check',str(inputs),str(outputs)],capture_output=True,text=True)
                self.assertEqual(cli.returncode,1);self.assertEqual(json.loads(cli.stdout)['variant_outputs_checked'],84)

    def test_required_q_role_deep_types_and_explicit_null_mappings(self):
        for case in (probe.magnitude_cases()[2],probe.magnitude_cases()[35]):
            with tempfile.TemporaryDirectory() as directory:
                inputs,outputs,records=self.fixture(Path(directory),[case]);self.assertTrue(probe.check(inputs,outputs)['passed'])
                for kind in ('logical_case','case','input_snapshot'):
                    for field in probe.magnitude_case_fields(case):
                        changed=copy.deepcopy(records);next(r for r in changed if r['type']==kind).pop(field)
                        with self.subTest(kind=kind,missing=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
                    for field,value in (('q',True),('q',float(case['q'])),('role','positive'),('product_signs',[True,-1]),('product_magnitudes',[case['q'],float(case['q']+1)]),('k_slots',[False,1]),('a_numerators',[-case['q'],True]),('matching_sign_pattern','pair-pp')):
                        changed=copy.deepcopy(records);next(r for r in changed if r['type']==kind)[field]=value
                        with self.subTest(kind=kind,field=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
                for field in probe.magnitude_protocol_metadata():
                    changed=copy.deepcopy(records);changed[1].pop(field)
                    with self.subTest(protocol_missing=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
                for field,value in (('q',True),('product_magnitudes',[1.0,0]),('matching_sign_pattern','positive12-k0')):
                    changed=copy.deepcopy(records);changed[1]['magnitude_patterns'][0][field]=value
                    with self.subTest(protocol_nested=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
                changed=copy.deepcopy(records);changed[1]['magnitude_patterns'][0].pop('matching_sign_pattern')
                with self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
                oracle=probe.control_oracle_metadata(suite=SUITE);oracle['magnitude_patterns'][0].pop('matching_sign_pattern')
                (inputs/'oracle.json').write_text(json.dumps(oracle))
                with self.assertRaises(ValueError):probe.check(inputs,outputs)

    def test_snapshot_corruption_and_missing_files_are_not_numerical_passes(self):
        case=probe.magnitude_cases()[-1]
        for phase in probe.SNAPSHOT_PHASES:
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as directory:
                inputs,outputs,_=self.fixture(Path(directory),[case]);path=outputs/(case['id']+'.'+phase+'.a.f16')
                data=probe.read_words(path,1024,2);data[16]=probe.halfword(1);path.write_bytes(struct.pack('<1024H',*data))
                checked=probe.check(inputs,outputs);self.assertFalse(checked['passed']);self.assertTrue(checked['structural_valid'])
                row=checked['cases'][0]['input_snapshots'][phase];self.assertEqual((row['q'],row['role']),(14,'pair'));self.assertIsNone(row['matching_sign_pattern'])
                self.assertFalse(row['operands']['a']['equal_to_prepared']);self.assertFalse(row['operands']['a']['equal_to_fixed_packing'])
                self.assertTrue(all(v['exact_passed'] for v in checked['cases'][0]['variants'].values()))
                path.unlink()
                with self.assertRaises(OSError):probe.check(inputs,outputs)

    def test_full_outputs_guard_extent_and_signed_zero(self):
        case=probe.magnitude_cases()[-1];correct=output_words(probe.magnitude_reference_output(case))
        for index,value in ((64,0),(65,correct[64]),(319,0xffffffff),(64,0x7f800000),(0,0),(383,0)):
            actual=correct.copy();actual[index]=value;checked=probe.analyze_control_output(case,actual,suite=SUITE)
            with self.subTest(index=index,value=value):self.assertFalse(checked['exact_passed'] and checked['guards_intact'])
        correct[65:320]=[0x80000000]*255;self.assertTrue(probe.analyze_control_output(case,correct,suite=SUITE)['exact_passed'])
        with self.assertRaises(ValueError):probe.analyze_control_output(case,correct[:-1],suite=SUITE)

    def test_mode4_is_explicit_and_current_upper_bound_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stub=root/'mxcc-stub';stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n');stub.chmod(0o700)
            base=dict(os.environ,MXCC=str(stub),C550_ARCH='xcore1000')
            for control,prefix,mode,accepted in ((1,0,4,True),(0,0,4,False),(1,1,4,False),(1,0,7,False),(1,0,-1,False)):
                env=dict(base,C550_WMMA_CONTROL=str(control),C550_WMMA_PREFIX=str(prefix),C550_WMMA_WITNESS=str(mode))
                result=subprocess.run(['bash',str(MODULE.with_name('compile.sh')),str(root/'probe')],env=env,text=True,capture_output=True)
                with self.subTest(control=control,prefix=prefix,mode=mode):
                    if accepted:self.assertEqual(result.returncode,0,result.stderr);self.assertIn('-DC550_WMMA_WITNESS=4',result.stdout.splitlines())
                    else:self.assertEqual((result.returncode,result.stdout),(2,''))


if __name__=='__main__':unittest.main()
