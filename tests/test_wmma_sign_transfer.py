"""Exact product-preserving sign-transfer CPU tests; no device-result fixtures."""
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
SPEC=importlib.util.spec_from_file_location('wmma_sign_transfer',MODULE)
probe=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(probe)
SUITE='sign-transfer-control'
FLIPS={'positive':((0,0),(1,0)),'negative':((0,0),(0,1)),'pair':((0,0),(1,0),(0,1),(1,1))}


def independent_operands(q,role,f0,f1):
    a=[Fraction(-q,16),Fraction(1,16)];b=[Fraction(-1,16),Fraction(-(q+1),16)]
    if role=='positive':a[1]=b[1]=Fraction(0)
    if role=='negative':a[0]=b[0]=Fraction(0)
    return ([x*(-1 if f else 1) for x,f in zip(a,(f0,f1))],
            [x*(-1 if f else 1) for x,f in zip(b,(f0,f1))])


def half_words(values):
    return list(struct.unpack('<2H',struct.pack('<2e',*(float(x) for x in values))))+[0]*1022


def reference(case):
    numerator=case['q'] if case['role']=='positive' else -(case['q']+1) if case['role']=='negative' else -1
    return [numerator/256]+[0.0]*255


def output_words(values):
    return [0xffffffff]*64+list(struct.unpack('<256I',struct.pack('<256f',*values)))+[0xffffffff]*64


class WmmaSignTransferTest(unittest.TestCase):
    def write_plan(self,path,cases,columns=probe.SIGN_TRANSFER_COLUMNS):
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=columns,delimiter='\t',lineterminator='\n')
            writer.writeheader();writer.writerows(cases)

    def test_closed24_conditions_orders_subsets_and_old_header_refusals(self):
        expected=[(q,r,f0,f1) for q in (6,7,12) for r in ('positive','negative','pair') for f0,f1 in FLIPS[r]]
        for order in probe.ORDERS:
            cases=probe.sign_transfer_cases(order)
            self.assertEqual([(c['q'],c['role'],c['flip_k0'],c['flip_k1']) for c in cases],expected)
            self.assertEqual([c['pattern'] for c in cases],[f'q{q:02d}-{r}-f{f0}{f1}' for q,r,f0,f1 in expected])
            self.assertEqual(cases[0]['id'],'sign_transfer_q06_positive_f00');self.assertEqual(cases[-1]['id'],'sign_transfer_q12_pair_f11')
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'cases.tsv'
                for selected in (cases,list(reversed(cases)),cases[:1]):
                    self.write_plan(path,selected);self.assertEqual(probe.read_sign_transfer_plan(path),selected)
                    for reader in (probe.read_witness_plan,probe.read_product_plan,probe.read_sign_plan,probe.read_magnitude_plan,probe.read_scale_plan,probe.read_reciprocal_plan):
                        with self.assertRaisesRegex(ValueError,'header'):reader(path)
                for old,columns in ((probe.default_cases(),probe.COLUMNS),(probe.control_cases(),probe.CONTROL_COLUMNS),
                    (probe.prefix_cases(),probe.PREFIX_COLUMNS),(probe.witness_cases(),probe.WITNESS_COLUMNS),
                    (probe.product_cases(),probe.PRODUCT_COLUMNS),(probe.sign_cases(),probe.SIGN_COLUMNS),
                    (probe.magnitude_cases(),probe.MAGNITUDE_COLUMNS),(probe.scale_cases(),probe.SCALE_COLUMNS),
                    (probe.reciprocal_cases(),probe.RECIPROCAL_COLUMNS)):
                    self.write_plan(path,old[:1],columns)
                    with self.assertRaisesRegex(ValueError,'header'):probe.read_sign_transfer_plan(path)

    def test_all24_exact_words_both_factor_signs_products_and_support(self):
        for c in probe.sign_transfer_cases():
            a,b=independent_operands(c['q'],c['role'],c['flip_k0'],c['flip_k1'])
            packed=probe.sign_transfer_packed_inputs(c);self.assertEqual(packed,(half_words(a),half_words(b)))
            original_a,original_b=independent_operands(c['q'],c['role'],0,0)
            self.assertEqual([x*y for x,y in zip(a,b)],[x*y for x,y in zip(original_a,original_b)])
            self.assertEqual(sum(x*y for x,y in zip(a,b)),Fraction(reference(c)[0]))
            self.assertEqual(probe.sign_transfer_reference_output(c),reference(c))
            meta=probe.sign_transfer_case_fields(c)
            self.assertEqual(meta['a_base_numerators'],[int(x*16) for x in original_a]);self.assertEqual(meta['b_base_numerators'],[int(x*16) for x in original_b])
            self.assertEqual(meta['a_numerators'],[int(x*16) for x in a]);self.assertEqual(meta['b_numerators'],[int(x*16) for x in b])
            self.assertEqual(meta['transfer_multipliers'],[-1 if c['flip_k0'] else 1,-1 if c['flip_k1'] else 1])
            self.assertEqual([Fraction(x,meta['product_denominator']) for x in meta['product_numerators']],[x*y for x,y in zip(a,b)])
            self.assertEqual(Fraction(meta['target_reference_numerator'],meta['target_reference_denominator']),Fraction(reference(c)[0]))
            for data,values in zip(packed,(a,b)):
                self.assertEqual([i for i,w in enumerate(data) if w],[i for i,x in enumerate(values) if x])
                self.assertEqual([Fraction(x) for x in struct.unpack('<1024e',struct.pack('<1024H',*data))[:2]],values)
                self.assertEqual(data[2:],[0]*1022)
            if c['role']=='positive':self.assertEqual((packed[0][1],packed[1][1],c['flip_k1']),(0,0,0))
            if c['role']=='negative':self.assertEqual((packed[0][0],packed[1][0],c['flip_k0']),(0,0,0))
            self.assertEqual(probe.validate_sign_transfer_inputs(c,*packed)['input_halfwords_checked'],2048)

    def test_complete_buffer_uniqueness_and_history_discovery(self):
        prior={c['pattern']:tuple(tuple(x) for x in probe.reciprocal_packed_inputs(c)) for c in probe.reciprocal_cases()}
        current={c['pattern']:tuple(tuple(x) for x in probe.sign_transfer_packed_inputs(c)) for c in probe.sign_transfer_cases()}
        self.assertEqual(len(current),24);self.assertEqual(len(set(current.values())),24)
        matches={}
        for c in probe.sign_transfer_cases():
            found=[name for name,words in prior.items() if words==current[c['pattern']]]
            meta=probe.sign_transfer_case_fields(c);self.assertIn('matching_reciprocal_patterns',meta);self.assertEqual(meta['matching_reciprocal_patterns'],found)
            if found:matches[c['pattern']]=found
        self.assertEqual(len(matches),9);self.assertEqual(sum(map(len,matches.values())),9)
        self.assertEqual(matches,{f'q{q:02d}-{r}-f00':[f'q{q:02d}-a0-b0-{r}'] for q in (6,7,12) for r in ('positive','negative','pair')})

    def test_inactive_flip_bounds_types_and25th_row_are_refused(self):
        pair=probe.sign_transfer_cases()[4]
        mutations=[dict(flip_k0=-1),dict(flip_k1=2),dict(flip_k0=True),dict(flip_k1=0.0),
                   dict(flip_k0='0.5'),dict(flip_k1='true'),dict(q=8),dict(q=False),dict(role='other'),
                   dict(pattern='q06-pair-f01'),dict(suite='reciprocal-control'),dict(input_rule='isolated-reciprocal-power-of-two-scale'),
                   dict(target_row=1),dict(target_col=1),dict(m=1),dict(n=1),dict(k=1),dict(order='unknown'),dict(warmups=0),dict(samples=9),dict(launches=11)]
        for fields in mutations:
            with self.subTest(fields=fields),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'cases.tsv';self.write_plan(path,[dict(pair,**fields)])
                with self.assertRaises(ValueError):probe.read_sign_transfer_plan(path)
        for role,flags in (('positive',(0,1)),('positive',(1,1)),('negative',(1,0)),('negative',(1,1))):
            bad=dict(pair,role=role,flip_k0=flags[0],flip_k1=flags[1],pattern=f'q06-{role}-f{flags[0]}{flags[1]}')
            with self.subTest(role=role,flags=flags),self.assertRaises(ValueError):probe.validate_sign_transfer_case(bad)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cases.tsv';self.write_plan(path,probe.sign_transfer_cases()+[dict(pair,id='twenty_fifth')])
            with self.assertRaisesRegex(ValueError,'1 to 24'):probe.read_sign_transfer_plan(path)
            self.write_plan(path,[pair,dict(pair,id='duplicate')])
            with self.assertRaisesRegex(ValueError,'Duplicate sign-transfer pattern'):probe.read_sign_transfer_plan(path)

    def test_one_sign_error_zero_padding_and_undeclared_transfer(self):
        for c in probe.sign_transfer_cases():
            actual_a,actual_b=probe.sign_transfer_packed_inputs(c);slot=0 if c['role']!='negative' else 1
            for mutation in ('a_only','b_only','both_undeclared','other_row','other_col','negative_zero','unused_chunk'):
                a,b=actual_a.copy(),actual_b.copy()
                if mutation=='a_only':a[slot]^=0x8000
                elif mutation=='b_only':b[slot]^=0x8000
                elif mutation=='both_undeclared':a[slot]^=0x8000;b[slot]^=0x8000
                elif mutation=='other_row':a[16]=probe.halfword(1)
                elif mutation=='other_col':b[16]=probe.halfword(1)
                elif mutation=='negative_zero':a[2]=0x8000
                else:b[256]=probe.halfword(1)
                decode=lambda w:Fraction(struct.unpack('<e',struct.pack('<H',w))[0])
                if mutation in ('a_only','b_only'):self.assertNotEqual(decode(a[slot])*decode(b[slot]),decode(actual_a[slot])*decode(actual_b[slot]))
                if mutation=='both_undeclared':self.assertEqual(decode(a[slot])*decode(b[slot]),decode(actual_a[slot])*decode(actual_b[slot]))
                with self.subTest(pattern=c['pattern'],mutation=mutation),self.assertRaises(ValueError):probe.validate_sign_transfer_inputs(c,a,b)
            if c['role']!='pair':
                a,b=actual_a.copy(),actual_b.copy();a[1-slot]=0x8000
                with self.assertRaises(ValueError):probe.validate_sign_transfer_inputs(c,a,b)

    def fixture(self,root,cases):
        inputs,outputs=root/'input',root/'output';probe.prepare(inputs,SUITE,cases[0]['order']);outputs.mkdir();self.write_plan(inputs/'cases.tsv',cases)
        records=[dict(type='device',name='MetaX C550',visible_device_count=1,wave_size_api=64,pci_bus_id='0000:01:00.0',runtime_version_api=1,driver_version_api=1),
                 dict(type='protocol',schema_version=1,experiment=probe.SIGN_TRANSFER_EXPERIMENT,operand_dtype='float16',accumulator_dtype='float32',output_elements=256,guard_elements_each_side=64,required_wave_size=64,timer='mcEventElapsedTime',comparison='finite exact numeric equality; signed zero equivalent; no tolerance',control_mode=1,variants_per_case=2,input_snapshots=list(probe.SNAPSHOT_PHASES),input_rewrite_between_variants=False,purpose='correctness_diagnostic',performance_accepted=False,**probe.sign_transfer_protocol_metadata(),**{k:None for k in probe.ENVIRONMENT})]
        for c in cases:
            a,b=probe.sign_transfer_packed_inputs(c)
            records.append(dict(type='logical_case',id=c['id'],**{k:c[k] for k in ('m','n','k','order')},**probe.sign_transfer_case_fields(c)))
            def snapshot(phase):
                meta=probe.snapshot_metadata(c,phase,suite=SUITE);records.append(dict(type='input_snapshot',id=c['id'],**meta))
                for op,data in (('a',a),('b',b)):(outputs/meta[op+'_file']).write_bytes(struct.pack('<1024H',*data))
            snapshot('before')
            for i,variant in enumerate(probe.ORDERS[c['order']]):
                if i:snapshot('between')
                meta=probe.case_metadata(c,variant,suite=SUITE)
                records.append(dict(type='case',id=c['id'],**meta,function_attributes_before_timing=dict(maxThreadsPerBlock=512,numRegs=16,sharedSizeBytes=0,localSizeBytes=0),pointer_alignment_observed_bytes=dict(a=512,b=256,c_payload=256)))
                records.extend(dict(type='sample',id=c['id'],variant=variant,sample=j,event_batch_ms=0.1,host_enqueue_batch_us=50.0) for j in range(10))
                (outputs/meta['output_file']).write_bytes(struct.pack('<384I',*output_words(reference(c))))
            snapshot('after')
        records.append(dict(type='complete',cases=len(cases),variant_executions=2*len(cases),cpu_correctness_checked=False))
        self.save(outputs,records);return inputs,outputs,records

    def save(self,outputs,records):(outputs/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))

    def test_complete24_both_orders_and_mixed_exact_failures(self):
        for order in probe.ORDERS:
            cases=probe.sign_transfer_cases(order)
            if order=='scalar-first':cases.reverse()
            with self.subTest(order=order),tempfile.TemporaryDirectory() as directory:
                inputs,outputs,_=self.fixture(Path(directory),cases);checked=probe.check(inputs,outputs)
                self.assertEqual(checked['status'],'pass');self.assertIs(checked['performance_accepted'],False)
                self.assertEqual((checked['logical_cases_checked'],checked['variant_outputs_checked'],checked['payload_elements_checked'],checked['guard_elements_checked']),(24,48,12288,6144))
                self.assertEqual((checked['prepared_input_halfwords_checked'],checked['input_snapshot_halfwords_checked']),(49152,147456))
                for index,variant in ((0,'wmma'),(1,'scalar')):
                    path=outputs/(cases[index]['id']+'.'+variant+'.f32');data=probe.read_words(path,384,4);data[64]^=1;path.write_bytes(struct.pack('<384I',*data))
                checked=probe.check(inputs,outputs);self.assertEqual(checked['status'],'diagnostic_failed')
                self.assertFalse(checked['cases'][0]['variants']['wmma']['exact_passed']);self.assertTrue(checked['cases'][0]['variants']['scalar']['exact_passed'])
                self.assertFalse(checked['cases'][1]['variants']['scalar']['exact_passed']);self.assertTrue(all(c['passed'] for c in checked['cases'][2:]))
                cli=subprocess.run(['python3',str(MODULE),'check',str(inputs),str(outputs)],capture_output=True,text=True)
                self.assertEqual(cli.returncode,1);self.assertEqual(json.loads(cli.stdout)['variant_outputs_checked'],48)

    def test_deep_flags_signs_and_required_empty_history_list(self):
        for c in (probe.sign_transfer_cases()[4],probe.sign_transfer_cases()[5]):
            with tempfile.TemporaryDirectory() as directory:
                inputs,outputs,records=self.fixture(Path(directory),[c]);self.assertTrue(probe.check(inputs,outputs)['passed'])
                for kind in ('logical_case','case','input_snapshot'):
                    for field in probe.sign_transfer_case_fields(c):
                        changed=copy.deepcopy(records);next(r for r in changed if r['type']==kind).pop(field)
                        with self.subTest(kind=kind,missing=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[c],suite=SUITE)
                    for field,value in (('flip_k0',bool(c['flip_k0'])),('flip_k1',0.0),('transfer_multipliers',[True,1]),
                                        ('a_base_numerators',[-6,True]),('a_numerators',[True,1]),('product_numerators',[6,-7.0]),
                                        ('matching_reciprocal_patterns',None),('matching_reciprocal_patterns',['wrong'])):
                        changed=copy.deepcopy(records);next(r for r in changed if r['type']==kind)[field]=value
                        with self.subTest(kind=kind,field=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[c],suite=SUITE)
                for field in probe.sign_transfer_protocol_metadata():
                    changed=copy.deepcopy(records);changed[1].pop(field)
                    with self.subTest(protocol_missing=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[c],suite=SUITE)
                changed=copy.deepcopy(records);changed[1]['allowed_role_flips']['positive'][1]=[True,0]
                with self.assertRaises(ValueError):probe.validate_control_records(changed,[c],suite=SUITE)
                changed=copy.deepcopy(records);changed[1]['sign_transfer_patterns'][0]['flip_k1']=False
                with self.assertRaises(ValueError):probe.validate_control_records(changed,[c],suite=SUITE)
                oracle=probe.control_oracle_metadata(suite=SUITE);oracle['sign_transfer_patterns'][1].pop('matching_reciprocal_patterns')
                (inputs/'oracle.json').write_text(json.dumps(oracle))
                with self.assertRaises(ValueError):probe.check(inputs,outputs)

    def test_snapshots_guards_full_outputs_and_signed_zero(self):
        c=probe.sign_transfer_cases()[-1]
        for phase in probe.SNAPSHOT_PHASES:
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as directory:
                inputs,outputs,_=self.fixture(Path(directory),[c]);path=outputs/(c['id']+'.'+phase+'.b.f16')
                data=probe.read_words(path,1024,2);data[16]=probe.halfword(1);path.write_bytes(struct.pack('<1024H',*data))
                checked=probe.check(inputs,outputs);self.assertFalse(checked['passed']);self.assertTrue(checked['structural_valid'])
                row=checked['cases'][0]['input_snapshots'][phase];self.assertEqual(row['transfer_multipliers'],[-1,-1])
                self.assertFalse(row['operands']['b']['equal_to_prepared']);self.assertFalse(row['operands']['b']['equal_to_fixed_packing'])
                self.assertTrue(all(v['exact_passed'] for v in checked['cases'][0]['variants'].values()))
                path.unlink()
                with self.assertRaises(OSError):probe.check(inputs,outputs)
        correct=output_words(reference(c))
        for index,value in ((64,0),(319,correct[64]),(65,0xffffffff),(64,0x7f800000),(0,0),(383,0)):
            actual=correct.copy();actual[index]=value;checked=probe.analyze_control_output(c,actual,suite=SUITE)
            self.assertFalse(checked['exact_passed'] and checked['guards_intact'])
        correct[65:320]=[0x80000000]*255;self.assertTrue(probe.analyze_control_output(c,correct,suite=SUITE)['exact_passed'])
        with self.assertRaises(ValueError):probe.analyze_control_output(c,correct[:-1],suite=SUITE)

    def test_mode7_is_explicit_and_eight_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stub=root/'mxcc-stub';stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n');stub.chmod(0o700)
            base=dict(os.environ,MXCC=str(stub),C550_ARCH='xcore1000')
            for control,prefix,mode,accepted in ((1,0,7,True),(0,0,7,False),(1,1,7,False),(1,0,8,False),(1,0,-1,False)):
                env=dict(base,C550_WMMA_CONTROL=str(control),C550_WMMA_PREFIX=str(prefix),C550_WMMA_WITNESS=str(mode))
                result=subprocess.run(['bash',str(MODULE.with_name('compile.sh')),str(root/'probe')],env=env,text=True,capture_output=True)
                with self.subTest(control=control,prefix=prefix,mode=mode):
                    if accepted:self.assertEqual(result.returncode,0,result.stderr);self.assertIn('-DC550_WMMA_WITNESS=7',result.stdout.splitlines())
                    else:self.assertEqual((result.returncode,result.stdout),(2,''))


if __name__=='__main__':unittest.main()
