"""Independent exact dyadic scale tests; synthetic outputs are not C550 evidence."""
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
SPEC=importlib.util.spec_from_file_location('wmma_scales',MODULE)
probe=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(probe)
SUITE='scale-control'


def operands(q,e,role):
    factor=Fraction(2)**e
    a=[-Fraction(q,16)*factor,Fraction(1,16)*factor]
    b=[-Fraction(1,16),-Fraction(q+1,16)]
    if role=='positive':a[1]=b[1]=Fraction(0)
    if role=='negative':a[0]=b[0]=Fraction(0)
    return a,b


def half_words(values):
    return list(struct.unpack('<2H',struct.pack('<2e',*(float(v) for v in values))))+[0]*1022


def expected_values(case):
    a,b=operands(case['q'],case['scale_exp'],case['role'])
    return [float(sum(x*y for x,y in zip(a,b)))]+[0.0]*255


def output_words(values):
    return [0xffffffff]*64+list(struct.unpack('<256I',struct.pack('<256f',*values)))+[0xffffffff]*64


class WmmaScalesTest(unittest.TestCase):
    def write_plan(self,path,cases,columns=probe.SCALE_COLUMNS):
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=columns,delimiter='\t',lineterminator='\n')
            writer.writeheader();writer.writerows(cases)

    def test_closed45_conditions_both_orders_and_all_older_headers(self):
        grid=[(q,e,r) for q in (6,7,12) for e in (-2,-1,0,1,2) for r in ('positive','negative','pair')]
        for order in probe.ORDERS:
            cases=probe.scale_cases(order);self.assertEqual([(c['q'],c['scale_exp'],c['role']) for c in cases],grid)
            self.assertEqual(cases[0]['pattern'],'q06-em2-positive');self.assertEqual(cases[-1]['id'],'scale_q12_ep2_pair')
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'cases.tsv'
                for selected in (cases,list(reversed(cases)),cases[:1]):
                    self.write_plan(path,selected);self.assertEqual(probe.read_scale_plan(path),selected)
                    for reader in (probe.read_witness_plan,probe.read_product_plan,probe.read_sign_plan,probe.read_magnitude_plan):
                        with self.assertRaisesRegex(ValueError,'header'):reader(path)
                for old,columns in ((probe.default_cases(),probe.COLUMNS),(probe.control_cases(),probe.CONTROL_COLUMNS),
                    (probe.prefix_cases(),probe.PREFIX_COLUMNS),(probe.witness_cases(),probe.WITNESS_COLUMNS),
                    (probe.product_cases(),probe.PRODUCT_COLUMNS),(probe.sign_cases(),probe.SIGN_COLUMNS),
                    (probe.magnitude_cases(),probe.MAGNITUDE_COLUMNS)):
                    self.write_plan(path,old[:1],columns)
                    with self.assertRaisesRegex(ValueError,'header'):probe.read_scale_plan(path)

    def test_all45_words_references_support_and_rational_metadata(self):
        for case in probe.scale_cases():
            a,b=operands(case['q'],case['scale_exp'],case['role']);packed=probe.scale_packed_inputs(case)
            self.assertEqual(packed,(half_words(a),half_words(b)))
            reference=sum(x*y for x,y in zip(a,b));self.assertEqual(Fraction(probe.scale_reference_output(case)[0]),reference)
            self.assertEqual(probe.scale_reference_output(case)[1:],[0.0]*255)
            self.assertEqual([i for i,w in enumerate(packed[0]) if w],[i for i,x in enumerate(a) if x])
            self.assertEqual([i for i,w in enumerate(packed[1]) if w],[i for i,x in enumerate(b) if x])
            meta=probe.scale_case_fields(case)
            self.assertEqual(Fraction(meta['scale_numerator'],meta['scale_denominator']),Fraction(2)**case['scale_exp'])
            self.assertEqual([Fraction(x,meta['a_denominator']) for x in meta['a_numerators']],a)
            self.assertEqual([Fraction(x,meta['b_denominator']) for x in meta['b_numerators']],b)
            self.assertEqual([Fraction(x,meta['product_denominator']) for x in meta['product_numerators']],[x*y for x,y in zip(a,b)])
            self.assertEqual(Fraction(meta['target_reference_numerator'],meta['target_reference_denominator']),reference)
            self.assertEqual(meta['a_base_numerators'],[-case['q'] if case['role']!='negative' else 0,1 if case['role']!='positive' else 0])
            self.assertEqual(meta['nonzero_product_k_slots'],[i for i,(x,y) in enumerate(zip(a,b)) if x*y])
            self.assertEqual(probe.validate_scale_inputs(case,*packed)['input_halfwords_checked'],2048)
            for data,values in zip(packed,(a,b)):
                decoded=struct.unpack('<1024e',struct.pack('<1024H',*data))
                self.assertEqual([Fraction(x) for x in decoded[:2]],values)
        pinned=next(c for c in probe.scale_cases() if c['pattern']=='q06-em2-pair')
        self.assertEqual(probe.scale_packed_inputs(pinned)[0][:2],[0xae00,0x2400])
        self.assertEqual(probe.scale_reference_output(pinned)[0],-1/1024)
        oracle=probe.control_oracle_metadata(suite=SUITE)
        self.assertNotIn('input_scale_denominator',oracle);self.assertNotIn('output_scale_denominator',oracle)
        for n in range(-15,16):
            self.assertEqual(probe.halfword(n),struct.unpack('<H',struct.pack('<e',n/16))[0])

    def test_41_distinct_inputs_and14_complete_historical_matches(self):
        cases=probe.scale_cases();buffers={c['pattern']:tuple(tuple(v) for v in probe.scale_packed_inputs(c)) for c in cases}
        self.assertEqual(len(cases),45);self.assertEqual(len(set(buffers.values())),41)
        aliases={(a,b) for i,a in enumerate(buffers) for b in list(buffers)[i+1:] if buffers[a]==buffers[b]}
        self.assertEqual(aliases,{('q06-em1-positive','q12-em2-positive'),('q06-e0-positive','q12-em1-positive'),
                                  ('q06-ep1-positive','q12-e0-positive'),('q06-ep2-positive','q12-ep1-positive')})
        historical={c['pattern']:tuple(tuple(v) for v in probe.magnitude_packed_inputs(c)) for c in probe.magnitude_cases()}
        matched={}
        for c in cases:
            matches=[name for name,data in historical.items() if data==buffers[c['pattern']]]
            self.assertLessEqual(len(matches),1)
            meta=probe.scale_case_fields(c);self.assertIn('matching_magnitude_pattern',meta)
            self.assertEqual(meta['matching_magnitude_pattern'],matches[0] if matches else None)
            if matches:matched[c['pattern']]=matches[0]
        self.assertEqual(len(matched),14)
        extra={k:v for k,v in matched.items() if '-e0-' not in k}
        self.assertEqual(extra,{'q06-em1-positive':'q03-positive','q06-ep1-positive':'q12-positive',
                               'q07-ep1-positive':'q14-positive','q12-em2-positive':'q03-positive','q12-em1-positive':'q06-positive'})

    def test_bad_q_exponent_role_bounds_and_encoder_domain(self):
        base=probe.scale_cases()[23]
        for fields in ({'q':8},{'q':True},{'scale_exp':-3},{'scale_exp':3},{'scale_exp':0.0},{'scale_exp':False},
                       {'scale_exp':'-1.5'},{'scale_exp':'true'},{'role':'other'},{'role':'positive'},
                       {'suite':'magnitude-control'},{'input_rule':'isolated-adjacent-magnitudes'},
                       {'pattern':'q07-em3-pair'},{'target_row':1},{'target_col':1},{'m':1},{'n':1},{'k':1},
                       {'order':'unknown'},{'warmups':0},{'samples':9},{'launches':11}):
            with self.subTest(fields=fields),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'cases.tsv';self.write_plan(path,[dict(base,**fields)])
                with self.assertRaises(ValueError):probe.read_scale_plan(path)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cases.tsv';self.write_plan(path,probe.scale_cases()+[dict(base,id='forty_sixth')])
            with self.assertRaisesRegex(ValueError,'1 to 45'):probe.read_scale_plan(path)
            self.write_plan(path,[base,dict(base,id='duplicate')])
            with self.assertRaisesRegex(ValueError,'Duplicate scale pattern'):probe.read_scale_plan(path)
        for n,e in ((1,-3),(1,3),(16,0),(1,True),(1,0.0),(1.0,0)):
            with self.assertRaises(ValueError):probe.scaled_halfword(n,e)

    def test_wrong_operand_scaling_truncation_factorization_and_padding(self):
        for name in ('q07-em2-pair','q07-ep1-pair'):
            case=next(c for c in probe.scale_cases() if c['pattern']==name)
            for mutation in ('unscaled_a','scaled_b','truncated_a','sign_transfer','swap_slots','other_row','other_col','negative_zero'):
                a,b=probe.scale_packed_inputs(case);meta=probe.scale_case_fields(case)
                if mutation=='unscaled_a':a[:2]=[probe.halfword(x) for x in meta['a_base_numerators']]
                elif mutation=='scaled_b':b[:2]=[probe.scaled_halfword(x,case['scale_exp']) for x in meta['b_numerators']]
                elif mutation=='truncated_a':
                    if case['scale_exp']>0:continue
                    a[:2]=[probe.halfword(int(Fraction(x)*Fraction(2)**case['scale_exp'])) for x in meta['a_base_numerators']]
                elif mutation=='sign_transfer':a[0]^=0x8000;b[0]^=0x8000
                elif mutation=='swap_slots':a[:2]=a[:2][::-1];b[:2]=b[:2][::-1]
                elif mutation=='other_row':a[16]=probe.halfword(1)
                elif mutation=='other_col':b[16]=probe.halfword(1)
                else:a[2]=0x8000
                with self.subTest(name=name,mutation=mutation),self.assertRaises(ValueError):probe.validate_scale_inputs(case,a,b)

    def fixture(self,root,cases):
        inputs,outputs=root/'input',root/'output';probe.prepare(inputs,SUITE,cases[0]['order']);outputs.mkdir();self.write_plan(inputs/'cases.tsv',cases)
        records=[dict(type='device',name='MetaX C550',visible_device_count=1,wave_size_api=64,pci_bus_id='0000:01:00.0',runtime_version_api=1,driver_version_api=1),
                 dict(type='protocol',schema_version=1,experiment=probe.SCALE_EXPERIMENT,operand_dtype='float16',accumulator_dtype='float32',output_elements=256,guard_elements_each_side=64,required_wave_size=64,timer='mcEventElapsedTime',comparison='finite exact numeric equality; signed zero equivalent; no tolerance',control_mode=1,variants_per_case=2,input_snapshots=list(probe.SNAPSHOT_PHASES),input_rewrite_between_variants=False,purpose='correctness_diagnostic',performance_accepted=False,**probe.scale_protocol_metadata(),**{k:None for k in probe.ENVIRONMENT})]
        for case in cases:
            a,b=probe.scale_packed_inputs(case)
            records.append(dict(type='logical_case',id=case['id'],**{k:case[k] for k in ('m','n','k','order')},**probe.scale_case_fields(case)))
            def snapshot(phase):
                meta=probe.snapshot_metadata(case,phase,suite=SUITE);records.append(dict(type='input_snapshot',id=case['id'],**meta))
                for op,data in (('a',a),('b',b)):(outputs/meta[op+'_file']).write_bytes(struct.pack('<1024H',*data))
            snapshot('before')
            for i,variant in enumerate(probe.ORDERS[case['order']]):
                if i:snapshot('between')
                meta=probe.case_metadata(case,variant,suite=SUITE)
                records.append(dict(type='case',id=case['id'],**meta,function_attributes_before_timing=dict(maxThreadsPerBlock=512,numRegs=16,sharedSizeBytes=0,localSizeBytes=0),pointer_alignment_observed_bytes=dict(a=512,b=256,c_payload=256)))
                records.extend(dict(type='sample',id=case['id'],variant=variant,sample=j,event_batch_ms=0.1,host_enqueue_batch_us=50.0) for j in range(10))
                (outputs/meta['output_file']).write_bytes(struct.pack('<384I',*output_words(expected_values(case))))
            snapshot('after')
        records.append(dict(type='complete',cases=len(cases),variant_executions=2*len(cases),cpu_correctness_checked=False))
        self.save(outputs,records);return inputs,outputs,records

    def save(self,outputs,records):(outputs/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))

    def test_full45_both_orders_and_preserved_mixed_failures(self):
        for order in probe.ORDERS:
            cases=probe.scale_cases(order)
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
                self.assertIn('dyadic rational',checked['tested_contract'])
                cli=subprocess.run(['python3',str(MODULE),'check',str(inputs),str(outputs)],capture_output=True,text=True)
                self.assertEqual(cli.returncode,1);self.assertEqual(json.loads(cli.stdout)['variant_outputs_checked'],90)

    def test_deep_types_fraction_metadata_and_explicit_null_required(self):
        for name in ('q07-em2-pair','q07-e0-pair'):
            case=next(c for c in probe.scale_cases() if c['pattern']==name)
            with tempfile.TemporaryDirectory() as directory:
                inputs,outputs,records=self.fixture(Path(directory),[case]);self.assertTrue(probe.check(inputs,outputs)['passed'])
                for kind in ('logical_case','case','input_snapshot'):
                    for field in probe.scale_case_fields(case):
                        changed=copy.deepcopy(records);next(r for r in changed if r['type']==kind).pop(field)
                        with self.subTest(kind=kind,missing=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
                    for field,value in (('scale_exp',False),('scale_exp',float(case['scale_exp'])),('scale_numerator',True),('scale_denominator',1.0),('a_denominator',16.0),('a_base_numerators',[-7,True]),('a_numerators',[-7,1.0]),('product_denominator',256),('target_reference_denominator',1024.0),('matching_magnitude_pattern','q06-pair')):
                        if field=='product_denominator' and case['scale_exp']==0:continue
                        changed=copy.deepcopy(records);next(r for r in changed if r['type']==kind)[field]=value
                        with self.subTest(kind=kind,field=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
                for field in probe.scale_protocol_metadata():
                    changed=copy.deepcopy(records);changed[1].pop(field)
                    with self.subTest(protocol_missing=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
                for field,value in (('scale_exp',-2.0),('a_base_numerators',[-6,False]),('scale_numerator',True)):
                    changed=copy.deepcopy(records);changed[1]['scale_patterns'][0][field]=value
                    with self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
                changed=copy.deepcopy(records);changed[1]['scale_patterns'][0].pop('matching_magnitude_pattern')
                with self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
                oracle=probe.control_oracle_metadata(suite=SUITE);oracle['scale_patterns'][0].pop('matching_magnitude_pattern')
                (inputs/'oracle.json').write_text(json.dumps(oracle))
                with self.assertRaises(ValueError):probe.check(inputs,outputs)

    def test_snapshot_integrity_full_output_and_guards(self):
        case=probe.scale_cases()[-1]
        with tempfile.TemporaryDirectory() as directory:
            inputs,outputs,_=self.fixture(Path(directory),[case]);path=outputs/(case['id']+'.between.a.f16')
            data=probe.read_words(path,1024,2);data[16]=probe.halfword(1);path.write_bytes(struct.pack('<1024H',*data))
            checked=probe.check(inputs,outputs);self.assertFalse(checked['passed']);self.assertTrue(checked['structural_valid'])
            self.assertTrue(all(v['exact_passed'] for v in checked['cases'][0]['variants'].values()))
            row=checked['cases'][0]['input_snapshots']['between'];self.assertFalse(row['operands']['a']['equal_to_fixed_packing']);self.assertEqual(row['scale_exp'],2)
            path.unlink()
            with self.assertRaises(OSError):probe.check(inputs,outputs)
        correct=output_words(expected_values(case))
        for index,value in ((64,0),(319,correct[64]),(65,0xffffffff),(64,0x7f800000),(0,0),(383,0)):
            actual=correct.copy();actual[index]=value;checked=probe.analyze_control_output(case,actual,suite=SUITE)
            self.assertFalse(checked['exact_passed'] and checked['guards_intact'])
        correct[65:320]=[0x80000000]*255;self.assertTrue(probe.analyze_control_output(case,correct,suite=SUITE)['exact_passed'])
        with self.assertRaises(ValueError):probe.analyze_control_output(case,correct[:-1],suite=SUITE)

    def test_mode5_is_explicit_and_six_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stub=root/'mxcc-stub';stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n');stub.chmod(0o700)
            base=dict(os.environ,MXCC=str(stub),C550_ARCH='xcore1000')
            for control,prefix,mode,accepted in ((1,0,5,True),(0,0,5,False),(1,1,5,False),(1,0,6,False),(1,0,-1,False)):
                env=dict(base,C550_WMMA_CONTROL=str(control),C550_WMMA_PREFIX=str(prefix),C550_WMMA_WITNESS=str(mode))
                result=subprocess.run(['bash',str(MODULE.with_name('compile.sh')),str(root/'probe')],env=env,text=True,capture_output=True)
                with self.subTest(control=control,prefix=prefix,mode=mode):
                    if accepted:self.assertEqual(result.returncode,0,result.stderr);self.assertIn('-DC550_WMMA_WITNESS=5',result.stdout.splitlines())
                    else:self.assertEqual((result.returncode,result.stdout),(2,''))


if __name__=='__main__':unittest.main()
