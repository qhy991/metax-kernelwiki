"""Product-control host tests; synthetic outputs are not device observations."""
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
SPEC=importlib.util.spec_from_file_location('wmma_products',MODULE)
probe=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(probe)
SUITE='product-control'
PAIRS={
    'positive-k0':([-12,0],[-1,0],12), 'positive-k1':([0,-12],[0,-1],12),
    'negative-k0':([1,0],[-13,0],-13), 'negative-k1':([0,1],[0,-13],-13),
    'pair-forward':([-12,1],[-1,-13],-1), 'pair-reversed':([1,-12],[-13,-1],-1),
}

def output_words(values):
    return [0xffffffff]*64+list(struct.unpack('<256I',struct.pack('<256f',*values)))+[0xffffffff]*64


class WmmaProductsTest(unittest.TestCase):
    def write_plan(self,path,cases,columns=probe.PRODUCT_COLUMNS):
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=columns,delimiter='\t',lineterminator='\n')
            writer.writeheader();writer.writerows(cases)

    def test_six_closed_patterns_both_orders_and_separate_admission(self):
        for order in probe.ORDERS:
            cases=probe.product_cases(order)
            self.assertEqual([c['pattern'] for c in cases],list(PAIRS))
            self.assertEqual([c['id'] for c in cases],['product_'+p.replace('-','_') for p in PAIRS])
            self.assertTrue(all((c['m'],c['n'],c['k'],c['target_row'],c['target_col'],c['order'])==(16,16,2,0,0,order) for c in cases))
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'cases.tsv'
                for selected in (cases,list(reversed(cases)),cases[3:4]):
                    self.write_plan(path,selected);self.assertEqual(probe.read_product_plan(path),selected)
                    # Same columns do not admit a different suite in witness mode.
                    with self.assertRaisesRegex(ValueError,'witness cases' if len(selected)<=3 else '1 to 3'):
                        probe.read_witness_plan(path)
                    for control,prefix in ((False,False),(True,False),(True,True)):
                        with self.assertRaisesRegex(ValueError,'header'):probe.read_plan(path,control=control,prefix=prefix)
                self.write_plan(path,probe.witness_cases())
                with self.assertRaisesRegex(ValueError,'product cases'):probe.read_product_plan(path)
                for old,columns in ((probe.default_cases(),probe.COLUMNS),(probe.control_cases(),probe.CONTROL_COLUMNS),(probe.prefix_cases(),probe.PREFIX_COLUMNS)):
                    self.write_plan(path,old,columns)
                    with self.assertRaisesRegex(ValueError,'header'):probe.read_product_plan(path)
        self.assertEqual([len(probe.default_cases()),len(probe.control_cases()),len(probe.prefix_cases()),len(probe.witness_cases())],[12,12,34,3])

    def test_shape_target_rule_and_distinct_pattern_bounds(self):
        base=probe.product_cases()[0]
        for field,value in (('suite','witness-control'),('pattern','isolated-c00'),('target_row',1),('target_col',1),
                            ('target_row',True),('input_rule','isolated-fixed-pairs'),('m',1),('n',1),('k',1),('k',3),
                            ('order','unknown'),('warmups',0),('samples',1),('launches',11)):
            with self.subTest(field=field,value=value),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'cases.tsv';self.write_plan(path,[dict(base,**{field:value})])
                with self.assertRaises(ValueError):probe.read_product_plan(path)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cases.tsv'
            self.write_plan(path,probe.product_cases()+[dict(base,id='seventh')])
            with self.assertRaisesRegex(ValueError,'1 to 6'):probe.read_product_plan(path)
            self.write_plan(path,[base,dict(base,id='duplicate')])
            with self.assertRaisesRegex(ValueError,'Duplicate product pattern'):probe.read_product_plan(path)
            self.write_plan(path,[])
            with self.assertRaisesRegex(ValueError,'1 to 6'):probe.read_product_plan(path)
        with self.assertRaises(ValueError):probe.control_oracle_metadata(prefix=True,suite=SUITE)
        with self.assertRaises(ValueError):probe.witness_packed_inputs(base)

    def test_all_input_words_and_independent_expected_products(self):
        for case in probe.product_cases():
            left,right,numerator=PAIRS[case['pattern']]
            a,b=probe.product_packed_inputs(case)
            expected_a=list(struct.unpack('<2H',struct.pack('<2e',*(x/16 for x in left))))+[0]*1022
            expected_b=list(struct.unpack('<2H',struct.pack('<2e',*(x/16 for x in right))))+[0]*1022
            self.assertEqual(a,expected_a);self.assertEqual(b,expected_b)
            reference=probe.product_reference_output(case)
            self.assertEqual(reference,[numerator/256]+[0.0]*255)
            self.assertEqual(sum(x*y for x,y in zip(left,right)),numerator)
            self.assertEqual(probe.validate_product_inputs(case,a,b)['input_halfwords_checked'],2048)
            meta=probe.product_case_fields(case)
            self.assertEqual(meta['a_numerators'],left);self.assertEqual(meta['b_numerators'],right)
            self.assertEqual(meta['k_slots'],[0,1]);self.assertEqual(meta['product_numerators'],[x*y for x,y in zip(left,right)])
            self.assertEqual(meta['nonzero_product_k_slots'],[i for i,(x,y) in enumerate(zip(left,right)) if x*y])
            self.assertEqual(meta['target_reference_numerator'],numerator)
        forward,reversed_case=probe.product_cases()[-2:]
        self.assertEqual(probe.product_reference_output(forward),probe.product_reference_output(reversed_case))
        self.assertNotEqual(probe.product_packed_inputs(forward),probe.product_packed_inputs(reversed_case))
        # The earlier C00 pair remains exactly the same input contract, under its own old label.
        self.assertEqual(probe.product_packed_inputs(forward),probe.witness_packed_inputs(probe.witness_cases()[2]))

    def test_misplacement_nonzero_padding_and_unannounced_slot_swap(self):
        for case in probe.product_cases():
            for mutation in ('wrong_row','wrong_col','other_a','other_b','swap_both','negative_zero','unused_chunk'):
                a,b=probe.product_packed_inputs(case)
                if mutation=='wrong_row':a[16:18],a[:2]=a[:2],[0,0]
                elif mutation=='wrong_col':b[16:18],b[:2]=b[:2],[0,0]
                elif mutation=='other_a':a[16]=probe.halfword(1)
                elif mutation=='other_b':b[16]=probe.halfword(1)
                elif mutation=='swap_both':a[:2]=a[:2][::-1];b[:2]=b[:2][::-1]
                elif mutation=='negative_zero':a[2]=0x8000
                else:b[256]=probe.halfword(1)
                with self.subTest(pattern=case['pattern'],mutation=mutation),self.assertRaisesRegex(ValueError,'packed product'):
                    probe.validate_product_inputs(case,a,b)

    def fixture(self,root,cases):
        inputs,outputs=root/'input',root/'output'
        probe.prepare(inputs,SUITE,cases[0]['order']);outputs.mkdir();self.write_plan(inputs/'cases.tsv',cases)
        records=[dict(type='device',name='MetaX C550',visible_device_count=1,wave_size_api=64,pci_bus_id='0000:01:00.0',runtime_version_api=1,driver_version_api=1),
                 dict(type='protocol',schema_version=1,experiment=probe.PRODUCT_EXPERIMENT,operand_dtype='float16',accumulator_dtype='float32',output_elements=256,
                      guard_elements_each_side=64,required_wave_size=64,timer='mcEventElapsedTime',comparison='finite exact numeric equality; signed zero equivalent; no tolerance',
                      control_mode=1,variants_per_case=2,input_snapshots=list(probe.SNAPSHOT_PHASES),input_rewrite_between_variants=False,
                      purpose='correctness_diagnostic',performance_accepted=False,**probe.product_protocol_metadata(),**{k:None for k in probe.ENVIRONMENT})]
        for case in cases:
            a,b=probe.product_packed_inputs(case)
            records.append(dict(type='logical_case',id=case['id'],**{k:case[k] for k in ('m','n','k','order')},**probe.product_case_fields(case)))
            def snapshot(phase):
                meta=probe.snapshot_metadata(case,phase,suite=SUITE);records.append(dict(type='input_snapshot',id=case['id'],**meta))
                for op,data in (('a',a),('b',b)):(outputs/meta[op+'_file']).write_bytes(struct.pack('<1024H',*data))
            snapshot('before')
            for i,variant in enumerate(probe.ORDERS[case['order']]):
                if i:snapshot('between')
                meta=probe.case_metadata(case,variant,suite=SUITE)
                records.append(dict(type='case',id=case['id'],**meta,function_attributes_before_timing=dict(maxThreadsPerBlock=512,numRegs=16,sharedSizeBytes=0,localSizeBytes=0),pointer_alignment_observed_bytes=dict(a=512,b=256,c_payload=256)))
                records.extend(dict(type='sample',id=case['id'],variant=variant,sample=j,event_batch_ms=0.1,host_enqueue_batch_us=50.0) for j in range(10))
                (outputs/meta['output_file']).write_bytes(struct.pack('<384I',*output_words(probe.product_reference_output(case))))
            snapshot('after')
        records.append(dict(type='complete',cases=len(cases),variant_executions=2*len(cases),cpu_correctness_checked=False))
        self.save(outputs,records);return inputs,outputs,records

    def save(self,outputs,records):(outputs/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))

    def test_both_orders_complete_diagnostics_and_mixed_exact_failures(self):
        for order in probe.ORDERS:
            cases=probe.product_cases(order)
            if order=='scalar-first':cases.reverse()
            with self.subTest(order=order),tempfile.TemporaryDirectory() as directory:
                inputs,outputs,_=self.fixture(Path(directory),cases);checked=probe.check(inputs,outputs)
                self.assertEqual(checked['status'],'pass')
                self.assertEqual((checked['logical_cases_checked'],checked['variant_outputs_checked'],checked['payload_elements_checked'],checked['guard_elements_checked']),(6,12,3072,1536))
                self.assertEqual((checked['prepared_input_halfwords_checked'],checked['input_snapshot_halfwords_checked']),(12288,36864))
                for index,variant in ((0,'wmma'),(1,'scalar')):
                    path=outputs/(cases[index]['id']+'.'+variant+'.f32');actual=probe.read_words(path,384,4);actual[64]^=1
                    path.write_bytes(struct.pack('<384I',*actual))
                checked=probe.check(inputs,outputs)
                self.assertEqual(checked['status'],'diagnostic_failed');self.assertIs(checked['performance_accepted'],False)
                self.assertFalse(checked['cases'][0]['variants']['wmma']['exact_passed']);self.assertTrue(checked['cases'][0]['variants']['scalar']['exact_passed'])
                self.assertFalse(checked['cases'][1]['variants']['scalar']['exact_passed']);self.assertTrue(checked['cases'][1]['variants']['wmma']['exact_passed'])
                self.assertTrue(all(c['passed'] for c in checked['cases'][2:]))
                cli=subprocess.run(['python3',str(MODULE),'check',str(inputs),str(outputs)],capture_output=True,text=True)
                self.assertEqual(cli.returncode,1);self.assertEqual(json.loads(cli.stdout)['variant_outputs_checked'],12)

    def test_nested_metadata_is_required_and_type_sensitive_everywhere(self):
        case=probe.product_cases()[-2]
        with tempfile.TemporaryDirectory() as directory:
            _,_,records=self.fixture(Path(directory),[case])
            for kind in ('logical_case','case','input_snapshot'):
                for field in probe.product_case_fields(case):
                    changed=copy.deepcopy(records);next(r for r in changed if r['type']==kind).pop(field)
                    with self.subTest(kind=kind,missing=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
                for field,value in (('k_slots',[0,True]),('k_slots',[0,1.0]),('a_numerators',[-12,True]),('b_numerators',[-1,-13.0]),
                                    ('product_numerators',[12,-13.0]),('nonzero_product_k_slots',[0,True]),('target_reference_numerator',-1.0)):
                    changed=copy.deepcopy(records);next(r for r in changed if r['type']==kind)[field]=value
                    with self.subTest(kind=kind,field=field,value=value),self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
            for field in probe.product_protocol_metadata():
                changed=copy.deepcopy(records);changed[1].pop(field)
                with self.subTest(protocol_missing=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
            for field,value in (('k_slots',[0,True]),('a_numerators',[-12,0.0]),('target_reference_numerator',12.0)):
                changed=copy.deepcopy(records);changed[1]['product_patterns'][0][field]=value
                with self.subTest(protocol_nested=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
            for value in (1,True,2.0,3):
                changed=copy.deepcopy(records);changed[1]['witness_mode']=value
                with self.subTest(mode=value),self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
            with self.assertRaises(ValueError):probe.validate_control_records(records,[case],suite=probe.WITNESS_SUITE)

    def test_oracle_binding_missing_snapshots_and_snapshot_corruption(self):
        case=probe.product_cases()[2]
        with tempfile.TemporaryDirectory() as directory:
            inputs,outputs,records=self.fixture(Path(directory),[case])
            oracle=probe.control_oracle_metadata(suite=SUITE)
            for changed in (probe.control_oracle_metadata(suite=probe.WITNESS_SUITE),dict(oracle,maximum_cases=3)):
                (inputs/'oracle.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError):probe.check(inputs,outputs)
            changed=copy.deepcopy(oracle);changed['product_patterns'][2]['a_numerators']=[True,0]
            (inputs/'oracle.json').write_text(json.dumps(changed))
            with self.assertRaises(ValueError):probe.check(inputs,outputs)
            (inputs/'oracle.json').write_text(json.dumps(oracle))
            path=outputs/(case['id']+'.between.b.f16');actual=probe.read_words(path,1024,2);actual[17]=probe.halfword(1);path.write_bytes(struct.pack('<1024H',*actual))
            checked=probe.check(inputs,outputs);self.assertFalse(checked['passed']);self.assertTrue(checked['structural_valid'])
            row=checked['cases'][0]['input_snapshots']['between'];self.assertEqual(row['product_numerators'],[-13,0])
            self.assertFalse(row['operands']['b']['equal_to_prepared']);self.assertFalse(row['operands']['b']['equal_to_fixed_packing'])
            self.assertTrue(all(v['exact_passed'] for v in checked['cases'][0]['variants'].values()))
            path.unlink()
            with self.assertRaises(OSError):probe.check(inputs,outputs)

    def test_other255_outputs_guards_nonfinite_and_signed_zero(self):
        case=probe.product_cases()[0];correct=output_words(probe.product_reference_output(case))
        for mutation in ('misplaced','extra_nonzero','unwritten','nonfinite','prefix_guard','suffix_guard'):
            actual=correct.copy()
            if mutation=='misplaced':actual[64],actual[65]=actual[65],actual[64]
            elif mutation=='extra_nonzero':actual[65]=actual[64]
            elif mutation in ('unwritten','nonfinite'):actual[319]=0xffffffff if mutation=='unwritten' else 0x7f800000
            elif mutation=='prefix_guard':actual[0]=0
            else:actual[-1]=0
            checked=probe.analyze_control_output(case,actual,suite=SUITE)
            with self.subTest(mutation=mutation):self.assertFalse(checked['exact_passed'] and checked['guards_intact'])
        correct[65:320]=[0x80000000]*255
        self.assertTrue(probe.analyze_control_output(case,correct,suite=SUITE)['exact_passed'])
        with self.assertRaises(ValueError):probe.analyze_control_output(case,correct[:-1],suite=SUITE)

    def test_mode2_is_explicit_and_requires_control_without_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stub=root/'mxcc-stub';stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n');stub.chmod(0o700)
            base=dict(os.environ,MXCC=str(stub),C550_ARCH='xcore1000')
            for control,prefix,mode,accepted in ((1,0,2,True),(0,0,2,False),(1,1,2,False),(1,0,7,False),(1,0,-1,False)):
                env=dict(base,C550_WMMA_CONTROL=str(control),C550_WMMA_PREFIX=str(prefix),C550_WMMA_WITNESS=str(mode))
                result=subprocess.run(['bash',str(MODULE.with_name('compile.sh')),str(root/'probe')],env=env,text=True,capture_output=True)
                with self.subTest(control=control,prefix=prefix,mode=mode):
                    if accepted:self.assertEqual(result.returncode,0,result.stderr);self.assertIn('-DC550_WMMA_WITNESS=2',result.stdout.splitlines())
                    else:self.assertEqual((result.returncode,result.stdout),(2,''))


if __name__=='__main__':unittest.main()
