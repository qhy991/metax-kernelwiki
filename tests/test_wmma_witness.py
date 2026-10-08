"""Witness admission and CPU diagnostics; synthetic outputs are not device evidence."""
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
SPEC=importlib.util.spec_from_file_location('wmma_witness',MODULE)
probe=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(probe)
SUITE=probe.WITNESS_SUITE


def words(values):
    return [probe.GUARD_VALUE]*64+list(struct.unpack('<256I',struct.pack('<256f',*values)))+[probe.GUARD_VALUE]*64


class WmmaWitnessTest(unittest.TestCase):
    def write_plan(self,path,cases,columns=None):
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=columns or probe.WITNESS_COLUMNS,delimiter='\t',lineterminator='\n')
            writer.writeheader();writer.writerows(cases)

    def test_closed_three_pattern_plan_both_orders_and_old_headers(self):
        for order in probe.ORDERS:
            cases=probe.witness_cases(order)
            self.assertEqual([c['id'] for c in cases],['witness_dense_origin','witness_isolated_origin','witness_isolated_c00'])
            self.assertEqual([(c['pattern'],c['target_row'],c['target_col']) for c in cases],
                             [('dense-origin',13,2),('isolated-origin',13,2),('isolated-c00',0,0)])
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'cases.tsv'
                for selected in (cases,list(reversed(cases)),cases[1:2]):
                    self.write_plan(path,selected)
                    self.assertEqual(probe.read_witness_plan(path),selected)
                    for control,prefix in ((False,False),(True,False),(True,True)):
                        with self.assertRaisesRegex(ValueError,'header'):
                            probe.read_plan(path,control=control,prefix=prefix)
                for old,columns in ((probe.default_cases(),probe.COLUMNS),(probe.control_cases(),probe.CONTROL_COLUMNS),
                                    (probe.prefix_cases(),probe.PREFIX_COLUMNS)):
                    self.write_plan(path,old,columns)
                    with self.assertRaisesRegex(ValueError,'header'):probe.read_witness_plan(path)
        self.assertEqual([len(probe.default_cases()),len(probe.control_cases()),len(probe.prefix_cases())],[12,12,34])
        with self.assertRaises(ValueError):probe.packed_inputs(cases[0])
        with self.assertRaises(ValueError):probe.control_oracle_metadata(prefix=True,suite=SUITE)
        with self.assertRaises(ValueError):probe.control_oracle_metadata(suite='arbitrary')

    def test_closed_shape_target_rule_order_and_count_admission(self):
        base=probe.witness_cases()[1]
        mutations=[('suite','prefix-control'),('pattern','unknown'),('target_row',0),('target_col',0),
                   ('target_row',True),('input_rule','dense-formulas'),('m',1),('n',1),('k',1),('k',3),
                   ('order','unknown'),('warmups',0),('samples',9),('launches',11)]
        for field,value in mutations:
            with self.subTest(field=field,value=value),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'cases.tsv';self.write_plan(path,[dict(base,**{field:value})])
                with self.assertRaises(ValueError):probe.read_witness_plan(path)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cases.tsv'
            self.write_plan(path,probe.witness_cases()+[dict(base,id='fourth')])
            with self.assertRaisesRegex(ValueError,'1 to 3'):probe.read_witness_plan(path)
            self.write_plan(path,[base,dict(base,id='duplicate_pattern')])
            with self.assertRaisesRegex(ValueError,'Duplicate witness pattern'):probe.read_witness_plan(path)
            self.write_plan(path,[])
            with self.assertRaisesRegex(ValueError,'1 to 3'):probe.read_witness_plan(path)

    def test_exact_packing_and_logical_integer_oracle(self):
        for case in probe.witness_cases():
            a,b=probe.witness_packed_inputs(case)
            av=struct.unpack('<1024e',struct.pack('<1024H',*a))
            bv=struct.unpack('<1024e',struct.pack('<1024H',*b))
            expected_a,expected_b=[0.0]*1024,[0.0]*1024
            for row in range(16):
                for k in range(2):
                    if case['pattern']=='dense-origin':
                        expected_a[16*row+k]=((67*row+13*k)%31-15)/16
                        expected_b[16*row+k]=((17*k+5*row+3)%29-14)/16
                    else:
                        if row==case['target_row']:expected_a[16*row+k]=(-12,1)[k]/16
                        if row==case['target_col']:expected_b[16*row+k]=(-1,-13)[k]/16
            self.assertEqual(list(av),expected_a);self.assertEqual(list(bv),expected_b)
            oracle=probe.witness_reference_output(case)
            self.assertEqual(oracle[case['target_row']*16+case['target_col']],-1/256)
            for i in range(16):
                for j in range(16):
                    # Independent logical matrix dot from the decoded prepared matrix.
                    self.assertEqual(oracle[16*i+j],sum(av[16*i+k]*bv[16*j+k] for k in range(2)))
            if case['pattern']!='dense-origin':
                self.assertEqual(sum(value!=0 for value in oracle),1)
                self.assertEqual(sum(value!=0 for value in a),2)
                self.assertEqual(sum(value!=0 for value in b),2)
            self.assertEqual(probe.validate_witness_inputs(case,a,b)['input_halfwords_checked'],2048)
            self.assertEqual(a[256:]+b[256:],[0]*1536)
            for variant in ('wmma','scalar'):
                meta=probe.case_metadata(case,variant,suite=SUITE)
                self.assertEqual((meta['k_chunks'],meta['output_elements']),(1,256))
                self.assertEqual(meta['physical_threads'],64 if variant=='wmma' else 256)

    def test_input_placement_reversed_terms_unrelated_values_and_positive_zero(self):
        case=probe.witness_cases()[1]
        for mutation in ('wrong_a_row','wrong_b_col','nonzero_other_a','nonzero_other_b','reversed_both','negative_zero_padding','tail_chunk'):
            a,b=probe.witness_packed_inputs(case)
            ar,bc=13*16,2*16
            if mutation=='wrong_a_row':a[:2],a[ar:ar+2]=a[ar:ar+2],[0,0]
            elif mutation=='wrong_b_col':b[:2],b[bc:bc+2]=b[bc:bc+2],[0,0]
            elif mutation=='nonzero_other_a':a[0]=probe.halfword(1)
            elif mutation=='nonzero_other_b':b[0]=probe.halfword(1)
            elif mutation=='reversed_both':
                a[ar:ar+2]=reversed(a[ar:ar+2]);b[bc:bc+2]=reversed(b[bc:bc+2])
            elif mutation=='negative_zero_padding':a[2]=0x8000
            else:b[256]=probe.halfword(1)
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,'packed witness'):
                probe.validate_witness_inputs(case,a,b)

    def fixture(self,root,cases):
        inputs,outputs=root/'input',root/'output'
        probe.prepare(inputs,SUITE,cases[0]['order']);outputs.mkdir()
        self.write_plan(inputs/'cases.tsv',cases)
        records=[dict(type='device',name='MetaX C550',visible_device_count=1,wave_size_api=64,
                      pci_bus_id='0000:01:00.0',runtime_version_api=1,driver_version_api=1),
                 dict(type='protocol',schema_version=1,experiment=probe.WITNESS_EXPERIMENT,
                      operand_dtype='float16',accumulator_dtype='float32',output_elements=256,
                      guard_elements_each_side=64,required_wave_size=64,timer='mcEventElapsedTime',
                      comparison='finite exact numeric equality; signed zero equivalent; no tolerance',
                      control_mode=1,variants_per_case=2,input_snapshots=list(probe.SNAPSHOT_PHASES),
                      input_rewrite_between_variants=False,purpose='correctness_diagnostic',performance_accepted=False,
                      **probe.witness_protocol_metadata(),**{field:None for field in probe.ENVIRONMENT})]
        for case in cases:
            a,b=probe.witness_packed_inputs(case)
            records.append(dict(type='logical_case',id=case['id'],**{key:case[key] for key in ('m','n','k','order')},
                                **probe.witness_case_fields(case)))
            def snapshot(phase):
                meta=probe.snapshot_metadata(case,phase,suite=SUITE)
                records.append(dict(type='input_snapshot',id=case['id'],**meta))
                for operand,data in (('a',a),('b',b)):(outputs/meta[operand+'_file']).write_bytes(struct.pack('<1024H',*data))
            snapshot('before')
            for index,variant in enumerate(probe.ORDERS[case['order']]):
                if index:snapshot('between')
                meta=probe.case_metadata(case,variant,suite=SUITE)
                records.append(dict(type='case',id=case['id'],**meta,
                                    function_attributes_before_timing=dict(maxThreadsPerBlock=512,numRegs=16,sharedSizeBytes=0,localSizeBytes=0),
                                    pointer_alignment_observed_bytes=dict(a=512,b=256,c_payload=256)))
                records.extend(dict(type='sample',id=case['id'],variant=variant,sample=i,event_batch_ms=0.1,
                                    host_enqueue_batch_us=50.0) for i in range(10))
                (outputs/meta['output_file']).write_bytes(struct.pack('<384I',*words(probe.witness_reference_output(case))))
            snapshot('after')
        records.append(dict(type='complete',cases=len(cases),variant_executions=len(cases)*2,cpu_correctness_checked=False))
        self.save_records(outputs,records)
        return inputs,outputs,records

    def save_records(self,outputs,records):
        (outputs/'raw.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))

    def test_both_orders_full_diagnostics_and_unpromoted_numeric_failure(self):
        for order in probe.ORDERS:
            cases=probe.witness_cases(order)
            if order=='scalar-first':cases.reverse()
            with self.subTest(order=order),tempfile.TemporaryDirectory() as directory:
                inputs,outputs,_=self.fixture(Path(directory),cases)
                checked=probe.check(inputs,outputs)
                self.assertEqual(checked['status'],'pass')
                self.assertEqual((checked['logical_cases_checked'],checked['variant_outputs_checked']),(3,6))
                self.assertEqual((checked['prepared_input_halfwords_checked'],checked['input_snapshot_halfwords_checked']),(6144,18432))
                self.assertEqual((checked['payload_elements_checked'],checked['guard_elements_checked']),(1536,768))
                self.assertEqual([c['pattern'] for c in checked['cases']],[c['pattern'] for c in cases])
                path=outputs/(cases[0]['id']+'.wmma.f32')
                actual=probe.read_words(path,384,4);actual[64+cases[0]['target_row']*16+cases[0]['target_col']]^=1
                path.write_bytes(struct.pack('<384I',*actual))
                checked=probe.check(inputs,outputs)
                self.assertEqual(checked['status'],'diagnostic_failed')
                self.assertFalse(checked['cases'][0]['variants']['wmma']['exact_passed'])
                self.assertTrue(checked['cases'][0]['variants']['scalar']['exact_passed'])
                self.assertTrue(checked['cases'][1]['passed'])
                self.assertIs(checked['performance_accepted'],False)
                cli=subprocess.run(['python3',str(MODULE),'check',str(inputs),str(outputs)],capture_output=True,text=True)
                self.assertEqual(cli.returncode,1)
                self.assertEqual(json.loads(cli.stdout)['variant_outputs_checked'],6)

    def test_all_witness_bindings_required_in_records_and_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs,outputs,records=self.fixture(Path(directory),[probe.witness_cases()[1]])
            for kind in ('logical_case','case','input_snapshot'):
                for field in ('suite','pattern','target_row','target_col','input_rule'):
                    for replacement in (None,'wrong'):
                        changed=copy.deepcopy(records);row=next(row for row in changed if row['type']==kind)
                        if replacement is None:row.pop(field)
                        else:row[field]=replacement
                        with self.subTest(kind=kind,field=field,replacement=replacement),self.assertRaises(ValueError):
                            probe.validate_control_records(changed,[probe.witness_cases()[1]],suite=SUITE)
            for field in probe.witness_protocol_metadata():
                changed=copy.deepcopy(records);changed[1].pop(field)
                with self.subTest(protocol_field=field),self.assertRaises(ValueError):
                    probe.validate_control_records(changed,[probe.witness_cases()[1]],suite=SUITE)
            for field,value in (('witness_mode',True),('maximum_cases',4),('logical_shape',[1,1,2]),
                                ('isolated_a_numerators',[1,-12]),('target_reference_numerator',1),
                                ('witness_patterns',[dict(probe.witness_pattern_contracts()[0],target_col=False)])):
                changed=copy.deepcopy(records);changed[1][field]=value
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                    probe.validate_control_records(changed,[probe.witness_cases()[1]],suite=SUITE)
            changed=copy.deepcopy(records);changed.pop(next(i for i,r in enumerate(changed) if r['type']=='sample'))
            with self.assertRaises(ValueError):probe.validate_control_records(changed,[probe.witness_cases()[1]],suite=SUITE)
            with self.assertRaises(ValueError):probe.validate_control_records(records,[probe.witness_cases()[1]])
            for oracle in (probe.control_oracle_metadata(),probe.control_oracle_metadata(prefix=True),
                           dict(probe.control_oracle_metadata(suite=SUITE),isolated_b_numerators=[-13,-1])):
                (inputs/'oracle.json').write_text(json.dumps(oracle))
                with self.assertRaises(ValueError):probe.check(inputs,outputs)

    def test_snapshot_mutation_is_complete_failure_and_missing_file_is_structural(self):
        case=probe.witness_cases()[2]
        with tempfile.TemporaryDirectory() as directory:
            inputs,outputs,_=self.fixture(Path(directory),[case])
            path=outputs/(case['id']+'.between.a.f16')
            actual=probe.read_words(path,1024,2);actual[31]=probe.halfword(1)
            path.write_bytes(struct.pack('<1024H',*actual))
            checked=probe.check(inputs,outputs)
            self.assertFalse(checked['passed']);self.assertTrue(checked['structural_valid'])
            snapshot=checked['cases'][0]['input_snapshots']['between']
            self.assertEqual(snapshot['pattern'],'isolated-c00')
            self.assertFalse(snapshot['operands']['a']['equal_to_prepared'])
            self.assertFalse(snapshot['operands']['a']['equal_to_fixed_packing'])
            self.assertTrue(all(v['exact_passed'] for v in checked['cases'][0]['variants'].values()))
            path.unlink()
            with self.assertRaises(OSError):probe.check(inputs,outputs)

    def test_full_output_wrong_placement_unwritten_nonfinite_guards_signed_zero(self):
        case=probe.witness_cases()[1];correct=words(probe.witness_reference_output(case));target=13*16+2
        for mutation in ('wrong_cell','extra_nonzero','unwritten','nonfinite','prefix_guard','suffix_guard','extent'):
            actual=correct.copy()
            if mutation=='wrong_cell':actual[64],actual[64+target]=actual[64+target],actual[64]
            elif mutation=='extra_nonzero':actual[64]=struct.unpack('<I',struct.pack('<f',1/256))[0]
            elif mutation in ('unwritten','nonfinite'):actual[64]=0xffffffff if mutation=='unwritten' else 0x7f800000
            elif mutation=='prefix_guard':actual[0]=0
            elif mutation=='suffix_guard':actual[-1]=0
            else:actual.pop()
            with self.subTest(mutation=mutation):
                if mutation=='extent':
                    with self.assertRaises(ValueError):probe.analyze_control_output(case,actual,suite=SUITE)
                else:
                    checked=probe.analyze_control_output(case,actual,suite=SUITE)
                    self.assertFalse(checked['exact_passed'] and checked['guards_intact'])
        correct[64]=0x80000000
        self.assertTrue(probe.analyze_control_output(case,correct,suite=SUITE)['exact_passed'])

    def test_compile_modes_are_closed_and_defaults_remain_zero(self):
        script=MODULE.with_name('compile.sh')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stub=root/'mxcc-stub'
            stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n');stub.chmod(0o700)
            base=dict(os.environ,MXCC=str(stub),C550_ARCH='xcore1000')
            for key in ('C550_WMMA_CONTROL','C550_WMMA_PREFIX','C550_WMMA_WITNESS'):base.pop(key,None)
            cases=[(None,None,None,True),('0','0','0',True),('1','0','0',True),('1','1','0',True),('1','0','1',True),
                   ('0','0','1',False),('1','1','1',False),('1','0','2',True),('1','0','3',True),('1','0','4',True),('1','0','5',False),('1','0','',False),('1','0','true',False)]
            for control,prefix,witness,accepted in cases:
                env=dict(base)
                for key,value in zip(('C550_WMMA_CONTROL','C550_WMMA_PREFIX','C550_WMMA_WITNESS'),(control,prefix,witness)):
                    if value is not None:env[key]=value
                result=subprocess.run(['bash',str(script),str(root/'probe')],env=env,text=True,capture_output=True)
                with self.subTest(control=control,prefix=prefix,witness=witness):
                    if accepted:
                        self.assertEqual(result.returncode,0,result.stderr)
                        self.assertIn('-DC550_WMMA_WITNESS='+str(witness or '0'),result.stdout.splitlines())
                    else:self.assertEqual((result.returncode,result.stdout),(2,''))


if __name__=='__main__':unittest.main()
