"""Prefix-suite admission and CPU oracle tests; these are not WMMA device results."""
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
SPEC=importlib.util.spec_from_file_location('wmma_prefix',MODULE)
probe=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(probe)


def words(values):
    return [probe.GUARD_VALUE]*64+list(struct.unpack('<256I',struct.pack('<256f',*values)))+[probe.GUARD_VALUE]*64


class WmmaPrefixTest(unittest.TestCase):
    def write_plan(self,path,cases,columns=None):
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=columns or probe.PREFIX_COLUMNS,delimiter='\t',lineterminator='\n')
            writer.writeheader();writer.writerows(cases)

    def test_exact_34_case_families_and_old_admission_stay_separate(self):
        for order in probe.ORDERS:
            cases=probe.prefix_cases(order)
            self.assertEqual(len(cases),34)
            self.assertEqual([(c['m'],c['n'],c['k']) for c in cases],[(side,side,k) for side in (1,16) for k in range(17)])
            self.assertEqual({c['order'] for c in cases},{order})
            self.assertEqual({c['suite'] for c in cases},{'prefix-control'})
            self.assertEqual([c['family'] for c in cases],['singleton']*17+['dense']*17)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'cases.tsv'
                self.write_plan(path,cases)
                self.assertEqual(probe.read_plan(path,control=True,prefix=True),cases)
                for control in (False,True):
                    with self.assertRaisesRegex(ValueError,'header'):
                        probe.read_plan(path,control=control)
                self.write_plan(path,list(reversed(cases)))
                self.assertEqual(probe.read_plan(path,control=True,prefix=True),list(reversed(cases)))
        self.assertEqual(len(probe.default_cases()),12)
        self.assertEqual(len(probe.control_cases()),12)
        with self.assertRaises(ValueError):probe.validate_case(probe.prefix_cases()[2])
        with self.assertRaises(ValueError):probe.read_plan(Path('unused'),prefix=True)

    def test_prefix_bounds_and_family_labels_are_not_generic_shape_permission(self):
        base=probe.prefix_cases()[2]
        for field,value in (('k',-1),('k',17),('m',2),('n',16),('suite','scalar-control'),
                            ('family','dense'),('family','other'),('order','unknown')):
            with self.subTest(field=field,value=value),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'cases.tsv'
                self.write_plan(path,[dict(base,**{field:value})])
                with self.assertRaises(ValueError):probe.read_plan(path,control=True,prefix=True)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cases.tsv'
            cases=probe.prefix_cases()+[dict(base,id='thirty_fifth')]
            self.write_plan(path,cases)
            with self.assertRaisesRegex(ValueError,'34'):probe.read_plan(path,control=True,prefix=True)
            cases=probe.control_cases()+[dict(probe.control_cases()[0],id='thirteenth')]
            self.write_plan(path,cases,probe.CONTROL_COLUMNS)
            with self.assertRaisesRegex(ValueError,'12'):probe.read_plan(path,control=True)

    def test_prefix_c00_and_input_row_column_are_invariant_between_families(self):
        numerators=[0,165,153,87,10,4,160,164,52,16,-11,-137,-129,11,4,-56,-128]
        cases=probe.prefix_cases()
        for k in range(17):
            singleton,dense=cases[k],cases[17+k]
            sa,sb=probe.packed_inputs(singleton,prefix=True)
            da,db=probe.packed_inputs(dense,prefix=True)
            self.assertEqual(sa[:16],da[:16])
            self.assertEqual(sb[:16],db[:16])
            self.assertEqual(sa[256:]+sb[256:]+da[256:]+db[256:],[0]*(768*4))
            sr=probe.reference_output(singleton,prefix=True)
            dr=probe.reference_output(dense,prefix=True)
            self.assertEqual(sr[0],numerators[k]/256)
            self.assertEqual(sr[0],dr[0])
            self.assertEqual(sr[1:],[0.0]*255)
            for case in (singleton,dense):
                for variant in ('wmma','scalar'):
                    meta=probe.case_metadata(case,variant,prefix=True)
                    self.assertEqual((meta['tile_m'],meta['tile_n'],meta['tile_k']),(16,16,16))
                    self.assertEqual(meta['k_chunks'],int(k>0))
                    self.assertEqual(meta['physical_threads'],64 if variant=='wmma' else 256)
                    self.assertEqual(meta['output_elements'],256)
                checked=probe.validate_output(case,words(probe.reference_output(case,prefix=True)),prefix=True)
                self.assertEqual((checked['payload_elements_checked'],checked['guard_elements_checked']),(256,128))

    def fixture(self,root,cases):
        inputs,outputs=root/'input',root/'output'
        probe.prepare(inputs,'prefix-control',cases[0]['order']);outputs.mkdir()
        self.write_plan(inputs/'cases.tsv',cases)
        records=[dict(type='device',name='MetaX C550',visible_device_count=1,wave_size_api=64,
                      pci_bus_id='0000:01:00.0',runtime_version_api=1,driver_version_api=1),
                 dict(type='protocol',schema_version=1,experiment=probe.PREFIX_EXPERIMENT,
                      operand_dtype='float16',accumulator_dtype='float32',output_elements=256,
                      guard_elements_each_side=64,required_wave_size=64,timer='mcEventElapsedTime',
                      comparison='finite exact numeric equality; signed zero equivalent; no tolerance',
                      control_mode=1,variants_per_case=2,input_snapshots=list(probe.SNAPSHOT_PHASES),
                      input_rewrite_between_variants=False,purpose='correctness_diagnostic',performance_accepted=False,
                      prefix_mode=1,suite='prefix-control',prefix_families=['singleton','dense'],prefix_k_min=0,prefix_k_max=16,maximum_cases=34,
                      **{field:None for field in probe.ENVIRONMENT})]
        for case in cases:
            a,b=probe.packed_inputs(case,prefix=True)
            records.append(dict(type='logical_case',id=case['id'],**{key:case[key] for key in ('m','n','k','suite','family','order')}))
            def snapshot(phase):
                meta=probe.snapshot_metadata(case,phase,prefix=True)
                records.append(dict(type='input_snapshot',id=case['id'],**meta))
                for operand,data in (('a',a),('b',b)):(outputs/meta[operand+'_file']).write_bytes(struct.pack('<1024H',*data))
            snapshot('before')
            for index,variant in enumerate(probe.ORDERS[case['order']]):
                if index:snapshot('between')
                meta=probe.case_metadata(case,variant,prefix=True)
                records.append(dict(type='case',id=case['id'],**meta,
                                    function_attributes_before_timing=dict(maxThreadsPerBlock=512,numRegs=16,sharedSizeBytes=0,localSizeBytes=0),
                                    pointer_alignment_observed_bytes=dict(a=512,b=256,c_payload=256)))
                records.extend(dict(type='sample',id=case['id'],variant=variant,sample=i,event_batch_ms=0.1,
                                    host_enqueue_batch_us=50.0) for i in range(10))
                (outputs/meta['output_file']).write_bytes(struct.pack('<384I',*words(probe.reference_output(case,prefix=True))))
            snapshot('after')
        records.append(dict(type='complete',cases=len(cases),variant_executions=len(cases)*2,cpu_correctness_checked=False))
        self.save_records(outputs,records)
        return inputs,outputs,records

    def save_records(self,outputs,records):
        (outputs/'raw.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))

    def test_prefix_checker_retains_both_orders_all_outputs_and_numeric_failure(self):
        for order in probe.ORDERS:
            cases=[probe.prefix_cases(order)[2],probe.prefix_cases(order)[19]]
            with self.subTest(order=order),tempfile.TemporaryDirectory() as directory:
                inputs,outputs,_=self.fixture(Path(directory),cases)
                checked=probe.check(inputs,outputs)
                self.assertEqual(checked['status'],'pass')
                self.assertEqual(checked['suite'],'prefix-control')
                self.assertEqual([c['family'] for c in checked['cases']],['singleton','dense'])
                self.assertEqual(checked['input_snapshot_halfwords_checked'],12288)
                path=outputs/(cases[1]['id']+'.wmma.f32')
                actual=probe.read_words(path,384,4);actual[64]^=1
                path.write_bytes(struct.pack('<384I',*actual))
                checked=probe.check(inputs,outputs)
                self.assertEqual(checked['status'],'diagnostic_failed')
                self.assertTrue(checked['cases'][0]['passed'])
                self.assertFalse(checked['cases'][1]['variants']['wmma']['exact_passed'])
                self.assertTrue(checked['cases'][1]['variants']['scalar']['exact_passed'])
                self.assertIs(checked['performance_accepted'],False)
                cli=subprocess.run(['python3',str(MODULE),'check',str(inputs),str(outputs)],capture_output=True,text=True)
                self.assertEqual(cli.returncode,1)
                self.assertEqual(json.loads(cli.stdout)['logical_cases_checked'],2)

    def test_prefix_suite_family_and_bound_metadata_are_required(self):
        c=probe.prefix_cases()[2]
        for mutation in ('protocol_suite','protocol_family','protocol_bound','logical_family','variant_family','snapshot_suite','wrong_oracle','old_reader'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                inputs,outputs,records=self.fixture(Path(directory),[c])
                if mutation=='protocol_suite':records[1]['suite']='scalar-control'
                elif mutation=='protocol_family':records[1]['prefix_families']=['dense']
                elif mutation=='protocol_bound':records[1]['maximum_cases']=12
                elif mutation=='logical_family':records[2]['family']='dense'
                elif mutation=='variant_family':next(row for row in records if row['type']=='case').pop('family')
                elif mutation=='snapshot_suite':next(row for row in records if row['type']=='input_snapshot').pop('suite')
                elif mutation=='wrong_oracle':(inputs/'oracle.json').write_text(json.dumps(probe.control_oracle_metadata()))
                else:
                    with self.assertRaises(ValueError):probe.validate_control_records(records,[c])
                    continue
                self.save_records(outputs,records)
                with self.assertRaises(ValueError):probe.check(inputs,outputs)

    def test_compile_prefix_flag_requires_an_explicit_control_mode(self):
        script=MODULE.with_name('compile.sh')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stub=root/'mxcc-stub'
            stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n');stub.chmod(0o700)
            base=dict(os.environ,MXCC=str(stub),C550_ARCH='xcore1000')
            base.pop('C550_WMMA_CONTROL',None);base.pop('C550_WMMA_PREFIX',None)
            for control,prefix,accepted in ((None,None,True),('0','0',True),('1','0',True),('1','1',True),
                                            ('0','1',False),(None,'1',False),('1','2',False),('1','',False),('1','true',False)):
                env=dict(base)
                if control is not None:env['C550_WMMA_CONTROL']=control
                if prefix is not None:env['C550_WMMA_PREFIX']=prefix
                result=subprocess.run(['bash',str(script),str(root/'probe')],env=env,text=True,capture_output=True)
                with self.subTest(control=control,prefix=prefix):
                    if accepted:
                        self.assertEqual(result.returncode,0,result.stderr)
                        self.assertIn('-DC550_WMMA_PREFIX='+str(prefix or '0'),result.stdout.splitlines())
                    else:self.assertEqual((result.returncode,result.stdout),(2,''))


if __name__=='__main__':unittest.main()
