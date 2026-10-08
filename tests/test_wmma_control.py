"""CPU fixtures for paired WMMA/scalar diagnostics; no device observations are fabricated."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

MODULE_PATH=Path(__file__).resolve().parents[1]/"experiments/wmma/experiment.py"
SPEC=importlib.util.spec_from_file_location("wmma_control",MODULE_PATH)
probe=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def output_words(values):
    return [probe.GUARD_VALUE]*64+list(struct.unpack('<256I',struct.pack('<256f',*values)))+[probe.GUARD_VALUE]*64


class WmmaControlTest(unittest.TestCase):
    def fixture(self, root, cases):
        inputs,outputs=root/'input',root/'output'
        probe.prepare(inputs,'scalar-control',cases[0]['order'])
        outputs.mkdir()
        with (inputs/'cases.tsv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=probe.CONTROL_COLUMNS,delimiter='\t')
            writer.writeheader();writer.writerows(cases)
        records=[dict(type='device',name='MetaX C550',visible_device_count=1,wave_size_api=64,
                      pci_bus_id='0000:01:00.0',runtime_version_api=1,driver_version_api=1),
                 dict(type='protocol',schema_version=1,experiment=probe.CONTROL_EXPERIMENT,
                      operand_dtype='float16',accumulator_dtype='float32',output_elements=256,
                      guard_elements_each_side=64,required_wave_size=64,timer='mcEventElapsedTime',
                      comparison='finite exact numeric equality; signed zero equivalent; no tolerance',
                      control_mode=1,variants_per_case=2,input_snapshots=list(probe.SNAPSHOT_PHASES),
                      input_rewrite_between_variants=False,purpose='correctness_diagnostic',performance_accepted=False,
                      **{field:None for field in probe.ENVIRONMENT})]
        for case in cases:
            a,b=probe.packed_inputs(case)
            records.append(dict(type='logical_case',id=case['id'],**{field:case[field] for field in ('m','n','k','order')}))
            def snapshot(phase):
                metadata=probe.snapshot_metadata(case,phase)
                records.append(dict(type='input_snapshot',id=case['id'],**metadata))
                for operand,words in (('a',a),('b',b)):
                    (outputs/metadata[operand+'_file']).write_bytes(struct.pack('<1024H',*words))
            snapshot('before')
            for index,variant in enumerate(probe.ORDERS[case['order']]):
                if index:snapshot('between')
                metadata=probe.case_metadata(case,variant)
                records.append(dict(type='case',id=case['id'],**metadata,
                                    function_attributes_before_timing=dict(maxThreadsPerBlock=512,numRegs=16,sharedSizeBytes=0,localSizeBytes=0),
                                    pointer_alignment_observed_bytes=dict(a=512,b=256,c_payload=256)))
                records.extend(dict(type='sample',id=case['id'],variant=variant,sample=i,
                                    event_batch_ms=0.1,host_enqueue_batch_us=50.0) for i in range(10))
                (outputs/metadata['output_file']).write_bytes(struct.pack('<384I',*output_words(probe.reference_output(case))))
            snapshot('after')
        records.append(dict(type='complete',cases=len(cases),variant_executions=2*len(cases),cpu_correctness_checked=False))
        self.save_records(outputs,records)
        return inputs,outputs,records

    def save_records(self, outputs, records):
        (outputs/'raw.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))

    def case(self, shape=(16,16,16), order='wmma-first'):
        return next(c for c in probe.control_cases(order) if (c['m'],c['n'],c['k'])==shape)

    def test_both_orders_and_all_snapshots_are_accepted_without_performance_acceptance(self):
        for order in probe.ORDERS:
            with self.subTest(order=order),tempfile.TemporaryDirectory() as directory:
                inputs,outputs,_=self.fixture(Path(directory),[self.case(order=order)])
                result=probe.check(inputs,outputs)
                self.assertEqual(result['status'],'pass')
                self.assertIs(result['performance_accepted'],False)
                self.assertEqual(result['input_snapshot_halfwords_checked'],6144)
                case=result['cases'][0]
                self.assertEqual(list(case['variants']),list(probe.ORDERS[order]))
                for variant,row in case['variants'].items():
                    self.assertTrue(row['exact_passed'])
                    self.assertTrue(row['guards_intact'])
                    self.assertEqual(row['physical_threads'],64 if variant=='wmma' else 256)
                    self.assertTrue(row['output_file'].endswith('.'+variant+'.f32'))
                for phase in probe.SNAPSHOT_PHASES:
                    for operand in ('a','b'):
                        observed=case['input_snapshots'][phase]['operands'][operand]
                        self.assertTrue(observed['equal_to_prepared'])
                        self.assertTrue(observed['equal_to_fixed_packing'])

    def test_mixed_numerical_results_retain_all_logical_cases_and_exit_one_with_complete_json(self):
        cases=[self.case((16,16,0)),self.case((16,16,16))]
        with tempfile.TemporaryDirectory() as directory:
            inputs,outputs,_=self.fixture(Path(directory),cases)
            path=outputs/(cases[1]['id']+'.wmma.f32')
            words=probe.read_words(path,384,4);words[64]^=1
            path.write_bytes(struct.pack('<384I',*words))
            result=probe.check(inputs,outputs)
            self.assertEqual(result['status'],'diagnostic_failed')
            self.assertIs(result['structural_valid'],True)
            self.assertEqual(len(result['cases']),2)
            self.assertTrue(result['cases'][0]['passed'])
            self.assertFalse(result['cases'][1]['variants']['wmma']['exact_passed'])
            self.assertTrue(result['cases'][1]['variants']['scalar']['exact_passed'])
            self.assertEqual(result['cases'][1]['variants']['wmma']['mismatch_count'],1)
            process=subprocess.run([os.environ.get('PYTHON','python3'),str(MODULE_PATH),'check',str(inputs),str(outputs)],text=True,capture_output=True)
            self.assertEqual(process.returncode,1)
            self.assertEqual(json.loads(process.stdout)['logical_cases_checked'],2)
            self.assertEqual(process.stderr,'')

    def test_each_snapshot_mutation_is_reported_against_both_host_and_fixed_contract(self):
        c=self.case((7,9,17))
        for phase in probe.SNAPSHOT_PHASES:
            for operand in ('a','b'):
                with self.subTest(phase=phase,operand=operand),tempfile.TemporaryDirectory() as directory:
                    inputs,outputs,_=self.fixture(Path(directory),[c])
                    path=outputs/f"{c['id']}.{phase}.{operand}.f16"
                    words=probe.read_words(path,1024,2);words[1023]^=1
                    path.write_bytes(struct.pack('<1024H',*words))
                    result=probe.check(inputs,outputs)
                    self.assertEqual(result['status'],'diagnostic_failed')
                    observed=result['cases'][0]['input_snapshots'][phase]['operands'][operand]
                    self.assertFalse(observed['equal_to_prepared'])
                    self.assertFalse(observed['equal_to_fixed_packing'])
                    self.assertEqual(observed['mismatch_indices'],[1023])
                    self.assertTrue(all(row['exact_passed'] for row in result['cases'][0]['variants'].values()))

    def test_missing_snapshot_bad_shape_or_order_is_structural_failure(self):
        c=self.case()
        for mutation in ('missing_file','truncated_file','snapshot_order','variant_order','sample_variant','missing_record'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                inputs,outputs,records=self.fixture(Path(directory),[c])
                if mutation=='missing_file':(outputs/f"{c['id']}.between.a.f16").unlink()
                elif mutation=='truncated_file':(outputs/f"{c['id']}.after.b.f16").write_bytes(b'\0\0')
                elif mutation=='snapshot_order':records[3]['phase']='after'
                elif mutation=='variant_order':next(row for row in records if row['type']=='case')['variant']='scalar'
                elif mutation=='sample_variant':next(row for row in records if row['type']=='sample')['variant']='scalar'
                else:records.pop(3)
                self.save_records(outputs,records)
                with self.assertRaises((ValueError,OSError)):
                    probe.check(inputs,outputs)

    def test_wrong_suite_or_input_contract_is_refused(self):
        c=self.case()
        for mutation in ('legacy_oracle','extra_oracle','tampered_prepared','wrong_tsv_order','performance_acceptance'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                inputs,outputs,records=self.fixture(Path(directory),[c])
                if mutation=='legacy_oracle':(inputs/'oracle.json').write_text(json.dumps(probe.oracle_metadata()))
                elif mutation=='extra_oracle':
                    oracle=probe.control_oracle_metadata();oracle['tolerance']=0.001
                    (inputs/'oracle.json').write_text(json.dumps(oracle))
                elif mutation=='tampered_prepared':
                    path=inputs/(c['id']+'.a.f16');words=probe.read_words(path,1024,2);words[0]^=1
                    path.write_bytes(struct.pack('<1024H',*words))
                elif mutation=='wrong_tsv_order':
                    text=(inputs/'cases.tsv').read_text().replace('wmma-first','scalar-first')
                    (inputs/'cases.tsv').write_text(text)
                else:
                    records[1]['performance_accepted']=True;self.save_records(outputs,records)
                with self.assertRaises(ValueError):probe.check(inputs,outputs)

    def test_scalar_wrong_index_nonfinite_and_guard_errors_are_complete_diagnostics(self):
        c=self.case((7,9,17))
        for mutation in ('transpose','nan','guard'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                inputs,outputs,_=self.fixture(Path(directory),[c])
                path=outputs/(c['id']+'.scalar.f32');words=probe.read_words(path,384,4)
                if mutation=='transpose':
                    payload=words[64:320]
                    words[64:320]=[payload[j*16+i] for i in range(16) for j in range(16)]
                elif mutation=='nan':words[64]=0x7fc00000
                else:words[0]=0
                path.write_bytes(struct.pack('<384I',*words))
                result=probe.check(inputs,outputs)
                self.assertEqual(result['status'],'diagnostic_failed')
                row=result['cases'][0]['variants']['scalar']
                self.assertFalse(row['guards_intact'] if mutation=='guard' else row['exact_passed'])
                json.dumps(result,allow_nan=False)
                if mutation=='nan':
                    self.assertIsNone(row['mismatches'][0]['observed_value'])
                    self.assertEqual(row['mismatches'][0]['observed_word_uint32'],0x7fc00000)

    def test_scalar_source_indexing_matches_independent_oracle_over_all_padded_chunks(self):
        for c in probe.default_cases():
            a,b=probe.packed_inputs(c)
            decoded_a=struct.unpack('<1024e',struct.pack('<1024H',*a))
            decoded_b=struct.unpack('<1024e',struct.pack('<1024H',*b))
            observed=[]
            for index in range(256):
                row,col=divmod(index,16)
                total=0.0
                for k in range(((c['k']+15)//16)*16):
                    chunk,local_k=divmod(k,16)
                    product=decoded_a[chunk*256+row*16+local_k]*decoded_b[chunk*256+col*16+local_k]
                    total=struct.unpack('<f',struct.pack('<f',total+product))[0]
                observed.append(total)
            self.assertEqual(observed,probe.reference_output(c))

    def test_compile_flag_admission_and_legacy_prepare_policy(self):
        script=MODULE_PATH.with_name('compile.sh')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stub=root/'mxcc-stub'
            stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n');stub.chmod(0o700)
            base=dict(os.environ,MXCC=str(stub),C550_ARCH='xcore1000');base.pop('C550_WMMA_CONTROL',None)
            for flag in (None,'0','1','','2','true','1.0'):
                env=dict(base)
                if flag is not None:env['C550_WMMA_CONTROL']=flag
                result=subprocess.run(['bash',str(script),str(root/'binary')],env=env,text=True,capture_output=True)
                if flag in (None,'0','1'):
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertIn('-DC550_WMMA_CONTROL='+str(flag or '0'),result.stdout.splitlines())
                else:self.assertEqual((result.returncode,result.stdout),(2,''))
            with self.assertRaises(ValueError):probe.prepare(root/'wrong-order','default','wmma-first')


if __name__=='__main__':
    unittest.main()
