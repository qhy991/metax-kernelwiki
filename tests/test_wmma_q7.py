"""Single-case CPU checker controls; synthetic files are not device evidence."""
import copy
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

MODULE=Path(__file__).resolve().parents[1]/'experiments/wmma_q7/check.py'
SPEC=importlib.util.spec_from_file_location('q7_check',MODULE)
check=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(check)


def correct_output():
    value=Fraction(-7,16)*Fraction(-1,16)+Fraction(1,16)*Fraction(-8,16)
    payload=list(struct.unpack('<256I',struct.pack('<256f',float(value),*([0.0]*255))))
    return [0xffffffff]*64+payload+[0xffffffff]*64


class WmmaQ7Test(unittest.TestCase):
    def fixture(self,root,order='wmma-first'):
        root.mkdir()
        fixed={'a':[0xb700,0x2c00]+[0]*1022,'b':[0xac00,0xb800]+[0]*1022}
        for op,data in fixed.items():(root/f'prepared.{op}.f16').write_bytes(struct.pack('<1024H',*data))
        records=[dict(type='device',name='MetaX C550',logical_device=0,visible_device_count=1,wave_size_api=64,
                      pci_bus_id='synthetic-test-device',runtime_version_api=1,driver_version_api=1,max_threads_per_block=512),
                 dict(check.protocol_metadata(order),environment={key:None for key in check.ENVIRONMENT})]
        def snapshot(phase):
            records.append(check.snapshot_metadata(order,phase))
            for op,data in fixed.items():(root/f'{phase}.{op}.f16').write_bytes(struct.pack('<1024H',*data))
        snapshot('before')
        for index,variant in enumerate(check.ORDERS[order]):
            records.append(dict(check.variant_metadata(order,variant),
                                function_attributes_before_launch=dict(maxThreadsPerBlock=512,numRegs=16,sharedSizeBytes=0,localSizeBytes=0),
                                pointer_alignment_observed_bytes=dict(a=256,b=512,c_payload=256)))
            (root/f'{variant}.f32').write_bytes(struct.pack('<384I',*correct_output()))
            snapshot('between' if index==0 else 'after')
        records.append(check.completion_metadata(order));self.save(root,records)
        return records

    def save(self,root,records):(root/'raw.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))

    def cli(self,root):
        return subprocess.run(['python3',str(MODULE),str(root)],capture_output=True,text=True)

    def test_fixed_halfwords_exact_fraction_and_complete_support(self):
        words=check.fixed_inputs()
        self.assertEqual(words,{'a':[0xb700,0x2c00]+[0]*1022,'b':[0xac00,0xb800]+[0]*1022})
        a=struct.unpack('<1024e',struct.pack('<1024H',*words['a']))
        b=struct.unpack('<1024e',struct.pack('<1024H',*words['b']))
        self.assertEqual([Fraction(x) for x in a[:2]],[Fraction(-7,16),Fraction(1,16)])
        self.assertEqual([Fraction(x) for x in b[:2]],[Fraction(-1,16),Fraction(-8,16)])
        self.assertEqual(a[2:],(0.0,)*1022);self.assertEqual(b[2:],(0.0,)*1022)
        self.assertEqual(check.reference_output(),[-1/256]+[0.0]*255)
        self.assertEqual(sum(Fraction(x)*Fraction(y) for x,y in zip(a,b)),-Fraction(1,256))

    def test_both_orders_exact_outputs_pass_without_expected_failure(self):
        for order in check.ORDERS:
            with self.subTest(order=order),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'output';records=self.fixture(root,order)
                result=check.check(root)
                self.assertEqual(result['status'],'pass');self.assertTrue(result['passed'])
                self.assertTrue(result['numeric_passed']);self.assertTrue(result['integrity_passed'])
                self.assertEqual(list(result['variants']),list(check.ORDERS[order]))
                self.assertEqual((result['prepared_input_halfwords_checked'],result['input_snapshot_halfwords_checked'],result['payload_elements_checked'],result['guard_elements_checked']),(2048,6144,512,256))
                self.assertEqual([records[i]['variant'] for i in (3,5)],list(check.ORDERS[order]))
                self.assertEqual([records[i]['c_allocation_slot'] for i in (3,5)],[0,1] if order=='wmma-first' else [1,0])
                self.assertEqual(self.cli(root).returncode,0)
                self.assertFalse(any('tim' in key or 'performance' in key for key in result))

    def test_numeric_failure_retains_both_variants_and_exit_one(self):
        for failed in ('wmma','scalar'):
            with self.subTest(variant=failed),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'output';self.fixture(root)
                actual=correct_output();actual[64]^=1
                (root/f'{failed}.f32').write_bytes(struct.pack('<384I',*actual))
                result=check.check(root)
                self.assertEqual(result['status'],'numeric_failed');self.assertFalse(result['passed'])
                self.assertFalse(result['numeric_passed']);self.assertTrue(result['integrity_passed'])
                self.assertFalse(result['variants'][failed]['exact_passed'])
                self.assertTrue(result['variants']['scalar' if failed=='wmma' else 'wmma']['exact_passed'])
                self.assertEqual(result['variants'][failed]['mismatch_count'],1)
                cli=self.cli(root);self.assertEqual(cli.returncode,1);self.assertEqual(json.loads(cli.stdout)['status'],'numeric_failed');self.assertEqual(cli.stderr,'')

    def test_payload_wrong_sign_position_zero_and_nonfinite_are_failures(self):
        for mutation in ('wrong_sign','wrong_position','zero_target','stray_last','unwritten','infinite'):
            data=correct_output()
            if mutation=='wrong_sign':data[64]^=0x80000000
            elif mutation=='wrong_position':data[64],data[65]=data[65],data[64]
            elif mutation=='zero_target':data[64]=0
            elif mutation=='stray_last':data[319]=data[64]
            elif mutation=='unwritten':data[65]=0xffffffff
            else:data[64]=0x7f800000
            row=check.analyze_output(data)
            with self.subTest(mutation=mutation):self.assertFalse(row['exact_passed']);self.assertTrue(row['guards_intact'])
        data=correct_output();data[65:320]=[0x80000000]*255
        self.assertTrue(check.analyze_output(data)['exact_passed'])

    def test_prepared_input_wrong_layout_sign_padding_are_integrity_failures(self):
        for mutation in ('b_layout','a_sign','padding','negative_zero','all_zero'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'output';self.fixture(root)
                op='b' if mutation=='b_layout' else 'a';data=check.fixed_inputs()[op]
                if mutation=='b_layout':data[16],data[1]=data[1],0
                elif mutation=='a_sign':data[0]^=0x8000
                elif mutation=='padding':data[-1]=0x3c00
                elif mutation=='negative_zero':data[2]=0x8000
                else:data=[0]*1024
                (root/f'prepared.{op}.f16').write_bytes(struct.pack('<1024H',*data))
                result=check.check(root)
                self.assertEqual(result['status'],'integrity_failed');self.assertFalse(result['prepared_inputs'][op]['passed'])
                self.assertTrue(result['numeric_passed']);self.assertTrue(result['structural_valid'])
                cli=self.cli(root);self.assertEqual(cli.returncode,2);self.assertEqual(json.loads(cli.stdout)['status'],'integrity_failed')

    def test_each_snapshot_compares_fixed_and_prepared_words(self):
        for phase in check.PHASES:
            for op in ('a','b'):
                with self.subTest(phase=phase,op=op),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory)/'output';self.fixture(root)
                    data=check.fixed_inputs()[op];data[-1]=0x3c00
                    (root/f'{phase}.{op}.f16').write_bytes(struct.pack('<1024H',*data))
                    result=check.check(root);self.assertEqual(result['status'],'integrity_failed')
                    row=result['snapshots'][phase][op]
                    self.assertFalse(row['against_fixed']['passed']);self.assertFalse(row['against_prepared']['passed'])
                    self.assertEqual(row['against_fixed']['mismatch_indices'],[1023])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'output';self.fixture(root)
            bad=check.fixed_inputs()['a'];bad[0]^=0x8000
            for name in ('prepared.a.f16','before.a.f16','between.a.f16','after.a.f16'):(root/name).write_bytes(struct.pack('<1024H',*bad))
            result=check.check(root);self.assertFalse(result['integrity_passed'])
            self.assertTrue(result['snapshots']['before']['a']['against_prepared']['passed'])
            self.assertFalse(result['snapshots']['before']['a']['against_fixed']['passed'])

    def test_guard_corruption_is_separate_from_numerical_status(self):
        for variant,index in (('wmma',0),('wmma',63),('scalar',320),('scalar',383)):
            with self.subTest(variant=variant,index=index),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'output';self.fixture(root)
                data=correct_output();data[index]=0;(root/f'{variant}.f32').write_bytes(struct.pack('<384I',*data))
                result=check.check(root);self.assertEqual(result['status'],'integrity_failed');self.assertTrue(result['numeric_passed'])
                self.assertFalse(result['variants'][variant]['guards_intact']);self.assertFalse(result['integrity_passed'])

    def test_missing_truncated_and_extra_files_do_not_change_expected_extents(self):
        for name in ('prepared.a.f16','before.b.f16','between.a.f16','after.b.f16','wmma.f32','scalar.f32'):
            for mutation in ('missing','truncated','oversized'):
                with self.subTest(name=name,mutation=mutation),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory)/'output';self.fixture(root);path=root/name
                    if mutation=='missing':path.unlink()
                    else:path.write_bytes(path.read_bytes()[:-1] if mutation=='truncated' else path.read_bytes()+b'\x00')
                    cli=self.cli(root);self.assertEqual(cli.returncode,2);self.assertEqual(cli.stdout,'')
                    error=json.loads(cli.stderr);self.assertEqual((error['status'],error['error_kind']),('error','structural'))

    def test_strict_schema_nested_types_device_attributes_and_no_timing_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'output';records=self.fixture(root)
            for index in range(8):
                for field in records[index]:
                    changed=copy.deepcopy(records);changed[index].pop(field)
                    with self.subTest(record=index,missing=field),self.assertRaises(check.StructuralError):check.validate_metadata(changed)
            for index,field,value in ((0,'name','Other GPU'),(0,'wave_size_api',32),(0,'visible_device_count',True),
                                      (0,'runtime_version_api',1.0),(1,'schema','old'),(1,'warmups',1),(1,'total_launches',220),
                                      (1,'launches_per_variant',True),(3,'c_allocation_slot',True),(3,'block',[True,1,1]),
                                      (5,'kernel','wmma_tile_kernel'),(7,'cpu_correctness_checked',True)):
                changed=copy.deepcopy(records);changed[index][field]=value
                with self.subTest(index=index,field=field),self.assertRaises(check.StructuralError):check.validate_metadata(changed)
            for mutation in ('nested_word_bool','attributes_bool','bad_alignment','changed_input_alignment','timing_field','extra_complete','missing_complete'):
                changed=copy.deepcopy(records)
                if mutation=='nested_word_bool':changed[1]['input_words_prefix']['a'][0]=True
                elif mutation=='attributes_bool':changed[3]['function_attributes_before_launch']['numRegs']=True
                elif mutation=='bad_alignment':changed[5]['pointer_alignment_observed_bytes']['a']=3
                elif mutation=='changed_input_alignment':changed[5]['pointer_alignment_observed_bytes']['a']=512
                elif mutation=='timing_field':changed[3]['event_batch_ms']=0.1
                elif mutation=='extra_complete':changed.append(changed[-1])
                else:changed.pop()
                with self.subTest(mutation=mutation),self.assertRaises(check.StructuralError):check.validate_metadata(changed)

    def test_order_snapshot_neighbors_and_output_roles_cannot_be_swapped(self):
        for order in check.ORDERS:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'output';records=self.fixture(root,order)
                for mutation in ('variant_order','records_swap','file_swap','slot_swap','snapshot_order','snapshot_neighbor','missing_null'):
                    changed=copy.deepcopy(records)
                    if mutation=='variant_order':changed[1]['variant_order'].reverse()
                    elif mutation=='records_swap':changed[3],changed[5]=changed[5],changed[3]
                    elif mutation=='file_swap':changed[3]['output_file']=changed[5]['output_file']
                    elif mutation=='slot_swap':changed[3]['c_allocation_slot']=changed[5]['c_allocation_slot']
                    elif mutation=='snapshot_order':changed[4]['order']='wmma-first' if order=='scalar-first' else 'scalar-first'
                    elif mutation=='snapshot_neighbor':changed[4]['after_variant']=changed[4]['before_variant']
                    else:changed[2].pop('after_variant')
                    with self.subTest(order=order,mutation=mutation),self.assertRaises(check.StructuralError):check.validate_metadata(changed)

    def test_duplicate_json_fields_and_nonfinite_metadata_are_structural(self):
        for suffix in ('{"type":"device","type":"device"}\n','NaN\n'):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'output';self.fixture(root);(root/'raw.jsonl').write_text(suffix)
                cli=self.cli(root);self.assertEqual(cli.returncode,2);self.assertEqual(cli.stdout,'')
                self.assertEqual(json.loads(cli.stderr)['error_kind'],'structural')

    def test_compile_flags_target_and_fresh_binary_refusal(self):
        script=MODULE.with_name('compile.sh')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stub=root/'mxcc-stub'
            stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n');stub.chmod(0o700)
            env=dict(os.environ,MXCC=str(stub),MACA_PATH='/sdk');env.pop('C550_ARCH',None)
            result=subprocess.run(['bash',str(script),str(root/'repro')],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);args=result.stdout.splitlines()
            for value in ('-O3','-std=c++17','-x','maca','-offload-arch=xcore1000','--maca-path=/sdk'):self.assertIn(value,args)
            self.assertTrue(any(value.endswith('/wmma_q7/repro.cpp') for value in args))
            self.assertFalse(any('C550_WMMA_' in value for value in args))
            env['C550_ARCH']='xcore1002'
            result=subprocess.run(['bash',str(script),str(root/'repro')],env=env,capture_output=True,text=True)
            self.assertEqual((result.returncode,result.stdout),(2,''))
            env['C550_ARCH']='xcore1000';(root/'repro').write_text('retained')
            result=subprocess.run(['bash',str(script),str(root/'repro')],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,2);self.assertEqual((root/'repro').read_text(),'retained')


if __name__=='__main__':unittest.main()
