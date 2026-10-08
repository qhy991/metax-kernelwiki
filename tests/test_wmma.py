"""CPU packing, exact arithmetic and metadata tests; no GPU WMMA execution."""
import csv
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location("wmma",Path(__file__).resolve().parents[1]/"experiments/wmma/experiment.py")
probe=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def case(m,n,k):
    return dict(id=f"wmma_m{m}_n{n}_k{k}",m=m,n=n,k=k,warmups=10,samples=10,launches=10)


def output_words(values):
    payload=list(struct.unpack("<256I",struct.pack("<256f",*values)))
    return [probe.GUARD_VALUE]*64+payload+[probe.GUARD_VALUE]*64


class WmmaTest(unittest.TestCase):
    def test_all_cases_have_fixed_packing_and_full_output_contract(self):
        cases=probe.default_cases()
        self.assertEqual(len(cases),12)
        self.assertEqual([(c['m'],c['n'],c['k']) for c in cases],list(probe.SHAPES))
        for c in cases:
            a,b=probe.packed_inputs(c)
            inputs=probe.validate_inputs(c,a,b)
            self.assertEqual(inputs['input_halfwords_checked'],2048)
            checked=probe.validate_output(c,output_words(probe.reference_output(c)))
            self.assertEqual(checked['payload_elements_checked'],256)
            self.assertEqual(checked['guard_elements_checked'],128)
            self.assertEqual(checked['finite_count'],256)
            self.assertEqual(probe.case_metadata(c)['k_chunks'],(c['k']+15)//16)

    def test_half_inputs_are_exact_and_b_uses_column_major_chunks(self):
        for numerator in range(-15,16):
            actual=struct.unpack('<e',struct.pack('<H',probe.halfword(numerator)))[0]
            self.assertEqual(actual,numerator/16)
        c=case(7,9,17)
        a,b=probe.packed_inputs(c)
        for chunk in range(4):
            for local_k in range(16):
                logical_k=16*chunk+local_k
                for i in range(16):
                    a_value=struct.unpack('<e',struct.pack('<H',a[256*chunk+16*i+local_k]))[0]
                    want=probe.a_numerator(i,logical_k)/16 if i<7 and logical_k<17 else 0.0
                    self.assertEqual(a_value,want)
                for j in range(16):
                    b_value=struct.unpack('<e',struct.pack('<H',b[256*chunk+16*j+local_k]))[0]
                    want=probe.b_numerator(logical_k,j)/16 if j<9 and logical_k<17 else 0.0
                    self.assertEqual(b_value,want)
        self.assertTrue(all(word==0 for word in a[512:]+b[512:]))

    def test_b_wrong_layout_tampered_values_and_nonzero_padding_are_refused(self):
        c=case(7,9,17)
        a,b=probe.packed_inputs(c)
        wrong_layout=[0]*1024
        for chunk in range(4):
            for j in range(16):
                for k in range(16):
                    wrong_layout[chunk*256+k*16+j]=b[chunk*256+j*16+k]
        with self.assertRaisesRegex(ValueError,'packed B'):
            probe.validate_inputs(c,a,wrong_layout)
        for operand,index in (('a',0),('a',15*16),('a',1023),('b',0),('b',15*16),('b',1023)):
            aa,bb=a[:],b[:]
            (aa if operand=='a' else bb)[index]^=1
            with self.subTest(operand=operand,index=index),self.assertRaisesRegex(ValueError,'input words'):
                probe.validate_inputs(c,aa,bb)

    def test_integer_oracle_exactness_and_prefix_bound(self):
        self.assertEqual(probe.reference_output(case(1,1,1))[0],165/256)
        self.assertEqual(probe.reference_output(case(16,16,0)),[0.0]*256)
        for c in probe.default_cases():
            for i in range(c['m']):
                for j in range(c['n']):
                    total=0
                    for k in range(c['k']):
                        total+=probe.a_numerator(i,k)*probe.b_numerator(k,j)
                        self.assertLessEqual(abs(total),13440)
                        value=total/256
                        self.assertEqual(struct.unpack('<f',struct.pack('<f',value))[0],value)

    def test_fp16_rounded_payload_is_rejected_by_exact_fp32_oracle(self):
        examples=((case(9,7,63),-2477/256,-9.671875),
                  (case(16,16,64),-2491/256,-9.734375))
        for c,exact_value,rounded_value in examples:
            expected=probe.reference_output(c)
            self.assertEqual(expected[4],exact_value)
            # Independent binary16 conversion models loss of output precision.
            rounded=[struct.unpack('<e',struct.pack('<e',value))[0] for value in expected]
            self.assertEqual(rounded[4],rounded_value)
            self.assertNotEqual(rounded[4],expected[4])
            with self.subTest(k=c['k']),self.assertRaisesRegex(ValueError,'unequal output'):
                probe.validate_output(c,output_words(rounded))

    def test_transposed_b_and_omitted_k_chunks_fail_output_oracle(self):
        c=case(16,16,16)
        wrong=[sum(probe.a_numerator(i,k)*probe.b_numerator(j,k) for k in range(16))/256
               for i in range(16) for j in range(16)]
        with self.assertRaisesRegex(ValueError,'unequal output'):
            probe.validate_output(c,output_words(wrong))
        for c in (case(7,9,17),case(15,16,31),case(16,16,33),case(9,7,63),case(16,16,64)):
            wrong=[0.0]*256
            incomplete_k=((c['k']+15)//16-1)*16
            for i in range(c['m']):
                for j in range(c['n']):
                    wrong[i*16+j]=sum(probe.a_numerator(i,k)*probe.b_numerator(k,j) for k in range(incomplete_k))/256
            with self.subTest(k=c['k']),self.assertRaisesRegex(ValueError,'unequal output'):
                probe.validate_output(c,output_words(wrong))

    def test_padding_must_be_stored_finite_and_exact_while_signed_zero_is_equivalent(self):
        c=case(1,1,1)
        good=output_words(probe.reference_output(c))
        for index,value in ((1,probe.GUARD_VALUE),(255,0x7F800000),(255,0x3F800000),(0,0x7FC00000)):
            mutated=good[:]
            mutated[64+index]=value
            with self.subTest(index=index,value=value),self.assertRaisesRegex(ValueError,'nonfinite or unequal'):
                probe.validate_output(c,mutated)
        negative_zero=good[:]
        negative_zero[65]=0x80000000
        checked=probe.validate_output(c,negative_zero)
        self.assertEqual(checked['padded_outputs_checked'],255)
        zero=output_words([-0.0]*256)
        self.assertEqual(probe.validate_output(case(16,16,0),zero)['zero_outputs_checked'],256)

    def test_c_extent_and_every_guard_word_are_checked(self):
        c=case(16,16,0)
        good=output_words(probe.reference_output(c))
        for index in (*range(64),*range(320,384)):
            mutated=good[:]
            mutated[index]=0
            with self.subTest(index=index),self.assertRaisesRegex(ValueError,'guard overwritten'):
                probe.validate_output(c,mutated)
        for size in (256,383,385):
            with self.subTest(size=size),self.assertRaisesRegex(ValueError,'extent'):
                probe.validate_output(c,[0]*size)

    def write_plan(self,directory,cases):
        path=Path(directory)/'cases.tsv'
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=probe.COLUMNS,delimiter='\t')
            writer.writeheader();writer.writerows(cases)
        return path

    def test_plan_admission_and_preparation_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs=Path(directory)/'input'
            self.assertEqual(probe.prepare(inputs)['cases'],12)
            self.assertEqual(probe.read_plan(inputs/'cases.tsv'),probe.default_cases())
            for c in probe.default_cases():
                a=probe.read_words(inputs/f"{c['id']}.a.f16",1024,2)
                b=probe.read_words(inputs/f"{c['id']}.b.f16",1024,2)
                probe.validate_inputs(c,a,b)
            for shape in ((0,16,16),(17,16,16),(16,17,16),(16,16,65),(16,16,-1),(8,8,8)):
                with self.subTest(shape=shape),self.assertRaises(ValueError):
                    probe.read_plan(self.write_plan(inputs,[case(*shape)]))

    def records(self,c):
        return [dict(type='device',name='MetaX C550',visible_device_count=1,wave_size_api=64,
                     pci_bus_id='0000:01:00.0',runtime_version_api=1,driver_version_api=1),
                dict(type='protocol',schema_version=1,experiment=probe.EXPERIMENT,operand_dtype='float16',
                     accumulator_dtype='float32',output_elements=256,guard_elements_each_side=64,required_wave_size=64,
                     timer='mcEventElapsedTime',comparison='finite exact numeric equality; signed zero equivalent; no tolerance',
                     **{field:None for field in probe.ENVIRONMENT}),
                dict(type='case',id=c['id'],**probe.case_metadata(c),
                     pointer_alignment_observed_bytes=dict(a=512,b=256,c_payload=256),
                     function_attributes_before_timing=dict(maxThreadsPerBlock=512,numRegs=16,sharedSizeBytes=0,localSizeBytes=0)),
                *[dict(type='sample',id=c['id'],sample=i,event_batch_ms=0.1,host_enqueue_batch_us=50.0) for i in range(10)],
                dict(type='complete',cases=1,cpu_correctness_checked=False)]

    def test_valid_records_and_nonprescriptive_alignment(self):
        c=case(7,9,17)
        records=self.records(c)
        self.assertEqual(len(probe.validate_records(records,[c])[c['id']]),10)
        records[2]['pointer_alignment_observed_bytes']=dict(a=2,b=2,c_payload=4)
        probe.validate_records(records,[c])

    def test_tampered_layout_chunk_output_and_alignment_metadata_fail(self):
        c=case(7,9,17)
        for field,wrong in (('b_layout','row_major'),('a_layout','col_major'),('c_layout','col_major'),
                            ('k_chunks',1),('packed_chunks',2),('output_elements',63),('guard_elements_each_side',32),
                            ('leading_dimension',8),('operand_dtype','float32'),('physical_threads',32)):
            for missing in (False,True):
                records=self.records(c)
                if missing:records[2].pop(field)
                else:records[2][field]=wrong
                with self.subTest(field=field,missing=missing),self.assertRaisesRegex(ValueError,'metadata mismatch'):
                    probe.validate_records(records,[c])
        for value in (None,True,0,3,256.0):
            records=self.records(c)
            if value is None:records[2]['pointer_alignment_observed_bytes'].pop('a')
            else:records[2]['pointer_alignment_observed_bytes']['a']=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'pointer-alignment'):
                probe.validate_records(records,[c])

    def test_missing_samples_bad_resources_and_wrong_wave_fail(self):
        c=case(16,16,16)
        for mutation in ('no_complete','missing_sample','duplicate_sample','nan','wave32','no_resource'):
            records=self.records(c)
            if mutation=='no_complete':records.pop()
            elif mutation=='missing_sample':records.pop(3)
            elif mutation=='duplicate_sample':records[3]['sample']=1
            elif mutation=='nan':records[3]['event_batch_ms']=float('nan')
            elif mutation=='wave32':records[0]['wave_size_api']=32
            else:records[2]['function_attributes_before_timing'].pop('numRegs')
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                probe.validate_records(records,[c])

    def test_complete_file_checker_validates_packed_inputs_and_full_c(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs,outputs=Path(directory)/'input',Path(directory)/'output'
            probe.prepare(inputs);outputs.mkdir()
            c=case(1,1,1)
            self.write_plan(inputs,[c])
            words=output_words([165/256]+[0.0]*255)
            (outputs/f"{c['id']}.f32").write_bytes(struct.pack('<384I',*words))
            (outputs/'raw.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in self.records(c)))
            checked=probe.check(inputs,outputs)
            self.assertEqual(checked['status'],'pass')
            self.assertEqual(checked['cases'][0]['input_halfwords_checked'],2048)
            self.assertEqual(checked['cases'][0]['payload_elements_checked'],256)
            self.assertEqual(checked['cases'][0]['padded_outputs_checked'],255)


if __name__=='__main__':
    unittest.main()
