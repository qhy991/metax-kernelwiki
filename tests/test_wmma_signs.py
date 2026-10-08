"""Closed sign contracts and CPU checker tests; no synthetic device claims."""
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
SPEC=importlib.util.spec_from_file_location('wmma_signs',MODULE)
probe=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(probe)
SUITE='sign-control'
CONTRACTS={
 'positive12-k0':([-12,0],[-1,0],12,'positive-k0'),
 'negative12-k0':([12,0],[-1,0],-12,None),
 'positive13-k1':([0,-1],[0,-13],13,None),
 'negative13-k1':([0,1],[0,-13],-13,'negative-k1'),
 'pair-pp':([-12,-1],[-1,-13],25,None),
 'pair-pn':([-12,1],[-1,-13],-1,'pair-forward'),
 'pair-np':([12,-1],[-1,-13],1,None),
 'pair-nn':([12,1],[-1,-13],-25,None),
}

def output_words(values):
 return [0xffffffff]*64+list(struct.unpack('<256I',struct.pack('<256f',*values)))+[0xffffffff]*64


class WmmaSignsTest(unittest.TestCase):
 def write_plan(self,path,cases,columns=probe.SIGN_COLUMNS):
  with path.open('w',newline='') as h:
   w=csv.DictWriter(h,fieldnames=columns,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(cases)

 def test_closed_eight_patterns_orders_subsets_and_old_suite_refusals(self):
  for order in probe.ORDERS:
   cases=probe.sign_cases(order)
   self.assertEqual([c['pattern'] for c in cases],list(CONTRACTS))
   self.assertEqual([c['id'] for c in cases],['sign_'+s.replace('-','_') for s in CONTRACTS])
   with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'cases.tsv'
    for selected in (cases,list(reversed(cases)),cases[4:5]):
     self.write_plan(path,selected);self.assertEqual(probe.read_sign_plan(path),selected)
     for reader in (probe.read_witness_plan,probe.read_product_plan):
      with self.assertRaises(ValueError):reader(path)
    for old,columns,reader in ((probe.default_cases(),probe.COLUMNS,probe.read_plan),
                               (probe.control_cases(),probe.CONTROL_COLUMNS,lambda p:probe.read_plan(p,True)),
                               (probe.prefix_cases(),probe.PREFIX_COLUMNS,lambda p:probe.read_plan(p,True,prefix=True)),
                               (probe.witness_cases(),probe.WITNESS_COLUMNS,probe.read_witness_plan),
                               (probe.product_cases(),probe.PRODUCT_COLUMNS,probe.read_product_plan)):
     self.write_plan(path,old[:1],columns)
     with self.assertRaises(ValueError):probe.read_sign_plan(path)
     self.write_plan(path,cases[:1])
     with self.assertRaises(ValueError):reader(path)
  self.assertEqual([len(probe.default_cases()),len(probe.control_cases()),len(probe.prefix_cases()),len(probe.witness_cases()),len(probe.product_cases())],[12,12,34,3,6])

 def test_admission_refuses_unknown_shapes_targets_rules_and_ninth_row(self):
  case=probe.sign_cases()[5]
  for field,value in (('suite','product-control'),('pattern','pair-forward'),('m',1),('n',1),('k',1),('k',3),
                       ('target_row',1),('target_col',1),('target_row',False),('input_rule','isolated-ordered-products'),
                       ('order','unknown'),('warmups',0),('samples',9),('launches',11)):
   with self.subTest(field=field,value=value),tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'cases.tsv';self.write_plan(path,[dict(case,**{field:value})])
    with self.assertRaises(ValueError):probe.read_sign_plan(path)
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'cases.tsv';self.write_plan(path,probe.sign_cases()+[dict(case,id='ninth')])
   with self.assertRaisesRegex(ValueError,'1 to 8'):probe.read_sign_plan(path)
   self.write_plan(path,[case,dict(case,id='duplicate')])
   with self.assertRaisesRegex(ValueError,'Duplicate sign pattern'):probe.read_sign_plan(path)
  with self.assertRaises(ValueError):probe.control_oracle_metadata(prefix=True,suite=SUITE)

 def test_exact_vectors_signs_magnitudes_and_three_known_baselines(self):
  old={c['pattern']:c for c in probe.product_cases()}
  for case in probe.sign_cases():
   left,right,numerator,baseline=CONTRACTS[case['pattern']]
   a,b=probe.sign_packed_inputs(case)
   self.assertEqual(a,list(struct.unpack('<2H',struct.pack('<2e',*(x/16 for x in left))))+[0]*1022)
   self.assertEqual(b,list(struct.unpack('<2H',struct.pack('<2e',*(x/16 for x in right))))+[0]*1022)
   self.assertEqual(probe.sign_reference_output(case),[numerator/256]+[0.0]*255)
   self.assertEqual(sum(x*y for x,y in zip(left,right)),numerator)
   meta=probe.sign_case_fields(case);products=[x*y for x,y in zip(left,right)]
   self.assertEqual(meta['product_numerators'],products)
   self.assertEqual(meta['product_signs'],[(x>0)-(x<0) for x in products])
   self.assertEqual(meta['product_magnitudes'],[abs(x) for x in products])
   self.assertEqual(meta['nonzero_product_k_slots'],[i for i,x in enumerate(products) if x])
   self.assertEqual(meta['matching_product_pattern'],baseline)
   self.assertEqual(probe.validate_sign_inputs(case,a,b)['input_halfwords_checked'],2048)
   if case['pattern'].startswith('pair-'):self.assertEqual(right,[-1,-13])
   else:
    for k,product in enumerate(products):
     if product==0:self.assertEqual((a[k],b[k]),(0,0))
   if baseline:self.assertEqual((a,b),probe.product_packed_inputs(old[baseline]))
  for row in probe.product_pattern_contracts():
   for key in ('product_signs','product_magnitudes','matching_product_pattern'):self.assertNotIn(key,row)

 def test_factorization_is_fixed_even_when_product_is_unchanged(self):
  for case in probe.sign_cases():
   original_a,original_b=probe.sign_packed_inputs(case)
   slot=next(i for i,(a,b) in enumerate(zip(original_a,original_b)) if a and b)
   moved_a,moved_b=original_a.copy(),original_b.copy();moved_a[slot]^=0x8000;moved_b[slot]^=0x8000
   decode=lambda w:struct.unpack('<e',struct.pack('<H',w))[0]
   self.assertEqual(decode(original_a[slot])*decode(original_b[slot]),decode(moved_a[slot])*decode(moved_b[slot]))
   with self.subTest(pattern=case['pattern']),self.assertRaisesRegex(ValueError,'packed sign'):
    probe.validate_sign_inputs(case,moved_a,moved_b)
   for mutation in ('swap_slots','other_row','other_col','inactive_b','negative_zero','unused_chunk'):
    a,b=original_a.copy(),original_b.copy()
    if mutation=='swap_slots':a[:2]=a[:2][::-1];b[:2]=b[:2][::-1]
    elif mutation=='other_row':a[16]=probe.halfword(1)
    elif mutation=='other_col':b[16]=probe.halfword(1)
    elif mutation=='inactive_b':b[2]=probe.halfword(-13)
    elif mutation=='negative_zero':a[2]=0x8000
    else:b[256]=probe.halfword(1)
    with self.subTest(pattern=case['pattern'],mutation=mutation),self.assertRaises(ValueError):probe.validate_sign_inputs(case,a,b)
   if not case['pattern'].startswith('pair-'):
    a,b=original_a.copy(),original_b.copy();b[1-slot]=probe.halfword(-13)
    # A zero times an altered inactive B still contributes zero; it violates input admission.
    self.assertEqual(a[1-slot],0)
    with self.assertRaises(ValueError):probe.validate_sign_inputs(case,a,b)

 def fixture(self,root,cases):
  inputs,outputs=root/'input',root/'output';probe.prepare(inputs,SUITE,cases[0]['order']);outputs.mkdir();self.write_plan(inputs/'cases.tsv',cases)
  records=[dict(type='device',name='MetaX C550',visible_device_count=1,wave_size_api=64,pci_bus_id='0000:01:00.0',runtime_version_api=1,driver_version_api=1),
   dict(type='protocol',schema_version=1,experiment=probe.SIGN_EXPERIMENT,operand_dtype='float16',accumulator_dtype='float32',output_elements=256,guard_elements_each_side=64,required_wave_size=64,timer='mcEventElapsedTime',comparison='finite exact numeric equality; signed zero equivalent; no tolerance',control_mode=1,variants_per_case=2,input_snapshots=list(probe.SNAPSHOT_PHASES),input_rewrite_between_variants=False,purpose='correctness_diagnostic',performance_accepted=False,**probe.sign_protocol_metadata(),**{k:None for k in probe.ENVIRONMENT})]
  for case in cases:
   a,b=probe.sign_packed_inputs(case)
   records.append(dict(type='logical_case',id=case['id'],**{k:case[k] for k in ('m','n','k','order')},**probe.sign_case_fields(case)))
   def snapshot(phase):
    meta=probe.snapshot_metadata(case,phase,suite=SUITE);records.append(dict(type='input_snapshot',id=case['id'],**meta))
    for op,data in (('a',a),('b',b)):(outputs/meta[op+'_file']).write_bytes(struct.pack('<1024H',*data))
   snapshot('before')
   for i,variant in enumerate(probe.ORDERS[case['order']]):
    if i:snapshot('between')
    meta=probe.case_metadata(case,variant,suite=SUITE)
    records.append(dict(type='case',id=case['id'],**meta,function_attributes_before_timing=dict(maxThreadsPerBlock=512,numRegs=16,sharedSizeBytes=0,localSizeBytes=0),pointer_alignment_observed_bytes=dict(a=512,b=256,c_payload=256)))
    records.extend(dict(type='sample',id=case['id'],variant=variant,sample=j,event_batch_ms=0.1,host_enqueue_batch_us=50.0) for j in range(10))
    (outputs/meta['output_file']).write_bytes(struct.pack('<384I',*output_words(probe.sign_reference_output(case))))
   snapshot('after')
  records.append(dict(type='complete',cases=len(cases),variant_executions=2*len(cases),cpu_correctness_checked=False))
  self.save(outputs,records);return inputs,outputs,records

 def save(self,outputs,records):(outputs/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))

 def test_both_orders_complete_outputs_and_mixed_failure_status(self):
  for order in probe.ORDERS:
   cases=probe.sign_cases(order)
   if order=='scalar-first':cases.reverse()
   with self.subTest(order=order),tempfile.TemporaryDirectory() as directory:
    inputs,outputs,_=self.fixture(Path(directory),cases);checked=probe.check(inputs,outputs)
    self.assertEqual(checked['status'],'pass');self.assertIs(checked['performance_accepted'],False)
    self.assertEqual((checked['logical_cases_checked'],checked['variant_outputs_checked'],checked['payload_elements_checked'],checked['guard_elements_checked']),(8,16,4096,2048))
    self.assertEqual((checked['prepared_input_halfwords_checked'],checked['input_snapshot_halfwords_checked']),(16384,49152))
    for index,variant in ((0,'wmma'),(1,'scalar')):
     path=outputs/(cases[index]['id']+'.'+variant+'.f32');words=probe.read_words(path,384,4);words[64]^=1;path.write_bytes(struct.pack('<384I',*words))
    checked=probe.check(inputs,outputs);self.assertEqual(checked['status'],'diagnostic_failed')
    self.assertFalse(checked['cases'][0]['variants']['wmma']['exact_passed']);self.assertTrue(checked['cases'][0]['variants']['scalar']['exact_passed'])
    self.assertFalse(checked['cases'][1]['variants']['scalar']['exact_passed']);self.assertTrue(all(c['passed'] for c in checked['cases'][2:]))
    cli=subprocess.run(['python3',str(MODULE),'check',str(inputs),str(outputs)],capture_output=True,text=True)
    self.assertEqual(cli.returncode,1);self.assertEqual(json.loads(cli.stdout)['variant_outputs_checked'],16)

 def test_nested_sign_metadata_and_explicit_null_are_required(self):
  # pair-pp has an explicit null prior mapping; pair-pn has a known baseline.
  for case in probe.sign_cases()[4:6]:
   with tempfile.TemporaryDirectory() as directory:
    inputs,outputs,records=self.fixture(Path(directory),[case]);self.assertTrue(probe.check(inputs,outputs)['passed'])
    for kind in ('logical_case','case','input_snapshot'):
     for field in probe.sign_case_fields(case):
      changed=copy.deepcopy(records);next(r for r in changed if r['type']==kind).pop(field)
      with self.subTest(pattern=case['pattern'],kind=kind,missing=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
     for field,value in (('product_signs',[True,1]),('product_signs',[1.0,1]),('product_magnitudes',[12,13.0]),('a_numerators',[-12,True]),('b_numerators',[-1,-13.0]),('k_slots',[False,1]),('matching_product_pattern','pair-reversed')):
      changed=copy.deepcopy(records);next(r for r in changed if r['type']==kind)[field]=value
      with self.subTest(kind=kind,field=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
    for field in probe.sign_protocol_metadata():
     changed=copy.deepcopy(records);changed[1].pop(field)
     with self.subTest(protocol_missing=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
    for field,value in (('product_signs',[True,0]),('product_magnitudes',[12.0,0]),('matching_product_pattern',None)):
     changed=copy.deepcopy(records);changed[1]['sign_patterns'][0][field]=value
     with self.subTest(protocol_nested=field),self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
    changed=copy.deepcopy(records);changed[1]['sign_patterns'][1].pop('matching_product_pattern')
    with self.assertRaises(ValueError):probe.validate_control_records(changed,[case],suite=SUITE)
    oracle=probe.control_oracle_metadata(suite=SUITE);oracle['sign_patterns'][1].pop('matching_product_pattern')
    (inputs/'oracle.json').write_text(json.dumps(oracle))
    with self.assertRaises(ValueError):probe.check(inputs,outputs)

 def test_snapshot_failures_remain_separate_from_output_numerics(self):
  case=probe.sign_cases()[7]
  for phase in probe.SNAPSHOT_PHASES:
   with self.subTest(phase=phase),tempfile.TemporaryDirectory() as directory:
    inputs,outputs,_=self.fixture(Path(directory),[case])
    path=outputs/(case['id']+'.'+phase+'.a.f16');words=probe.read_words(path,1024,2);words[16]=probe.halfword(1);path.write_bytes(struct.pack('<1024H',*words))
    checked=probe.check(inputs,outputs);self.assertFalse(checked['passed']);self.assertTrue(checked['structural_valid'])
    observed=checked['cases'][0]['input_snapshots'][phase]
    self.assertEqual(observed['product_signs'],[-1,-1]);self.assertIsNone(observed['matching_product_pattern'])
    self.assertFalse(observed['operands']['a']['equal_to_prepared']);self.assertFalse(observed['operands']['a']['equal_to_fixed_packing'])
    self.assertTrue(all(v['exact_passed'] for v in checked['cases'][0]['variants'].values()))
    path.unlink()
    with self.assertRaises(OSError):probe.check(inputs,outputs)

 def test_full_output_nonfinite_stray_values_guards_and_signed_zero(self):
  case=probe.sign_cases()[4];correct=output_words(probe.sign_reference_output(case))
  for mutation in ('wrong_target','stray_last','unwritten','nonfinite','prefix_guard','suffix_guard'):
   actual=correct.copy()
   if mutation=='wrong_target':actual[64],actual[65]=actual[65],actual[64]
   elif mutation=='stray_last':actual[319]=actual[64]
   elif mutation=='unwritten':actual[319]=0xffffffff
   elif mutation=='nonfinite':actual[64]=0x7f800000
   elif mutation=='prefix_guard':actual[0]=0
   else:actual[-1]=0
   check=probe.analyze_control_output(case,actual,suite=SUITE)
   with self.subTest(mutation=mutation):self.assertFalse(check['exact_passed'] and check['guards_intact'])
  correct[65:320]=[0x80000000]*255
  self.assertTrue(probe.analyze_control_output(case,correct,suite=SUITE)['exact_passed'])
  with self.assertRaises(ValueError):probe.analyze_control_output(case,correct[:-1],suite=SUITE)

 def test_mode3_is_closed_and_requires_control_without_prefix(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);stub=root/'mxcc-stub';stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n');stub.chmod(0o700)
   base=dict(os.environ,MXCC=str(stub),C550_ARCH='xcore1000')
   for control,prefix,mode,accepted in ((1,0,3,True),(0,0,3,False),(1,1,3,False),(1,0,5,False),(1,0,-1,False)):
    env=dict(base,C550_WMMA_CONTROL=str(control),C550_WMMA_PREFIX=str(prefix),C550_WMMA_WITNESS=str(mode))
    result=subprocess.run(['bash',str(MODULE.with_name('compile.sh')),str(root/'probe')],env=env,text=True,capture_output=True)
    with self.subTest(control=control,prefix=prefix,mode=mode):
     if accepted:self.assertEqual(result.returncode,0,result.stderr);self.assertIn('-DC550_WMMA_WITNESS=3',result.stdout.splitlines())
     else:self.assertEqual((result.returncode,result.stdout),(2,''))


if __name__=='__main__':unittest.main()
