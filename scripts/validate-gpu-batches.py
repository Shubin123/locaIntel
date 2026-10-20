"""Check all nine batch exports against their shipped batch-2 references.

Optional dependencies: numpy, onnxruntime. Run from the repository root:
python scripts/validate-gpu-batches.py
"""
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
root=Path('web/models');manifest=json.loads((root/'vehicle-facing-v2.json').read_text());rng=np.random.default_rng(731)
# Unique, structured inputs ensure batch indexing does not accidentally repeat
# one crop. Validate both outputs, including the mirrored half of every pair.
inputs=rng.random((16,3,224,224),dtype=np.float32)
options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
report=[]
for key in ['fast','gpu32','gpu']:
 ref=ort.InferenceSession(str(root/manifest[key]['file']),sess_options=options,providers=['CPUExecutionProvider'])
 expected=[[],[]]
 for i in range(0,16,2):
  values=ref.run(None,{'image':inputs[i:i+2]})
  for j,value in enumerate(values):expected[j].append(value)
 expected=[np.concatenate(x) for x in expected];del ref
 for cars in [2,4,8]:
  spec=manifest['gpuBatches'][str(cars)][key]
  session=ort.InferenceSession(str(root/spec['file']),sess_options=options,providers=['CPUExecutionProvider']);actual=session.run(None,{'image':inputs[:cars*2]})
  errors=[float(np.max(np.abs(x-y[:cars*2]))) for x,y in zip(actual,expected)]
  for x,y in zip(actual,expected):np.testing.assert_allclose(x,y[:cars*2],rtol=2e-3 if key=='gpu' else 1e-4,atol=2e-3 if key=='gpu' else 3e-5)
  assert np.array_equal(actual[0].argmax(1),expected[0][:cars*2].argmax(1))
  report.append({'model':key,'cars':cars,'max_abs_errors':errors});print(report[-1],flush=True);del session
print('All FP32/FP16 batch outputs and view classes match the original graphs.')
