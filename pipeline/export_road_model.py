"""Export pinned SegFormer-B0 road probabilities for browser CPU/WebGPU inference."""
import hashlib
import json
from pathlib import Path
import torch
from transformers import SegformerForSemanticSegmentation

SOURCE='nvidia/segformer-b0-finetuned-cityscapes-1024-1024'
REVISION='21b3847fae21ddee674abd31129307b6a1235bd9'

class Road(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model=model
    def forward(self,pixels):
        return self.model(pixel_values=pixels).logits.softmax(dim=1)[:,0:1]


def export(local=None):
    torch.set_num_threads(2)
    model=SegformerForSemanticSegmentation.from_pretrained(local or SOURCE,**({}if local else {'revision':REVISION})).eval()
    target=Path('web/models/segformer-b0-road-256.onnx')
    torch.onnx.export(Road(model).eval(),torch.zeros(1,3,256,256),target,input_names=['pixels'],output_names=['road_probability'],opset_version=17,dynamo=False)
    raw=target.read_bytes()
    manifest={'version':'segformer-b0-road-256-v1','model':SOURCE,'revision':REVISION,'source':f'https://huggingface.co/{SOURCE}/tree/{REVISION}','file':target.name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'input':{'size':256,'mean':[.485,.456,.406],'std':[.229,.224,.225]},'output_size':64,'road_class':0,'limitations':'Predicted road surface; no lane travel rules, measured movement, speed or surveyed compass heading.'}
    Path('web/models/road-v1.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(manifest)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--local');args=parser.parse_args();export(args.local)
