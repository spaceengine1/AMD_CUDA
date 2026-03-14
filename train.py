
import torch
import os
from ultralytics import YOLO

os.environ['ULTRALYTICS_OFFLINE'] = '1'
os.environ['ULTRALYTICS_AMP'] = '0'

os.environ['MIOPEN_FIND_MODE'] = 'NORMAL'
os.environ['MIOPEN_DEBUG_AMD_ROCM_METADATA_ENFORCE'] = '0'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'  # 根据你的GPU调整

torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

model = YOLO(r'D:\SAM\venv1\models\yolov8s-seg.pt')

model.train(data=r'D:\SAM\venv1\models\farm-seg.yaml', workers=0, epochs=50, batch=2,device='cuda:0',amp=False)
