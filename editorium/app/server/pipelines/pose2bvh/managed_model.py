import json
import gc
import torch
import os

from controlnet_aux import  OpenposeDetector
from pipelines.common.model_manager import ManagedModel

# see: https://github.com/KevinLTT/video2bvh/
MODEL_CONFIG = '''
{
  "DATASET": {
    "CAM_PARAMS": "/home/kevin/public98/3dpose/Dataset/h36m/cameras.h5",
    "IMAGE_HEIGHT": 1002,
    "IMAGE_WIDTH": 1000,
    "INPUT_LEFT_JOINTS": [
      5,
      6,
      7,
      12,
      13,
      14,
      16,
      18,
      19,
      20,
      21
    ],
    "INPUT_RIGHT_JOINTS": [
      2,
      3,
      4,
      9,
      10,
      11,
      15,
      17,
      22,
      23,
      24
    ],
    "INPUT_ROOT": "/home/kevin/HDD/h36m_dataset/2D_openpose",
    "IN_CHANNEL": 2,
    "IN_JOINT": 25,
    "NAME": "h36m",
    "OUTPUT_LEFT_JOINTS": [
      4,
      5,
      6,
      11,
      12,
      13
    ],
    "OUTPUT_RIGHT_JOINTS": [
      1,
      2,
      3,
      14,
      15,
      16
    ],
    "OUT_CHANNEL": 3,
    "OUT_JOINT": 17,
    "SEQ_LEN": 243,
    "TARGET_ROOT": "/home/kevin/HDD/h36m_dataset/3D_gt",
    "TEST_FLIP": true,
    "TRAIN_FLIP": true
  },
  "MODEL": {
    "ACTIVATION": "relu",
    "BIAS": true,
    "DROPOUT": 0.25,
    "DSC": false,
    "FILTER_WIDTHS": [
      3,
      3,
      3,
      3,
      3
    ],
    "HIDDEN_SIZE": 1024,
    "NAME": "video_pose",
    "PRETRAIN": "",
    "RESIDUAL": true
  },
  "TRAIN": {
    "AMSGRAD": true,
    "BATCH_SIZE": 1024,
    "BUFFER_SIZE": 4000000,
    "EPOCH": 80,
    "EVAL_FREQ": 5,
    "LR": 0.001,
    "LR_DECAY": 0.95,
    "MPJPE_WEIGHT": 1,
    "OPTIMIZER": "adam",
    "PRINT_FREQ": 50,
    "SNAP_FREQ": 10000,
    "WORKERS": 4
  }
}
'''


def create_model(cfg, checkpoint_file):
    from pipelines.pose2bvh.video_pose import VideoPose
    model = VideoPose(
        in_joint=cfg['DATASET']['IN_JOINT'],
        in_channel=cfg['DATASET']['IN_CHANNEL'],
        out_joint=cfg['DATASET']['OUT_JOINT'],
        out_channel=cfg['DATASET']['OUT_CHANNEL'],
        filter_widths=cfg['MODEL']['FILTER_WIDTHS'],
        hidden_size=cfg['MODEL']['HIDDEN_SIZE'],
        dropout=cfg['MODEL']['DROPOUT'],
        dsc=cfg['MODEL']['DSC']   
    )

    print(f'=> Load checkpoint {checkpoint_file}')
    pretrained_dict = torch.load(checkpoint_file)['model_state']
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.eval()
    return model   


class Pose3dModels(ManagedModel):
    def __init__(self):
        super().__init__("3dpose-detector")
        self.models_root_path = self.model_dir('3d-objects', '3dpose-detector')
        self.model_path = os.path.join(self.models_root_path, 'github_KevinLTT_video2bvh_model.pth')
        self.cfg = json.loads(MODEL_CONFIG)
        self.model = None
        self.openpose = None
        
    def release_model(self):
        self.model = None
        self.openpose = None
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_models(self):
        self.release_other_models()
        has_changes = any([
            self.model is None,
            self.openpose is None,
        ])
        if not has_changes:
            return
        if not os.path.exists(self.model_path):
            print("See: https://github.com/KevinLTT/video2bvh/tree/master?tab=readme-ov-file#pre-trained-models")
            print("I can not verify the model file safety, so please download it manually yourself.")
            print("Download the model and place it in the following path: ", self.model_path)
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.release_model()
        self.openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        self.model = create_model(self.cfg, self.model_path)
        

pose3d_models = Pose3dModels()

__all__ = ['pose3d_models']
