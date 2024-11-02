import numpy as np
import torch
import torch.utils.data
import json
import os

from PIL import Image
from controlnet_aux.util import HWC3, resize_image

from pipelines.pose2bvh.managed_model import pose3d_models
from pipelines.pose2bvh.cmu_skeleton import CMUSkeleton

# see: https://github.com/KevinLTT/video2bvh/blob/master/pose_estimator_3d/estimator_3d.py


        
def normalize_screen_coordiantes(pose, w, h):
    assert pose.shape[-1] == 2, f"Wrong shape: {pose.shape}"
    return pose/w*2 - [1, h/w]



def world2camera(pose, R, T):
    """
    Args:
        pose: numpy array with shape (-1, 3)
        R: numpy array with shape (3, 3)
        T: numyp array with shape (3, 1)
    """
    assert pose.shape[-1] == 3
    original_shape = pose.shape 
    pose_world = pose.copy().reshape((-1, 3)).T
    pose_cam = np.matmul(R.T, pose_world - T)
    pose_cam = pose_cam.T.reshape(original_shape)
    return pose_cam


def camera2world(pose, R, T):
    """
    Args:
        pose: numpy array with shape (..., 3)
        R: numpy array with shape (3, 3)
        T: numyp array with shape (3, 1)
    """
    assert pose.shape[-1] == 3
    original_shape = pose.shape
    pose_cam = pose.copy().reshape((-1, 3)).T
    pose_world = np.matmul(R, pose_cam) + T
    pose_world = pose_world.T.reshape(original_shape)
    return pose_world


OPEN_POSE_TO_BODY_25 = {
    0: -1,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: -1, # lambda x: (x[8] + x[11]) / 2,
    9: 8,
    10: 9,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 14, # no more from here
    20: 14,
    21: 14,
    22: 11,
    23: 11,
    24: 11
}

def openpose_to_body25(keypoints):
    keypoints = [
        [k.x, k.y, k.score] if k else [0, 0, 0] for k in keypoints
    ]
    converted = [[0, 0, 0]] * 25
    for i in range(25):
        if OPEN_POSE_TO_BODY_25[i] == -1:
            continue
        converted[i] = keypoints[OPEN_POSE_TO_BODY_25[i]]
        if i >= 19:
            converted[i] = converted[i][:]
            converted[i][2] = converted[i][2] / 2
    converted[8] = [
        (keypoints[8][0] + keypoints[11][0]) / 2, 
        (keypoints[8][1] + keypoints[11][1]) / 2, 
        (keypoints[8][2] + keypoints[11][2]) / 2
    ]
    converted[0] =  [
        (keypoints[16][0] + keypoints[17][0]) / 2, 
        (keypoints[16][1] + keypoints[17][1]) / 2, 
        (keypoints[16][2] + keypoints[17][2]) / 2
    ]
    return np.array(converted)

class WildPoseDataset(object):
    def __init__(self, input_poses, seq_len, image_width, image_height):
        self.seq_len = seq_len
        self.input_poses = normalize_screen_coordiantes(input_poses, image_width, image_height)

    def __len__(self):
        return self.input_poses.shape[0]


    def __getitem__(self, idx):
        frame = idx
        start = frame - self.seq_len//2
        end = frame + self.seq_len//2 + 1
        
        valid_start = max(0, start)
        valid_end = min(self.input_poses.shape[0], end)
        pad = (valid_start - start, end - valid_end)
        input_pose = self.input_poses[valid_start:valid_end]
        if pad != (0, 0):
            input_pose = np.pad(input_pose, (pad, (0, 0), (0, 0)), 'edge')
        if input_pose.shape[0] == 1:
            # squeeze time dimension if sequence length is 1
            input_pose = np.squeeze(input_pose, axis=0)

        sample = { 'input_pose': input_pose }
        return sample


def open_pose_to_bvh(image: Image.Image):
    image = np.array(image, dtype=np.uint8)
    image = HWC3(image)
    image = resize_image(image, 512)
    image_width, image_height = image.shape[1], image.shape[0]
    
    # see https://github.com/huggingface/controlnet_aux/blob/master/src/controlnet_aux/open_pose/__init__.py
    # see https://github.com/KevinLTT/video2bvh
    # see 
    pose3d_models.load_models()
    cfg = pose3d_models.cfg
    model = pose3d_models.model
    
    poses = pose3d_models.openpose.detect_poses(image) # List[PoseResult]
    keypoints_result = list(poses[0].body.keypoints)
    keypoints = openpose_to_body25(poses[0].body.keypoints)
    keypoints_list = [keypoints]
    poses_2d = np.stack(keypoints_list)[:, :, :2]
   
    dataset = WildPoseDataset(
        input_poses=poses_2d,
        seq_len=cfg['DATASET']['SEQ_LEN'],
        image_width=image_width,
        image_height=image_height
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg['TRAIN']['BATCH_SIZE']
    )

    poses_3d = np.zeros((poses_2d.shape[0], cfg['DATASET']['OUT_JOINT'], 3))
    frame = 0
    print('=> Begin to estimate 3D poses.')
    with torch.no_grad():
        for batch in loader:
            input_pose = batch['input_pose'].float().cuda()
            output = model(input_pose)
            # output = output.cpu()
            if cfg['DATASET']['TEST_FLIP']:
                input_lefts = cfg['DATASET']['INPUT_LEFT_JOINTS']
                input_rights = cfg['DATASET']['INPUT_RIGHT_JOINTS']
                output_lefts = cfg['DATASET']['OUTPUT_LEFT_JOINTS']
                output_rights = cfg['DATASET']['OUTPUT_RIGHT_JOINTS']

                flip_input_pose = input_pose.clone()
                flip_input_pose[..., :, 0] *= -1
                flip_input_pose[..., input_lefts + input_rights, :] = flip_input_pose[..., input_rights + input_lefts, :]

                flip_output = model(flip_input_pose)
                flip_output[..., :, 0] *= -1
                flip_output[..., output_lefts + output_rights, :] = flip_output[..., output_rights + output_lefts, :]
                # output = np.concatenate(output, flip_output) / 2
                # output = np.mean(np.array([ output, flip_output ]), axis=0 )
                output = (output + flip_output) / 2
                
            output[:, 0] = 0 # center the root joint
            output *= 1000 # m -> mm

            batch_size = output.shape[0]
            poses_3d[frame:frame+batch_size] = output.cpu().numpy()
            frame += batch_size
            print(f'{frame} / {poses_2d.shape[0]}')
    
    return poses_3d, keypoints_result


def create_openpose_json_file(keypoints, filepath):
    data = {
        "version": 1.1,
        "people": [],
        "part_candidates": []
    }
    for index, k in enumerate(keypoints):
        if k is not None:
            data['part_candidates'].append({
                f"{index}": [k.x, k.y, k.score]
            })
        else:
            data['part_candidates'].append({
                f"{index}": []
            })
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def process_workflow_task(base_dir: str, name: str, input: dict, config: dict, callback: callable):
    images = input.get('default', {}).get('output', None) or input.get('default', {}).get('result', None)
    if images is None:
        raise ValueError("It's required a image pre-process the image #config.input=value")
    if type(images) is str:
        images = [Image.open(images)]
    
    bvh_paths = []
    json_paths = []
    keypoints_list = []
    for index, image in enumerate(images):
        pose3d, keypoints = open_pose_to_bvh(image)
        keypoints_list.append(keypoints)
        pose3d_world = pose3d
        # pose3d_world = camera2world(pose=pose3d, R=R, T=T)
        # pose3d_world[:, :, 2] -= np.min(pose3d_world[:, :, 2]) # rebase the height
        filepath_bvh = os.path.join(base_dir, f'{name}_{index}.bvh')
        filepath_json = os.path.join(base_dir, f'{name}_{index}.json')
        cmu_skel = CMUSkeleton()
        cmu_skel.poses2bvh(pose3d_world, output_file=filepath_bvh)
        create_openpose_json_file(keypoints, filepath_json)
        bvh_paths.append(filepath_bvh)
        json_paths.append(filepath_json)

    return {
        "default": {
            "bvh_paths": bvh_paths, 
            "json_paths": json_paths,
            "coordinates": keypoints_list,
        }
    }





