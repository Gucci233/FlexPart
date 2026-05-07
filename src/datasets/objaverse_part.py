from src.utils.typing_utils import *

import json
import os
import random

import accelerate
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.data_utils import load_surface, load_surfaces,load_sharp_surfaces,vector_to_obb_mesh


def sample_and_normalize_point_from_masks(
    masks: List[np.ndarray],
    boxes: List[np.ndarray],
    k_samples: int = 10
) -> List[Optional[Tuple[float, float]]]:
    """
    Sample a "random centroid offset" point from mask list and normalize.
    Args:
        masks (List[np.ndarray]): 
            List of (H, W) boolean masks.
        k_samples (int): 
            Number of candidate points to randomly sample per mask.

    Returns:
        List[Optional[Tuple[float, float]]]:
            List of (x_norm, y_norm) coordinates, None if mask is empty.
    """
    sampled_points_normalized = []
    boxes_normalized = []
    if not masks:
        return []
        
    img_height, img_width = masks[0].shape
    float_width = float(img_width)
    float_height = float(img_height)
    
    for (mask,box) in zip(masks,boxes):
        y_indices, x_indices = np.where(mask)
        
        num_positive_pixels = len(y_indices)
        
        if num_positive_pixels > 0:
            
            centroid_y = np.mean(y_indices)
            centroid_x = np.mean(x_indices)
            
            num_to_sample = min(k_samples, num_positive_pixels)
            
            candidate_indices = np.random.choice(
                num_positive_pixels, size=num_to_sample, replace=False
            )
            
            sampled_y = y_indices[candidate_indices]
            sampled_x = x_indices[candidate_indices]
            
            distances_sq = (sampled_y - centroid_y)**2 + (sampled_x - centroid_x)**2
            
            best_of_k_idx = np.argmin(distances_sq)
            
            
            x_pixel = sampled_x[best_of_k_idx]
            y_pixel = sampled_y[best_of_k_idx]
            
            x_norm = (float(x_pixel) + 0.5) / float_width
            y_norm = (float(y_pixel) + 0.5) / float_height
            x_norm = np.clip(x_norm, 0.0, 1.0)
            y_norm = np.clip(y_norm, 0.0, 1.0)
            
            sampled_points_normalized.append((x_norm, y_norm))
            boxes_x1_norm = (float(box[0]) + 0.5) / float_width
            boxes_y1_norm = (float(box[1]) + 0.5) / float_height
            boxes_x2_norm = (float(box[2]) + 0.5) / float_width
            boxes_y2_norm = (float(box[3]) + 0.5) / float_height
            boxes_x1_norm = np.clip(boxes_x1_norm, 0.0, 1.0)
            boxes_y1_norm = np.clip(boxes_y1_norm, 0.0, 1.0)
            boxes_x2_norm = np.clip(boxes_x2_norm, 0.0, 1.0)
            boxes_y2_norm = np.clip(boxes_y2_norm, 0.0, 1.0)

            boxes_normalized.append((boxes_x1_norm,boxes_y1_norm,boxes_x2_norm,boxes_y2_norm))
        else:
            sampled_points_normalized.append(None) 
            boxes_normalized.append(None)
            
    return sampled_points_normalized,boxes_normalized



def get_box_validity_with_dropout(
    normalized_boxes: List[Union[np.ndarray, List[float]]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_parts = len(normalized_boxes)
    
    if num_parts == 0:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.bool)

    valid_flags = []

    rand_mode = np.random.rand()

    if rand_mode < 0.25:
        valid_flags = [False] * num_parts
        
    elif rand_mode < 0.50:
        for _ in range(num_parts):
            valid_flags.append(np.random.rand() >= 0.5)
            
    else:
        valid_flags = [True] * num_parts

    valid_tensor = torch.tensor(valid_flags, dtype=torch.bool)                  # [N]

    return valid_tensor

def get_mask_validity_with_dropout(
    normalized_masks: List[Union[np.ndarray, List[float]]]
) -> Tuple[torch.Tensor, torch.Tensor]:

    num_parts = len(normalized_masks)
    
    if num_parts == 0:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.bool)

    valid_flags = []

    rand_mode = np.random.rand()

    if rand_mode < 0.5:
        valid_flags = [False] * num_parts
        
    else:
        valid_flags = [True] * num_parts

    valid_tensor = torch.tensor(valid_flags, dtype=torch.bool) 

    return valid_tensor

def get_point_validity_with_dropout(
    normalized_points: List[Union[np.ndarray, List[float]]]
) -> Tuple[torch.Tensor, torch.Tensor]:

    num_parts = len(normalized_points)
    
    if num_parts == 1:
        valid_flags = [False]
        valid_tensor = torch.tensor(valid_flags, dtype=torch.bool) 
        return valid_tensor

    valid_flags = []

    rand_mode = np.random.rand()

    if rand_mode < 0.1:
        valid_flags = [False] * num_parts
            
    else:
        valid_flags = [True] * num_parts

    valid_tensor = torch.tensor(valid_flags, dtype=torch.bool)                  # [N]

    return valid_tensor

class ObjaversePartDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        configs: DictConfig, 
        training: bool = True, 
    ):
        super().__init__()
        self.configs = configs
        self.training = training

        self.min_num_parts = configs['dataset']['min_num_parts']
        self.max_num_parts = configs['dataset']['max_num_parts']
        self.val_min_num_parts = configs['val']['min_num_parts']
        self.val_max_num_parts = configs['val']['max_num_parts']


        self.shuffle_parts = configs['dataset']['shuffle_parts']
        self.training_ratio = configs['dataset']['training_ratio']
        self.balance_object_and_parts = configs['dataset'].get('balance_object_and_parts', False)


        if isinstance(configs['dataset']['config'], ListConfig):
            data_configs = []
            for config in configs['dataset']['config']:
                local_data_configs = json.load(open(config))
                if self.balance_object_and_parts:
                    if self.training:
                        local_data_configs = local_data_configs[:int(len(local_data_configs) * self.training_ratio)]
                    else:
                        local_data_configs = local_data_configs[int(len(local_data_configs) * self.training_ratio):]
                        local_data_configs = [config for config in local_data_configs if self.val_min_num_parts <= config['num_parts'] <= self.val_max_num_parts]
                data_configs += local_data_configs
        else:
            data_configs = json.load(open(configs['dataset']['config']))
        data_configs = [config for config in data_configs if config['valid']]
        data_configs = [config for config in data_configs if self.min_num_parts <= config['num_parts'] <= self.max_num_parts]
        if not self.balance_object_and_parts:
            if self.training:
                data_configs = data_configs[:int(len(data_configs) * self.training_ratio)]
            else:
                data_configs = data_configs[int(len(data_configs) * self.training_ratio):]
                data_configs = [config for config in data_configs if self.val_min_num_parts <= config['num_parts'] <= self.val_max_num_parts]
        self.data_configs = data_configs

        self.image_size = (512,512)

    def __len__(self) -> int:
        return len(self.data_configs)
    
    def _get_data_by_config(self, data_config):
        surface_path = data_config['surface_path']
        surface_data = np.load(surface_path, allow_pickle=True).item()
        part_surfaces = surface_data['parts'] if len(surface_data['parts']) > 0 else [surface_data['object']]
        part_obbs = surface_data['part_obbs']
        temp = list(zip(part_surfaces,part_obbs))
        if self.shuffle_parts:
            random.shuffle(temp)
        part_surfaces, part_obbs = zip(*temp)
        part_surfaces = list(part_surfaces)
        part_obbs = list(part_obbs)

        part_masks = [data["2d_mask"] for data in part_surfaces]
        part_boxes = [data["2d_box"] for data in part_surfaces]
        prompt_points,prompt_boxes = sample_and_normalize_point_from_masks(part_masks,part_boxes)
        prompt_points = [torch.FloatTensor(point) for point in prompt_points]
        prompt_points = torch.stack(prompt_points,dim=0)
        valid_points = get_point_validity_with_dropout(prompt_points)
        
        valid_boxes = get_box_validity_with_dropout(prompt_boxes)
        prompt_boxes = [torch.FloatTensor(box) for box in prompt_boxes]
        prompt_boxes = torch.stack(prompt_boxes,dim=0)

        valid_masks = get_mask_validity_with_dropout(part_masks)
        prompt_masks = [torch.FloatTensor(mask) for mask in part_masks]
        prompt_masks = torch.stack(prompt_masks,dim=0)

        obbs = [torch.FloatTensor(vector_to_obb_mesh(obb)) for obb in part_obbs]
        obbs = torch.stack(obbs,dim=0)

        try:
            part_surfaces = load_sharp_surfaces(part_surfaces) # [N, P, 6]
        except:
            part_surfaces = load_surfaces(part_surfaces) # [N, P, 6]

        image_path = data_config['image_path']
        image = Image.open(image_path).resize(self.image_size)

        image = np.array(image)
        image = torch.from_numpy(image).to(torch.uint8) # [H, W, 3]
        images = torch.stack([image] * part_surfaces.shape[0], dim=0) # [N, H, W, 3]

        return {
            "images": images,
            "part_surfaces": part_surfaces,
            "prompt_points": prompt_points,
            "prompt_boxes":prompt_boxes,
            "prompt_masks":prompt_masks,
            "valid_points":valid_points,
            "valid_boxes":valid_boxes,
            "valid_masks":valid_masks,
            "obbs":obbs,
        }
    
    def __getitem__(self, idx: int):
        data_config = self.data_configs[idx]
        data = self._get_data_by_config(data_config)
        return data
        
class BatchedObjaversePartDataset(ObjaversePartDataset):
    def __init__(
        self,
        configs: DictConfig,
        batch_size: int,
        is_main_process: bool = False,
        shuffle: bool = True,
        training: bool = True,
    ):
        assert training
        assert batch_size > 1
        super().__init__(configs, training)
        self.batch_size = batch_size
        self.is_main_process = is_main_process
        if batch_size < self.max_num_parts:
            self.data_configs = [config for config in self.data_configs if config['num_parts'] <= batch_size]
        
        if shuffle:
            random.shuffle(self.data_configs)

        self.object_configs = [config for config in self.data_configs if config['num_parts'] == 1]
        self.parts_configs = [config for config in self.data_configs if config['num_parts'] > 1]
        
        self.object_ratio = configs['dataset']['object_ratio']
        self.object_configs = self.object_configs[:int(len(self.parts_configs) * self.object_ratio)]

        dropped_data_configs = self.parts_configs + self.object_configs
        if shuffle:
            random.shuffle(dropped_data_configs)
        if len(self.object_configs)> 0:
            self.data_configs = self._get_batched_configs(dropped_data_configs, batch_size)
        else:
            self.data_configs = self._get_batched_configs_improve(dropped_data_configs, batch_size)
    
    def _get_batched_configs(self, data_configs, batch_size):
        batched_data_configs = []
        num_data_configs = len(data_configs)
        progress_bar = tqdm(
            range(len(data_configs)),
            desc="Batching Dataset",
            ncols=125,
            disable=not self.is_main_process,
        )
        while len(data_configs) > 0:
            temp_batch = []
            temp_num_parts = 0
            unchosen_configs = []
            while temp_num_parts < batch_size and len(data_configs) > 0:
                config = data_configs.pop() # pop the last config
                num_parts = config['num_parts']
                if temp_num_parts + num_parts <= batch_size:
                    temp_batch.append(config)
                    temp_num_parts += num_parts
                    progress_bar.update(1)
                else:
                    unchosen_configs.append(config) # add back to the end
            data_configs = data_configs + unchosen_configs # concat the unchosen configs
            if temp_num_parts == batch_size:
                if len(temp_batch) < batch_size:
                    temp_batch += [{}] * (batch_size - len(temp_batch))
                batched_data_configs += temp_batch
        progress_bar.close()
        return batched_data_configs

    def _get_batched_configs_improve(self, data_configs, batch_size):
        """
        Use multi-pass "First-Fit Decreasing" algorithm to pack batches.
        This method significantly reduces data waste.
        """
        final_batched_configs = []
        
        current_configs_to_pack = sorted(data_configs, key=lambda x: x['num_parts'], reverse=True)

        progress_bar = tqdm(
            total=len(current_configs_to_pack),
            desc="Batching Dataset (Improved)",
            ncols=125,
            disable=not self.is_main_process,
        )

        while True:
            new_full_batches_count = 0
            bins = []
            next_pass_configs = []

            for config in current_configs_to_pack:
                num_parts = config['num_parts']
                if num_parts > batch_size:
                    progress_bar.update(1) 
                    continue

                placed_in_bin = False
                for i in range(len(bins)):
                    bin_current_parts = bins[i][1]
                    if bin_current_parts + num_parts <= batch_size:
                        bins[i][0].append(config)
                        bins[i][1] += num_parts
                        placed_in_bin = True
                        break 
                
                if not placed_in_bin:
                    bins.append( [ [config], num_parts ] )

            for items_list, current_sum in bins:
                if current_sum == batch_size:
                    new_full_batches_count += 1
                    
                    progress_bar.update(len(items_list))
                    
                    if len(items_list) < batch_size:
                        items_list += [{}] * (batch_size - len(items_list))
                    final_batched_configs += items_list
                else:
                    next_pass_configs += items_list
            
            if new_full_batches_count == 0:
                progress_bar.update(len(next_pass_configs))
                break
            else:
                current_configs_to_pack = sorted(next_pass_configs, key=lambda x: x['num_parts'], reverse=True)

        progress_bar.close()
        
        return final_batched_configs
    
    def __getitem__(self, idx: int):
        data_config = self.data_configs[idx]
        if len(data_config) == 0:
            return {}
        data = self._get_data_by_config(data_config)
        return data
    
    def collate_fn(self, batch):
        batch = [data for data in batch if len(data) > 0]
        images = torch.cat([data['images'] for data in batch], dim=0) # [N, H, W, 3]
        surfaces = torch.cat([data['part_surfaces'] for data in batch], dim=0) # [N, P, 6]
        num_parts = torch.LongTensor([data['part_surfaces'].shape[0] for data in batch])
        prompt_points = torch.cat([data['prompt_points'] for data in batch], dim=0) # [N, 2]
        prompt_boxes = torch.cat([data['prompt_boxes'] for data in batch], dim=0) # [N, 4]
        prompt_masks = torch.cat([data['prompt_masks'] for data in batch], dim=0) # [N,2,3]
        valid_points = torch.cat([data['valid_points'] for data in batch], dim=0) # [N, 1]
        valid_boxes = torch.cat([data['valid_boxes'] for data in batch], dim=0) # [N, 1]
        valid_masks = torch.cat([data['valid_masks'] for data in batch], dim=0) # [N, 1]
        obbs = torch.cat([data['obbs'] for data in batch], dim=0) # [N, 2,3]
    
        assert images.shape[0] == surfaces.shape[0] == num_parts.sum() == self.batch_size
        batch = {
            "images": images,
            "part_surfaces": surfaces,
            "num_parts": num_parts,
            "prompt_points":prompt_points,
            "prompt_boxes":prompt_boxes,
            "prompt_masks":prompt_masks,
            "valid_points":valid_points,
            "valid_boxes":valid_boxes,
            "valid_masks":valid_masks,
            "obbs":obbs,
        }
        return batch