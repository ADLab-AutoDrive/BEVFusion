import math
import numpy as np

import torch
from mmcv.runner import auto_fp16
from torch import nn

from ..builder import MIDDLE_ENCODERS
from mmdet3d.ops import flat2window, window2flat
import random
import pickle as pkl
import os

@MIDDLE_ENCODERS.register_module()
class SSTInputLayer(nn.Module):
    """
    This is one of the core class of SST, converting the output of voxel_encoder to sst input.
    There are 3 things to be done in this class:
    1. Reginal Grouping : assign window indices to each voxel.
    2. Voxel drop and region batching: see our paper for detail
    3. Pre-computing the transfomation information for converting flat features ([N x C]) to region features ([R, T, C]). R is the number of regions containing at most T tokens (voxels). See function flat2window and window2flat for details.

    Main args:
        drop_info (dict): drop configuration for region batching. 
        window_shape (tuple[int]): (num_x, num_y). Each window is divided to num_x * num_y pillars (including empty pillars).
        shift_list (list[tuple]): [(shift_x, shift_y), ]. shift_x = 5 means all windonws will be shifted for 5 voxels along positive direction of x-aixs.
        debug: apply strong assertion for developing. 
    """

    def __init__(self,
        drop_info,
        shifts_list,
        window_shape,
        point_cloud_range,
        voxel_size,
        shuffle_voxels=True,
        debug=True,
        ):
        super().__init__()
        self.fp16_enabled = False
        self.meta_drop_info = drop_info
        self.shifts_list = shifts_list
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.shuffle_voxels = shuffle_voxels
        self.debug = debug
        self.window_shape = window_shape



    @auto_fp16(apply_to=('voxel_feat', ))
    def forward(self, voxel_feat, coors):
        '''
        Args:
            voxel_feat: shape=[N, C], N is thex voxel num in the batch.
            coors: shape=[N, 4], [b, z, y, x]
        Returns:
            feat_3d_dict: contains region features (feat_3d) of each region batching level. Shape of feat_3d is [num_windows, num_max_tokens, C].
            flat2win_inds_list: two dict containing transformation information for non-shifted grouping and shifted grouping, respectively. The two dicts are used in function flat2window and window2flat.
            voxel_info: dict containing extra information of each voxel for usage in the backbone.
        '''
        self.set_drop_info()
        voxel_info = {}
        coors = coors.long()
        if self.shuffle_voxels:
            # shuffle the voxels to make the drop process uniform.
            num_voxel = len(voxel_feat)
            shuffle_inds = torch.randperm(num_voxel)
            voxel_feat = voxel_feat[shuffle_inds]
            coors = coors[shuffle_inds]
            for k, tensor in voxel_info.items():
                if isinstance(tensor, torch.Tensor) and len(tensor) == num_voxel:
                    voxel_info[k] = tensor[shuffle_inds]

        voxel_info = self.window_partition(coors, voxel_info) 
        voxel_info = self.get_voxel_keep_inds(voxel_info, len(self.shifts_list)) # voxel_info is updated in this function

        voxel_keep_inds = voxel_info['voxel_keep_inds']

        voxel_num_before_drop = len(voxel_feat)
        voxel_feat = voxel_feat[voxel_keep_inds]
        coors = coors[voxel_keep_inds]
        voxel_info['coors'] = coors

        # Some other variables need to be dropped.
        for k, v in voxel_info.items():
            if isinstance(v, torch.Tensor) and len(v) == voxel_num_before_drop:
                voxel_info[k] = v[voxel_keep_inds]

        flat2win_inds_list = [
            self.get_flat2win_inds(voxel_info[f'batch_win_inds_shift{i}'], voxel_info[f'voxel_drop_level_shift{i}']) 
            for i in range(len(self.shifts_list))
        ]

        if self.debug:
            coors_3d_dict_shift0 = flat2window(coors, voxel_info['voxel_drop_level_shift0'], flat2win_inds_list[0], self.drop_info)
            coors_2d = window2flat(coors_3d_dict_shift0, flat2win_inds_list[0])
            assert (coors_2d == coors).all()
        
        return voxel_feat, flat2win_inds_list, voxel_info


    @torch.no_grad()
    def get_flat2win_inds(self, batch_win_inds, voxel_drop_lvl):
        '''
        Args:
            batch_win_inds: shape=[N, ]. Indicates which window a voxel belongs to. Window inds is unique is the whole batch.
            voxel_drop_lvl: shape=[N, ]. Indicates batching_level of the window the voxel belongs to.
        Returns:
            flat2window_inds_dict: contains flat2window_inds of each voxel, shape=[N,]
                Determine the voxel position in range [0, num_windows * max_tokens) of each voxel.
        '''
        device = batch_win_inds.device

        flat2window_inds_dict = {}

        drop_info = self.drop_info

        for dl in drop_info:

            dl_mask = voxel_drop_lvl == dl
            if not dl_mask.any():
                continue

            conti_win_inds = self.make_continuous_inds(batch_win_inds[dl_mask])

            num_windows = len(torch.unique(conti_win_inds))
            max_tokens = drop_info[dl]['max_tokens']

            inner_win_inds = self.get_inner_win_inds(conti_win_inds)

            flat2window_inds = conti_win_inds * max_tokens + inner_win_inds

            flat2window_inds_dict[dl] = (flat2window_inds, torch.where(dl_mask))

            if self.debug:
                assert inner_win_inds.max() < max_tokens, f'Max inner inds({inner_win_inds.max()}) larger(equal) than {max_tokens}'
                assert (flat2window_inds >= 0).all()
                max_ind = flat2window_inds.max().item()
                assert  max_ind < num_windows * max_tokens, f'max_ind({max_ind}) larger than upper bound({num_windows * max_tokens})'
                assert  max_ind >= (num_windows-1) * max_tokens, f'max_ind({max_ind}) less than lower bound({(num_windows-1) * max_tokens})'

        return flat2window_inds_dict

    @torch.no_grad()
    def get_inner_win_inds(self, win_inds):
        '''
        Fast version of get_innner_win_inds_slow

        Args:
            win_inds indicates which windows a voxel belongs to. Voxels share a window have same inds.
            shape = [N,]

        Return:
            inner_inds: shape=[N,]. Indicates voxel's id in a window. if M voxels share a window, their inner_inds would be torch.arange(M, dtype=torch.long)

        Note that this function might output different results from get_inner_win_inds_slow due to the unstable pytorch sort.
        '''

        sort_inds, order = win_inds.sort() #sort_inds is like [0,0,0, 1, 2,2] -> [0,1, 2, 0, 0, 1]
        roll_inds_left = torch.roll(sort_inds, -1) # [0,0, 1, 2,2,0]

        diff = sort_inds - roll_inds_left #[0, 0, -1, -1, 0, 2]
        end_pos_mask = diff != 0

        bincount = torch.bincount(win_inds)
        # assert bincount.max() <= max_tokens
        unique_sort_inds, _ = torch.sort(torch.unique(win_inds))
        num_tokens_each_win = bincount[unique_sort_inds] #[3, 1, 2]

        template = torch.ones_like(win_inds) #[1,1,1, 1, 1,1]
        template[end_pos_mask] = (num_tokens_each_win-1) * -1 #[1,1,-2, 0, 1,-1]

        inner_inds = torch.cumsum(template, 0) #[1,2,0, 0, 1,0]
        inner_inds[end_pos_mask] = num_tokens_each_win #[1,2,3, 1, 1,2]
        inner_inds -= 1 #[0,1,2, 0, 0,1]


        #recover the order
        inner_inds_reorder = -torch.ones_like(win_inds)
        inner_inds_reorder[order] = inner_inds

        ##sanity check
        if self.debug:
            assert (inner_inds >= 0).all()
            assert (inner_inds == 0).sum() == len(unique_sort_inds)
            assert (num_tokens_each_win > 0).all()
            random_win = unique_sort_inds[random.randint(0, len(unique_sort_inds)-1)]
            random_mask = win_inds == random_win
            num_voxel_this_win = bincount[random_win].item()
            random_inner_inds = inner_inds_reorder[random_mask] 

            assert len(torch.unique(random_inner_inds)) == num_voxel_this_win
            assert random_inner_inds.max() == num_voxel_this_win - 1
            assert random_inner_inds.min() == 0

        return inner_inds_reorder
    
    def get_inner_win_inds_slow(self, win_inds):
        unique_win_inds = torch.unique(win_inds)
        inner_inds = -torch.ones_like(win_inds)
        for ind in unique_win_inds:
            mask = win_inds == ind
            num = mask.sum().item()
            inner_inds[mask] = torch.arange(num, dtype=win_inds.dtype, device=win_inds.device)
        assert (inner_inds >= 0).all()
        return inner_inds
        
    
    def drop_single_shift(self, batch_win_inds):
        drop_info = self.drop_info
        drop_lvl_per_voxel = -torch.ones_like(batch_win_inds)
        inner_win_inds = self.get_inner_win_inds(batch_win_inds)
        bincount = torch.bincount(batch_win_inds)
        num_per_voxel_before_drop = bincount[batch_win_inds] #
        target_num_per_voxel = torch.zeros_like(batch_win_inds)

        for dl in drop_info:
            max_tokens = drop_info[dl]['max_tokens']
            lower, upper = drop_info[dl]['drop_range']
            range_mask = (num_per_voxel_before_drop >= lower) & (num_per_voxel_before_drop < upper)
            target_num_per_voxel[range_mask] = max_tokens
            drop_lvl_per_voxel[range_mask] = dl
        
        if self.debug:
            assert (target_num_per_voxel > 0).all()
            assert (drop_lvl_per_voxel >= 0).all()

        keep_mask = inner_win_inds < target_num_per_voxel
        return keep_mask, drop_lvl_per_voxel

    @torch.no_grad()
    def get_voxel_keep_inds(self, voxel_info, num_shifts):
        '''
        To make it clear and easy to follow, we do not use loop to process two shifts.
        '''

        batch_win_inds_s0 = voxel_info['batch_win_inds_shift0']
        num_all_voxel = batch_win_inds_s0.shape[0]

        voxel_keep_inds = torch.arange(num_all_voxel, device=batch_win_inds_s0.device, dtype=torch.long)

        keep_mask_s0, drop_lvl_s0 = self.drop_single_shift(batch_win_inds_s0)
        if self.debug:
            assert (drop_lvl_s0 >= 0).all()

        drop_lvl_s0 = drop_lvl_s0[keep_mask_s0]
        voxel_keep_inds = voxel_keep_inds[keep_mask_s0]
        batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s0]

        if num_shifts == 1:
            voxel_info['voxel_keep_inds'] = voxel_keep_inds
            voxel_info['voxel_drop_level_shift0'] = drop_lvl_s0
            voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
            return voxel_info

        batch_win_inds_s1 = voxel_info['batch_win_inds_shift1']
        batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s0]

        keep_mask_s1, drop_lvl_s1 = self.drop_single_shift(batch_win_inds_s1)
        if self.debug:
            assert (drop_lvl_s1 >= 0).all()

        # drop data in first shift again
        drop_lvl_s0 = drop_lvl_s0[keep_mask_s1]
        voxel_keep_inds = voxel_keep_inds[keep_mask_s1]
        batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s1]

        drop_lvl_s1 = drop_lvl_s1[keep_mask_s1]
        batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s1]

        voxel_info['voxel_keep_inds'] = voxel_keep_inds
        voxel_info['voxel_drop_level_shift0'] = drop_lvl_s0
        voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
        voxel_info['voxel_drop_level_shift1'] = drop_lvl_s1
        voxel_info['batch_win_inds_shift1'] = batch_win_inds_s1
        ### sanity check
        if self.debug:
            for dl in self.drop_info:
                max_tokens = self.drop_info[dl]['max_tokens']

                mask_s0 = drop_lvl_s0 == dl
                if not mask_s0.any():
                    print(f'No voxel belongs to drop_level:{dl} in shift 0')
                    continue
                real_max = torch.bincount(batch_win_inds_s0[mask_s0]).max()
                assert real_max <= max_tokens, f'real_max({real_max}) > {max_tokens} in shift0'

                mask_s1 = drop_lvl_s1 == dl
                if not mask_s1.any():
                    print(f'No voxel belongs to drop_level:{dl} in shift 1')
                    continue
                real_max = torch.bincount(batch_win_inds_s1[mask_s1]).max()
                assert real_max <= max_tokens, f'real_max({real_max}) > {max_tokens} in shift1'
        ###
        return voxel_info

    @torch.no_grad()
    def window_partition(self, coors, voxel_info):

        shifts_list = self.shifts_list
        win_shape_x, win_shape_y = self.window_shape
        pc_range = self.point_cloud_range
        voxel_size = self.voxel_size # using the min voxel size
        assert isinstance(voxel_size, tuple)

        bev_shape_x = int(np.ceil((pc_range[3] - pc_range[0])/voxel_size[0]))
        bev_shape_y = int(np.ceil((pc_range[4] - pc_range[1])/voxel_size[1]))

        max_num_win_x = int(np.ceil((bev_shape_x / win_shape_x)) + 1) # plus one here to meet the needs of shift.
        max_num_win_y = int(np.ceil((bev_shape_y / win_shape_y)) + 1) # plus one here to meet the needs of shift.
        max_num_win_per_sample = max_num_win_x * max_num_win_y

        for i in range(len(shifts_list)):
            shift_x, shift_y = shifts_list[i]
            assert shift_x == 0 or shift_x == win_shape_x // 2, 'Usually ...'
            shifted_coors_x = coors[:, 3] + (win_shape_x - shift_x if shift_x > 0 else 0)
            shifted_coors_y = coors[:, 2] + (win_shape_y - shift_y if shift_y > 0 else 0)

            win_coors_x = shifted_coors_x // win_shape_x
            win_coors_y = shifted_coors_y // win_shape_y
            batch_win_inds = coors[:, 0] * max_num_win_per_sample + win_coors_x * max_num_win_y + win_coors_y
            voxel_info[f'batch_win_inds_shift{i}'] = batch_win_inds

            coors_in_win_x = shifted_coors_x % win_shape_x
            coors_in_win_y = shifted_coors_y % win_shape_y
            voxel_info[f'coors_in_win_shift{i}'] = torch.stack([coors_in_win_x, coors_in_win_y], dim=-1)
        
        return voxel_info

    @torch.no_grad()
    def make_continuous_inds(self, inds):
        '''
        Make batch_win_inds continuous, e.g., [1, 3, 4, 6, 10] -> [0, 1, 2, 3, 4].
        '''

        dtype = inds.dtype
        device = inds.device

        unique_inds, _ = torch.sort(torch.unique(inds))
        num_valid_inds = len(unique_inds)
        max_origin_inds = unique_inds.max().item()
        canvas = -torch.ones((max_origin_inds+1,), dtype=dtype, device=device)
        canvas[unique_inds] = torch.arange(num_valid_inds, dtype=dtype, device=device)

        conti_inds = canvas[inds]

        if self.debug:
            assert conti_inds.max() == len(torch.unique(conti_inds)) - 1, 'Continuity check failed.'
            assert conti_inds.min() == 0, '-1 in canvas should not be indexed.'
        return conti_inds

    def set_drop_info(self):
        if hasattr(self, 'drop_info'):
            return
        meta = self.meta_drop_info
        if isinstance(meta, tuple):
            if self.training:
                self.drop_info = meta[0]
            else:
                self.drop_info = meta[1]
        else:
            self.drop_info = meta
        print(f'drop_info is set to {self.drop_info}, in input_layer')