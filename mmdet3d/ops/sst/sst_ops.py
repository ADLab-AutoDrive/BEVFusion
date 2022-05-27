import torch
import random
import numpy as np
from mmdet3d.ops import spconv

def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.

    this function don't contain except handle code. so use this carefully when
    indice repeats, don't support repeat add which is supported in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret

@torch.no_grad()
def get_flat2win_inds(batch_win_inds, voxel_drop_lvl, drop_info, debug=True):
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

    for dl in drop_info: # dl: short for drop level

        dl_mask = voxel_drop_lvl == dl
        if not dl_mask.any():
            continue

        conti_win_inds = make_continuous_inds(batch_win_inds[dl_mask])

        num_windows = len(torch.unique(conti_win_inds))
        max_tokens = drop_info[dl]['max_tokens']

        inner_win_inds = get_inner_win_inds(conti_win_inds)

        flat2window_inds = conti_win_inds * max_tokens + inner_win_inds

        flat2window_inds_dict[dl] = (flat2window_inds, torch.where(dl_mask))

        if debug:
            assert inner_win_inds.max() < max_tokens, f'Max inner inds({inner_win_inds.max()}) larger(equal) than {max_tokens}'
            assert (flat2window_inds >= 0).all()
            max_ind = flat2window_inds.max().item()
            assert  max_ind < num_windows * max_tokens, f'max_ind({max_ind}) larger than upper bound({num_windows * max_tokens})'
            assert  max_ind >= (num_windows-1) * max_tokens, f'max_ind({max_ind}) less than lower bound({(num_windows-1) * max_tokens})'

    return flat2window_inds_dict


def flat2window(feat, voxel_drop_lvl, flat2win_inds_dict, drop_info):
    '''
    Args:
        feat: shape=[N, C], N is the voxel num in the batch.
        voxel_drop_lvl: shape=[N, ]. Indicates drop_level of the window the voxel belongs to.
    Returns:
        feat_3d_dict: contains feat_3d of each drop level. Shape of feat_3d is [num_windows, num_max_tokens, C].
    
    drop_info:
    {1:{'max_tokens':50, 'range':(0, 50)}, }
    '''
    dtype = feat.dtype
    device = feat.device
    feat_dim = feat.shape[-1]

    feat_3d_dict = {}

    for dl in drop_info:

        dl_mask = voxel_drop_lvl == dl
        if not dl_mask.any():
            continue

        feat_this_dl = feat[dl_mask]

        this_inds = flat2win_inds_dict[dl][0]

        max_tokens = drop_info[dl]['max_tokens']
        num_windows = (this_inds // max_tokens).max().item() + 1
        feat_3d = torch.zeros((num_windows * max_tokens, feat_dim), dtype=dtype, device=device)
        if this_inds.max() >= num_windows * max_tokens:
            from ipdb import set_trace
            set_trace()
        feat_3d[this_inds] = feat_this_dl
        feat_3d = feat_3d.reshape((num_windows, max_tokens, feat_dim))
        feat_3d_dict[dl] = feat_3d

    return feat_3d_dict

def window2flat(feat_3d_dict, inds_dict):
    flat_feat_list = []

    num_all_voxel = 0
    for dl in inds_dict:
        num_all_voxel += inds_dict[dl][0].shape[0]
    
    dtype = feat_3d_dict[list(feat_3d_dict.keys())[0]].dtype
    
    device = feat_3d_dict[list(feat_3d_dict.keys())[0]].device
    feat_dim = feat_3d_dict[list(feat_3d_dict.keys())[0]].shape[-1]

    all_flat_feat = torch.zeros((num_all_voxel, feat_dim), device=device, dtype=dtype)
    check_feat = -torch.ones((num_all_voxel,), device=device, dtype=torch.long)

    for dl in feat_3d_dict:
        feat = feat_3d_dict[dl]
        feat_dim = feat.shape[-1]
        inds, flat_pos = inds_dict[dl]
        feat = feat.reshape(-1, feat_dim)
        flat_feat = feat[inds]
        all_flat_feat[flat_pos] = flat_feat
        check_feat[flat_pos] = 0
        # flat_feat_list.append(flat_feat)
    assert (check_feat == 0).all()
    
    return all_flat_feat

def get_flat2win_inds_v2(batch_win_inds, voxel_drop_lvl, drop_info, debug=True):
    transform_dict = get_flat2win_inds(batch_win_inds, voxel_drop_lvl, drop_info, debug)
    # add voxel_drop_lvl and batching_info into transform_dict for better wrapping
    transform_dict['voxel_drop_level'] = voxel_drop_lvl
    transform_dict['batching_info'] = drop_info
    return transform_dict
    
def window2flat_v2(feat_3d_dict, inds_dict):
    inds_v1 = {k:inds_dict[k] for k in inds_dict if not isinstance(k, str)}
    return window2flat(feat_3d_dict, inds_v1)

def flat2window_v2(feat, inds_dict):
    assert 'voxel_drop_level' in inds_dict, 'voxel_drop_level should be in inds_dict in v2 function'
    inds_v1 = {k:inds_dict[k] for k in inds_dict if not isinstance(k, str)}
    batching_info = inds_dict['batching_info']
    return flat2window(feat, inds_dict['voxel_drop_level'], inds_v1, batching_info)


@torch.no_grad()
def get_inner_win_inds(win_inds):
    '''
    Args:
        win_inds indicates which windows a voxel belongs to. Voxels share a window have same inds.
        shape = [N,]
    Return:
        inner_inds: shape=[N,]. Indicates voxel's id in a window. if M voxels share a window, their inner_inds would
            be torch.arange(m, dtype=torch.long)
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

@torch.no_grad()
def get_window_coors(coors, sparse_shape, window_shape, do_shift):

    if len(window_shape) == 2:
        win_shape_x, win_shape_y = window_shape
        win_shape_z = sparse_shape[-1]
    else:
        win_shape_x, win_shape_y, win_shape_z = window_shape

    sparse_shape_x, sparse_shape_y, sparse_shape_z = sparse_shape
    assert sparse_shape_z < sparse_shape_x, 'Usually holds... in case of wrong order'

    max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)) + 1) # plus one here to meet the needs of shift.
    max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y)) + 1) # plus one here to meet the needs of shift.
    max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z)) + 1) # plus one here to meet the needs of shift.
    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

    if do_shift:
        shift_x, shift_y, shift_z = win_shape_x // 2, win_shape_y // 2, win_shape_z // 2
    else:
        shift_x, shift_y, shift_z = win_shape_x, win_shape_y, win_shape_z
    
    # compatibility between 2D window and 3D window
    if sparse_shape_z == win_shape_z:
        shift_z = 0

    shifted_coors_x = coors[:, 3] + shift_x
    shifted_coors_y = coors[:, 2] + shift_y
    shifted_coors_z = coors[:, 1] + shift_z
    
    win_coors_x = shifted_coors_x // win_shape_x
    win_coors_y = shifted_coors_y // win_shape_y
    win_coors_z = shifted_coors_z // win_shape_z
    # print(coors, 'coors')
    print(shifted_coors_z, 'shift_coors')
    print(win_coors_z, 'win_coors_z')
    # if len(window_shape) == 2:
    #     assert (win_coors_z == 0).all()

    batch_win_inds = coors[:, 0] * max_num_win_per_sample + \
                        win_coors_x * max_num_win_y * max_num_win_z + \
                        win_coors_y * max_num_win_z + \
                        win_coors_z

    coors_in_win_x = shifted_coors_x % win_shape_x
    coors_in_win_y = shifted_coors_y % win_shape_y
    coors_in_win_z = shifted_coors_z % win_shape_z
    coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)
    # coors_in_win = torch.stack([coors_in_win_x, coors_in_win_y], dim=-1)
    
    return batch_win_inds, coors_in_win

@torch.no_grad()
def make_continuous_inds(inds):

    ### make batch_win_inds continuous
    dtype = inds.dtype
    device = inds.device

    unique_inds, _ = torch.sort(torch.unique(inds))
    num_valid_inds = len(unique_inds)
    max_origin_inds = unique_inds.max().item()
    canvas = -torch.ones((max_origin_inds+1,), dtype=dtype, device=device)
    canvas[unique_inds] = torch.arange(num_valid_inds, dtype=dtype, device=device)

    conti_inds = canvas[inds]

    assert conti_inds.max() == len(torch.unique(conti_inds)) - 1, 'Continuity check failed.'
    assert conti_inds.min() == 0, '-1 in canvas should not be indexed.'
    return conti_inds

class SRATensor(object):

    def __init__(self,
                 features,
                 indices,
                 spatial_shape,
                 batch_size,
                 shuffled=False,
                 ):
        """
        Similar to SparseConvTensor with the almost same interfaces.
        """
        if indices.dtype != torch.int64:
            indices = indices.long()
        self._features = features
        self._indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.reusable_pool = {}
        self.shuffled = shuffled
        self.window_shape = None
        self.dropped = False
        self.keep_inds = None
        self.ready = False
        self.shifted = False

        # to ensure transformation to window and transformation back appear in pair.
        self._transformed_to_window = False
        self._last_transform_key = None
        self._transform_counter = 0

    @property
    def spatial_size(self):
        return np.prod(self.spatial_shape)

    @property
    def features(self):
        return self._features

    def set_features(self, value):
        self._features = value

    def set_indices(self, value):
        self._indices = value

    @property
    def indices(self):
        return self._indices

    def get_reuse(self, key, do_shift, name, allow_missing=True):
        if key is None:
            return None
        key = key + '_shifted' if do_shift else key + '_not_shifted'

        if not allow_missing:
            assert key in self.reusable_pool 
            assert name in self.reusable_pool[key]

        if key in self.reusable_pool and name in self.reusable_pool[key]:
            return self.reusable_pool[key][name]
        return None

    def set_reuse(self, key, do_shift, name, value, allow_override=False):
        key = key + '_shifted' if do_shift else key + '_not_shifted'
        if key not in self.reusable_pool:
            self.reusable_pool[key] = {}
        if not allow_override:
            assert name not in self.reusable_pool[key]
        self.reusable_pool[key][name] = value

    def dense(self, channels_first=True):
        output_shape = [self.batch_size] + list(
            self.spatial_shape) + [self._features.shape[1]]
        res = scatter_nd(self.indices.long(), self._features, output_shape)
        if not channels_first:
            return res
        ndim = len(self.spatial_shape)
        trans_params = list(range(0, ndim + 1))
        trans_params.insert(1, ndim + 1)
        return res.permute(*trans_params).contiguous()

    @property
    def sparity(self):
        return (self.indices.shape[0] / np.prod(self.spatial_shape) /
                self.batch_size)
    
    def shuffle(self):
        assert not self.shuffled
        num_voxel = len(self._features)
        shuffle_inds = torch.randperm(num_voxel)
        self._features = self._features[shuffle_inds]
        self._indices = self._indices[shuffle_inds]
        self.shuffled = True
    
    def drop_and_partition(self, batching_info, key):
        assert not self.dropped
        # win_shape = self.window_shape

        batch_win_inds_s0, coors_in_win_s0 = self.window_partition(False)
        batch_win_inds_s1, coors_in_win_s1 = self.window_partition(True)
        voxel_keep_inds, drop_lvl_s0, drop_lvl_s1, batch_win_inds_s0, batch_win_inds_s1 = \
            self.get_voxel_keep_inds(batch_win_inds_s0, batch_win_inds_s1, batching_info)

        self.keep_inds = voxel_keep_inds
        self._features = self._features[voxel_keep_inds]
        self._indices = self._indices[voxel_keep_inds]
        coors_in_win_s0 = coors_in_win_s0[voxel_keep_inds]
        coors_in_win_s1 = coors_in_win_s1[voxel_keep_inds]
        self.dropped = True

        self.set_reuse(key, False, 'drop_level', drop_lvl_s0, allow_override=False)
        self.set_reuse(key, False, 'batch_win_inds', batch_win_inds_s0, allow_override=False)
        self.set_reuse(key, False, 'coors_in_win', coors_in_win_s0, allow_override=False)

        self.set_reuse(key, True, 'drop_level', drop_lvl_s1, allow_override=False)
        self.set_reuse(key, True, 'batch_win_inds', batch_win_inds_s1, allow_override=False)
        self.set_reuse(key, True, 'coors_in_win', coors_in_win_s1, allow_override=False)

    
    def setup(self, batching_info, key, window_shape, temperature):
        assert self.window_shape is None
        assert not self.ready
        self.window_shape = window_shape
        self.batching_info = batching_info
        self.key = key

        self.shuffle()
        self.drop_and_partition(batching_info, key)

        self.compute_and_add_transform_info(batching_info, key, False)
        self.compute_and_add_transform_info(batching_info, key, True)

        transform_info_s1 = self.get_reuse(key, False, 'transform_info', allow_missing=False)
        transform_info_s2 = self.get_reuse(key, True, 'transform_info', allow_missing=False)

        drop_lvl_s1 = self.get_reuse(key, False, 'drop_level', allow_missing=False)
        drop_lvl_s2 = self.get_reuse(key, True, 'drop_level', allow_missing=False)

        mask_s1 = self.get_key_padding_mask(transform_info_s1, drop_lvl_s1, batching_info, self._features.device)
        mask_s2 = self.get_key_padding_mask(transform_info_s2, drop_lvl_s2, batching_info, self._features.device)

        self.set_reuse(key, False, 'mask', mask_s1, False)
        self.set_reuse(key, True, 'mask', mask_s2, False)

        coors_in_win_s1 = self.get_reuse(key, False, 'coors_in_win', allow_missing=False)
        coors_in_win_s2 = self.get_reuse(key, True, 'coors_in_win', allow_missing=False)

        feat_dim = self._features.size(1)
        pos_s1 = self.get_pos_embed(transform_info_s1, coors_in_win_s1, drop_lvl_s1, batching_info, feat_dim, temperature, self._features.dtype)
        pos_s2 = self.get_pos_embed(transform_info_s2, coors_in_win_s2, drop_lvl_s2, batching_info, feat_dim, temperature, self._features.dtype)

        self.set_reuse(key, False, 'pos', pos_s1, False)
        self.set_reuse(key, True, 'pos', pos_s2, False)

        self.ready = True

    
    def window_tensor(self, do_shift):

        assert self.ready
        assert not self._transformed_to_window, 'window_tensor should not be called twice without update'
        assert self.dropped
        assert self.shuffled
        assert do_shift == (self._transform_counter % 2 == 1)

        key = self.key

        transform_info = self.get_reuse(key, do_shift, 'transform_info', False)
        drop_level = self.get_reuse(key, do_shift, 'drop_level', False)

        # def flat2window(feat, voxel_drop_lvl, flat2win_inds_dict, drop_info):
        window_tensor_dict = flat2window(self._features, drop_level, transform_info, self.batching_info)
        key_padding_mask = self.get_reuse(key, do_shift, 'mask', allow_missing=False)

        for k in window_tensor_dict:
            mask = key_padding_mask[k] #[num_win, num_token]
            win_tensor = window_tensor_dict[k] #[num_win, num_token, c]
            assert ((win_tensor.abs().sum(2) != 0) == (~mask)).all()

        self._transformed_to_window = True
        self._transform_counter += 1
        self.shifted = do_shift

        return window_tensor_dict, key_padding_mask
    
    def update(self, window_tensor_dict):
        assert self._transformed_to_window

        transform_info = self.get_reuse(self.key, self.shifted, 'transform_info', False)
        features = window2flat(window_tensor_dict, transform_info)
        assert len(features) == len(self._features)
        # assert len(indices) == len(self._indices)
        self._features = features
        # self._indices = indices
        self._transformed_to_window = False
    
    def compute_and_add_transform_info(self, batching_info, key, do_shift):
        batch_win_inds = self.get_reuse(key, do_shift, 'batch_win_inds', allow_missing=False)
        drop_level = self.get_reuse(key, do_shift, 'drop_level', allow_missing=False)
        transform_info = self.get_transform_info(batch_win_inds, drop_level, batching_info)
        self.set_reuse(key, do_shift, 'transform_info', transform_info, allow_override=False)


    @torch.no_grad()
    def get_transform_info(self, batch_win_inds, voxel_drop_lvl, drop_info):
        '''
        Args:
            feat: shape=[N, C], N is the voxel num in the batch.
            batch_win_inds: shape=[N, ]. Indicates which window a voxel belongs to. Window inds is unique is the whole batch.
            voxel_drop_lvl: shape=[N, ]. Indicates drop_level of the window the voxel belongs to.
        Returns:
            flat2window_inds_dict: contains flat2window_inds of each voxel, shape=[N,]
                Determine the voxel position in range [0, num_windows * max_tokens) of each voxel.
        '''
        device = batch_win_inds.device

        flat2window_inds_dict = {}

        for dl in drop_info:

            dl_mask = voxel_drop_lvl == dl
            if not dl_mask.any():
                continue

            conti_win_inds = make_continuous_inds(batch_win_inds[dl_mask])

            num_windows = len(torch.unique(conti_win_inds))
            max_tokens = drop_info[dl]['max_tokens']

            # flat2window_inds = self.get_flat2window_inds_single_drop_level(inds_this_dl) #shape=[N,]

            inner_win_inds = get_inner_win_inds(conti_win_inds)

            flat2window_inds = conti_win_inds * max_tokens + inner_win_inds


            flat2window_inds_dict[dl] = (flat2window_inds, torch.where(dl_mask))

            assert inner_win_inds.max() < max_tokens, f'Max inner inds({inner_win_inds.max()}) larger(equal) than {max_tokens}'
            assert (flat2window_inds >= 0).all()
            max_ind = flat2window_inds.max().item()
            assert  max_ind < num_windows * max_tokens, f'max_ind({max_ind}) larger than upper bound({num_windows * max_tokens})'
            assert  max_ind >= (num_windows-1) * max_tokens, f'max_ind({max_ind}) less than lower bound({(num_windows-1) * max_tokens})'

        return flat2window_inds_dict

    @torch.no_grad()
    def window_partition(self, do_shift):

        win_shape_x, win_shape_y, win_shape_z = self.window_shape

        sparse_shape_x, sparse_shape_y, sparse_shape_z = self.spatial_shape
        assert sparse_shape_z < sparse_shape_x

        max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)) + 1) # plus one here to meet the needs of shift.
        max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y)) + 1) # plus one here to meet the needs of shift.
        max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z)) + 1) # plus one here to meet the needs of shift.
        # max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

        max_num_win_per_sample = max_num_win_x * max_num_win_y

        if do_shift:
            shift_x, shift_y, shift_z = win_shape_x // 2, win_shape_y // 2, win_shape_z // 2
        else:
            shift_x, shift_y, shift_z = 0, 0, 0
        
        if sparse_shape_z == win_shape_z:
            shift_z = 0
        if sparse_shape_y == win_shape_y:
            shift_y = 0
        if sparse_shape_x == win_shape_x:
            shift_x = 0

        shifted_coors_x = self.indices[:, 3] + (win_shape_x - shift_x)
        shifted_coors_y = self.indices[:, 2] + (win_shape_y - shift_y)
        # shifted_coors_z = self.indices[:, 1] + (win_shape_z - shift_z)

        win_coors_x = shifted_coors_x // win_shape_x
        win_coors_y = shifted_coors_y // win_shape_y
        # win_coors_z = shifted_coors_z // win_shape_z
        # win_coors = torch.stack([self.indices[:, 0], win_coors_z, win_coors_y, win_coors_x], dim=1)

        # batch_win_inds = self.indices[:, 0] * max_num_win_per_sample + \ 
        #                  win_coors_z * (max_num_win_x * max_num_win_y) + \
        #                  win_coors_y * max_num_win_x + \
        #                  win_coors_x

        batch_win_inds = self.indices[:, 0] * max_num_win_per_sample + win_coors_x * max_num_win_y + win_coors_y

        coors_in_win_x = shifted_coors_x % win_shape_x
        coors_in_win_y = shifted_coors_y % win_shape_y
        coors_in_win = torch.stack([coors_in_win_x, coors_in_win_y], dim=-1)

        # coors_in_win_x = shifted_coors_x % win_shape_x
        # coors_in_win_y = shifted_coors_y % win_shape_y
        # coors_in_win_z = shifted_coors_z % win_shape_z
        # coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)
        
        return batch_win_inds, coors_in_win

    def drop_single_shift(self, batch_win_inds, drop_info):
        drop_lvl_per_voxel = -torch.ones_like(batch_win_inds)
        inner_win_inds = get_inner_win_inds(batch_win_inds)
        bincount = torch.bincount(batch_win_inds)
        num_per_voxel_before_drop = bincount[batch_win_inds] #
        target_num_per_voxel = torch.zeros_like(batch_win_inds)

        for dl in drop_info:
            max_tokens = drop_info[dl]['max_tokens']
            lower, upper = drop_info[dl]['drop_range']
            range_mask = (num_per_voxel_before_drop >= lower) & (num_per_voxel_before_drop < upper)
            target_num_per_voxel[range_mask] = max_tokens
            drop_lvl_per_voxel[range_mask] = dl
        
        assert (target_num_per_voxel > 0).all()
        assert (drop_lvl_per_voxel >= 0).all()

        keep_mask = inner_win_inds < target_num_per_voxel
        return keep_mask, drop_lvl_per_voxel

    @torch.no_grad()
    def get_voxel_keep_inds(self, batch_win_inds_s0, batch_win_inds_s1, drop_info):
        '''
        To make it clear and easy to follow, we do not use loop to process two shifts.
        '''

        num_all_voxel = batch_win_inds_s0.shape[0]

        voxel_keep_inds = torch.arange(num_all_voxel, device=batch_win_inds_s0.device, dtype=torch.long)

        keep_mask_s0, drop_lvl_s0 = self.drop_single_shift(batch_win_inds_s0, drop_info)

        assert (drop_lvl_s0 >= 0).all()

        drop_lvl_s0 = drop_lvl_s0[keep_mask_s0]
        voxel_keep_inds = voxel_keep_inds[keep_mask_s0]
        batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s0]

        # if num_shifts == 1:
        #     voxel_info['voxel_keep_inds'] = voxel_keep_inds
        #     voxel_info['voxel_drop_level_shift0'] = drop_lvl_s0
        #     voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
        #     return voxel_info

        batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s0]

        keep_mask_s1, drop_lvl_s1 = self.drop_single_shift(batch_win_inds_s1, drop_info)

        assert (drop_lvl_s1 >= 0).all()

        # drop data in first shift again
        drop_lvl_s0 = drop_lvl_s0[keep_mask_s1]
        voxel_keep_inds = voxel_keep_inds[keep_mask_s1]
        batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s1]

        drop_lvl_s1 = drop_lvl_s1[keep_mask_s1]
        batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s1]

        # voxel_info['voxel_keep_inds'] = voxel_keep_inds
        # voxel_info['voxel_drop_level_shift0'] = drop_lvl_s0
        # voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
        # voxel_info['voxel_drop_level_shift1'] = drop_lvl_s1
        # voxel_info['batch_win_inds_shift1'] = batch_win_inds_s1
        ### sanity check
        for dl in drop_info:
            max_tokens = drop_info[dl]['max_tokens']

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
        return voxel_keep_inds, drop_lvl_s0, drop_lvl_s1, batch_win_inds_s0, batch_win_inds_s1

    def get_key_padding_mask(self, transform_info, voxel_drop_level, batching_info, device):

        num_all_voxel = len(voxel_drop_level)
        key_padding = torch.ones((num_all_voxel, 1)).to(device).bool()

        window_key_padding_dict = flat2window(key_padding, voxel_drop_level, transform_info, batching_info)

        # logical not. True mens masked
        for key, value in window_key_padding_dict.items():
            window_key_padding_dict[key] = value.logical_not().squeeze(2)
        
        return window_key_padding_dict
    
    def position_embedding(self, do_shift):
        assert do_shift == self.shifted
        return self.get_reuse(self.key, self.shifted, 'pos', False)

    def get_pos_embed(self, transform_info, coors_in_win, voxel_drop_level, batching_info, d_model, pos_temperature, dtype):
        '''
        '''
        # [N,]

        win_x, win_y, win_z = self.window_shape

        x, y = coors_in_win[:, 0] - win_x/2, coors_in_win[:, 1] - win_y/2
        assert (x >= -win_x/2 - 1e-4).all()
        assert (x <= win_x/2-1 + 1e-4).all()

        # if self.normalize_pos:
        #     x = x / win_x * 2 * 3.1415 #[-pi, pi]
        #     y = y / win_y * 2 * 3.1415 #[-pi, pi]
        
        pos_length = d_model // 2
        # [pos_length]
        inv_freq = torch.arange(
            pos_length, dtype=torch.float32, device=coors_in_win.device)
        inv_freq = pos_temperature ** (2 * (inv_freq // 2) / pos_length)

        # [num_tokens, pos_length]
        embed_x = x[:, None] / inv_freq[None, :]
        embed_y = y[:, None] / inv_freq[None, :]

        # [num_tokens, pos_length]
        embed_x = torch.stack([embed_x[:, ::2].sin(), embed_x[:, 1::2].cos()],
                              dim=-1).flatten(1)
        embed_y = torch.stack([embed_y[:, ::2].sin(), embed_y[:, 1::2].cos()],
                              dim=-1).flatten(1)
        # [num_tokens, pos_length * 2]
        pos_embed_2d = torch.cat([embed_x, embed_y], dim=-1).to(dtype)

        window_pos_emb_dict = flat2window(
            pos_embed_2d, voxel_drop_level, transform_info, batching_info)
        
        return window_pos_emb_dict
        
class DebugSRATensor(object):

    def __init__(self,
                 features,
                 indices,
                 spatial_shape=None,
                 batch_size=None,
                 shuffled=False,
                 ):
        """
        Similar to SparseConvTensor with the almost same interfaces.
        """
        self.features = features
        self.indices = indices