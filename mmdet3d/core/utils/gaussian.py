import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def gaussian_2d(shape, sigma=1):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gausian.
        K (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom,
                 radius - left:radius + right]).to(heatmap.device,
                                                   torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.5):
    """Get radius of gaussian.

    Args:
        det_size (tuple[torch.Tensor]): Size of the detection result.
        min_overlap (float): Gaussian_overlap. Defaults to 0.5.

    Returns:
        torch.Tensor: Computed radius.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def generate_guassian_depth_target(depth, stride, cam_depth_range, constant_std=None):
    B, tH, tW = depth.shape
    kernel_size = stride
    center_idx = kernel_size*kernel_size//2
    H = tH//stride
    W = tW//stride
    unfold_depth = F.unfold(depth.unsqueeze(1), kernel_size, dilation=1, padding=0, stride=stride) #B, Cxkxk, HxW
    unfold_depth = unfold_depth.view(B, -1, H, W).permute(0, 2, 3, 1).contiguous() # B, H, W, kxk
    valid_mask = (unfold_depth!=0) # BN, H, W, kxk

    valid_mask_f = valid_mask.float() # BN, H, W, kxk
    valid_num = torch.sum(valid_mask_f, dim=-1) # BN, H, W
    valid_num[valid_num==0] = 1e10
    if constant_std is None:
        mean = torch.sum(unfold_depth, dim=-1) / valid_num
        var_sum = torch.sum(((unfold_depth-mean.unsqueeze(-1))**2) * valid_mask_f, dim=-1) # BN, H, W
        std_var = torch.sqrt(var_sum/valid_num)
        std_var[valid_num==1] = 1 # set std_var to 1 when only one point in patch
    else:
        std_var = torch.ones((B, H, W), dtype=torch.float32) * constant_std

    unfold_depth[~valid_mask] = 1e10
    min_depth = torch.min(unfold_depth, dim=-1)[0] #BN, H, W
    min_depth[min_depth==1e10] = 0
    
    x = torch.arange(cam_depth_range[0], cam_depth_range[1]+1, cam_depth_range[2])
    dist = Normal(min_depth/cam_depth_range[2], std_var/cam_depth_range[2]) # BN, H, W, D
    cdfs = []
    for i in x:
        cdf = dist.cdf(i)
        cdfs.append(cdf)
    cdfs = torch.stack(cdfs, dim=-1)
    depth_dist = cdfs[..., 1:] - cdfs[...,:-1]
    return depth_dist, min_depth, std_var