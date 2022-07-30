# Sparse R-CNN: End-to-End Object Detection with Learnable Proposals

## Introduction

[ALGORITHM]

```
@article{peize2020sparse,
  title   =  {{SparseR-CNN}: End-to-End Object Detection with Learnable Proposals},
  author  =  {Peize Sun and Rufeng Zhang and Yi Jiang and Tao Kong and Chenfeng Xu and Wei Zhan and Masayoshi Tomizuka and Lei Li and Zehuan Yuan and Changhu Wang and Ping Luo},
  journal =  {arXiv preprint arXiv:2011.12450},
  year    =  {2020}
}
```

## Results and Models

| Model        | Backbone  | Style   | Lr schd | Number of Proposals |Multi-Scale| RandomCrop  | box AP  | Config | Download |
|:------------:|:---------:|:-------:|:-------:|:-------:            |:-------: |:---------:|:------:|:------:|:--------:|
| Sparse R-CNN | R-50-FPN  | pytorch | 1x      |   100               | False     |  False     |  37.9  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.log.json) |
| Sparse R-CNN | R-50-FPN  | pytorch | 3x      |   100               | True     |   False     |  42.8  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco_20201218_154234-7bc5c054.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco_20201218_154234-7bc5c054.log.json) |
| Sparse R-CNN | R-50-FPN  | pytorch | 3x      |   300               | True      |  True      |  45.0  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.log.json) |
| Sparse R-CNN | R-101-FPN | pytorch | 3x      |   100               | True      |  False     |  44.2  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco_20201223_121552-6c46c9d6.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco_20201223_121552-6c46c9d6.log.json) |
| Sparse R-CNN | R-101-FPN | pytorch | 3x      |   300               | True      |  True      |  46.2  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_023452-c23c3564.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_023452-c23c3564.log.json) |

### Notes

We observe about 0.3 AP noise especially when using ResNet-101 as the backbone.
