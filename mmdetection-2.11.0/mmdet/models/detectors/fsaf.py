from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class FSAF(SingleStageDetector):
    """Implementation of `FSAF <https://arxiv.org/abs/1903.00621>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FSAF, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)
