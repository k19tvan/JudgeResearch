"""D-FINE Mini: A minimal but complete D-FINE object detection pipeline."""

from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .iou import box_area, box_iou, generalized_box_iou
from .losses import sigmoid_focal_loss, varifocal_loss
from .matcher import HungarianMatcher
from .criterion import SetCriterion
from .backbone import HGNetV2Stem
from .positional_encoding import PositionEmbeddingSine2D
from .attention import MultiHeadAttention
from .encoder import TransformerEncoderLayer
from .decoder import TransformerDecoderLayer
from .model import DFINEMini
from .train import make_synthetic_batch, train_one_epoch, evaluate

__version__ = "1.0.0"
__all__ = [
    "box_cxcywh_to_xyxy",
    "box_xyxy_to_cxcywh",
    "box_area",
    "box_iou",
    "generalized_box_iou",
    "sigmoid_focal_loss",
    "varifocal_loss",
    "HungarianMatcher",
    "SetCriterion",
    "HGNetV2Stem",
    "PositionEmbeddingSine2D",
    "MultiHeadAttention",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "DFINEMini",
    "make_synthetic_batch",
    "train_one_epoch",
    "evaluate",
]
