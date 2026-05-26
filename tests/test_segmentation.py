import numpy as np
import torch
from src.segmentation.segmentation_metrics import SegmentationMetrics

def test_calculate_iou():
    pred = torch.tensor([0, 1, 1, 0, 1])
    target = torch.tensor([0, 1, 0, 0, 1])
    
    # intersection for class 1: 2 (indices 1, 4)
    # union for class 1: 3 (indices 1, 2, 4)
    # IoU for class 1: 2 / 3 = 0.666...
    
    metric = SegmentationMetrics(num_classes=2)
    metric.update(pred, target)
    res = metric.compute()
    
    assert np.isclose(res["iou_class_1"], 0.6666667)
