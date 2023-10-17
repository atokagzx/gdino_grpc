import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass
class Mask:
    mask: np.ndarray
    confidence: float  

@dataclass
class DetectedItem:
    bounding_box: Tuple[int, int, int, int]
    label: str
    detection_confidence: float
 
@dataclass
class SegmentedItem(DetectedItem):
    masks: Tuple[Mask]