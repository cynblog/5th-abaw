import os
from easydict import EasyDict
import numpy as np


_C = EasyDict()
cfg = _C

_C.save_folder = '/data02/mxy/CVPR_ABAW/precode/output/1'

_C.data_root = '/data02/mxy/CVPR_ABAW/dataset/cropped_aligned'
_C.label_root = '/data02/mxy/CVPR_ABAW/dataset/VA_Estimation_Challenge'
_C.result_folder = '/data02/mxy/CVPR_ABAW/precode/result'
