from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DynamicDecomposeBlock(nn.Module):
    def __init__(self, 
            dim: int, decompose_type='weighted'
        ) -> None:
        super().__init__()
        self.decompose_type = decompose_type
        self.W = nn.Conv3d(dim, 1, (1, 3, 3), 1, (0, 1, 1))

    def forward(self, feature: Tensor, mask: Optional[Tensor] = None):
        """
        Args:
            feature: [b, c, t, h, w]
            mask: [b, 1, t, h, w]
        Return:
            fea_obj: [b, c, t]
            fea_th: [b, c, t, h]
            fea_tw: [b, c, t, w]
            A_obj: [b, 1, t, h, w]
        """
        b, _, t, h, w = feature.shape
        if mask is None:
            mask = torch.ones((b, 1, t, h, w)).to(feature.device)
        if self.decompose_type == 'weighted' or self.decompose_type == 'weighted_wo_group':
            A = self.W(feature)
            A_obj = A.sigmoid() * mask
            A_w = A.mean(dim=-2, keepdim=True).softmax(dim=-1)  # [b, 1, t, 1, w]
            A_h = A.mean(dim=-1, keepdim=True).softmax(dim=-2)  # [b, 1, t, h, 1]
            
            fea_th = torch.matmul(feature, A_w.transpose(-2, -1)).squeeze(-1)   # [b, c, t, h]
            fea_tw = torch.matmul(A_h.transpose(-2, -1), feature).squeeze(-2)   # [b, c, t, w]
        else:
            A = self.W(feature)
            A_obj = A.sigmoid() * mask
            fea_th = feature.mean(dim=-1)
            fea_tw = feature.mean(dim=-2)
        fea_obj = (feature * A_obj).sum((-2, -1)) / (A_obj.sum((-2, -1)) + 1e-5)   # [b, c, t]
        return fea_obj, fea_th, fea_tw, A_obj


class DynamicComposeBlock(nn.Module):
    def __init__(self, dim: int = 256, out_dim: int = 256):
        super().__init__()
        self.linear_fea_3d = nn.Conv3d(dim, out_dim, 1, 1)
        self.linear_fea_1d = nn.Conv1d(dim, out_dim, 1, 1)
    
    def forward(self, fea_th: Tensor, fea_tw: Tensor, fea_obj: Optional[Tensor] = None, heatmap: Optional[Tensor] = None, mask: Optional[Tensor] = None):
        
        out_fea = fea_th[..., None] * fea_tw[..., None, :]
        out_fea = self.linear_fea_3d(out_fea)
        if fea_obj is not None:
            fea_obj_ = self.linear_fea_1d(fea_obj)[..., None, None]
            # fea_obj_ = fea_obj[..., None, None]
            if heatmap is not None:
                out_fea = out_fea * (1 - heatmap) + fea_obj_ * heatmap
            else:
                out_fea = out_fea * fea_obj_
        if mask is not None:
            out_fea = out_fea * mask
        return out_fea
