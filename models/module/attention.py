import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum
from torch import Tensor
from typing import List, Optional
from models.module.decompose import DynamicDecomposeBlock, DynamicComposeBlock

class GlobalTextPresentation(nn.Module):
    def __init__(self, text_dim):
        super(GlobalTextPresentation, self).__init__()
        self.W_txt = nn.Linear(text_dim, text_dim)

    def forward(self, fea_text, mask=None):
        weight_text = self.W_txt(fea_text)  # B*L*C
        if mask is not None:
            weight_text = weight_text.masked_fill(mask == 0, -1e9)
        weight_text = weight_text.softmax(dim=1)
        fea_text_global = fea_text * weight_text
        fea_text_global = fea_text_global.sum(dim=1)  # B*C
        return fea_text_global


class MuTan(nn.Module):
    def __init__(self, video_fea_dim, text_fea_dim, out_fea_dim, heads = 5):
        super(MuTan, self).__init__()

        self.heads = heads
        self.Wv = nn.ModuleList([nn.Conv2d(video_fea_dim+8, out_fea_dim, 1, 1) for i in range(heads)])
        self.Wt = nn.ModuleList([nn.Conv2d(text_fea_dim, out_fea_dim, 1, 1) for i in range(heads)])

    def forward(self, video_fea, text_fea, spatial):
        video_fea = torch.cat([video_fea, spatial], dim=1)
        fea_outs = []
        for i in range(self.heads):
            fea_v = self.Wv[i](video_fea)
            fea_v = torch.tanh(fea_v)  # B*C*H*W

            fea_t = self.Wt[i](text_fea)
            fea_t = torch.tanh(fea_t)  # B*C*1*1

            fea_out = fea_v * fea_t
            fea_outs.append(fea_out.unsqueeze(-1))
        fea_outs = torch.cat(fea_outs, dim=-1)
        fea_outs = torch.sum(fea_outs, dim=-1)
        mutan_fea = torch.tanh(fea_outs)
        mutan_fea = F.normalize(mutan_fea, dim=1)
        return mutan_fea


# class RelevanceFilter(nn.Module):
#     def __init__(self, text_fea_dim, video_fea_dim, attention_dim, groups=8, kernelsize=(1, 1, 1)):
#         super(RelevanceFilter, self).__init__()
#         assert text_fea_dim % groups == 0
#         assert attention_dim % groups == 0
#         self.groups = groups
#         self.Wv = nn.Conv3d(video_fea_dim, 2 * attention_dim, 1, 1)

#         self.Wt = nn.Linear(text_fea_dim, attention_dim *
#                             kernelsize[0] * kernelsize[1] * kernelsize[2])
#         self.kernel_size = kernelsize

#     def forward(self, video_fea, text_fea, frame_mask):

#         fea = self.Wv(video_fea)  # B*C*T*H*W
#         B, C, T, H, W = video_fea.shape
#         k, v = fea.chunk(2, dim=1)
#         kernel = self.Wt(text_fea)  # B*(C*K*K)
#         kernel = repeat(kernel, 'b (g c t h w) -> (b g) c t h w',
#                         t=self.kernel_size[0], h=self.kernel_size[1], w=self.kernel_size[2], g=self.groups)
#         k = repeat(k, 'b c t h w -> n (b c) t h w', n=1)
#         att = F.conv3d(k, kernel, padding=(
#             self.kernel_size[0]//2, self.kernel_size[1]//2, self.kernel_size[2]//2), groups=B*self.groups)
#         att = rearrange(
#             att, 'n (b g c) t h w -> (n b) g c t h w', b=B, g=self.groups)
#         active_map = att.mean(dim=1)
#         v = rearrange(v, 'b (g c) t h w -> b g c t h w', g=self.groups)
#         out = v * torch.sigmoid(att) * frame_mask.unsqueeze(1)
#         out = rearrange(out, 'b g c t h w -> b (g c) t h w')

#         maps = rearrange(active_map, 'b c t h w -> (b t) c h w')
#         frame_mask = rearrange(frame_mask, 'b c t h w -> (b t) c h w')
#         maps = maps.sigmoid() * frame_mask
#         return maps, out


class RelevanceFilter(nn.Module):
    def __init__(self, text_fea_dim, video_fea_dim, attention_dim, groups=8, kernelsize=(1, 1), dilation=(1,1), phase='3D'):
        super().__init__()
        assert phase in ['1D', '2D', '3D']
        assert text_fea_dim % groups == 0
        assert attention_dim % groups == 0
        self.phase = phase
        self.groups = groups
        self.kernel_size = kernelsize
        self.dilation = dilation
        if phase == '1D':
            assert len(kernelsize) == 1 and len(dilation) == 1
            self.Wkv = nn.Conv1d(video_fea_dim, 2*attention_dim, 1, 1)
            self.Wt = nn.Linear(text_fea_dim, attention_dim * kernelsize[0])
            self.padding = (kernelsize[0]//2)*dilation[0]
        elif phase == '2D':
            assert len(kernelsize) == 2 and len(dilation) == 2
            self.Wkv = nn.Conv2d(video_fea_dim, 2*attention_dim, 1, 1)
            self.Wt = nn.Linear(text_fea_dim, attention_dim * kernelsize[0] * kernelsize[1])
            self.padding = ((kernelsize[0]//2)*dilation[0], (kernelsize[1]//2)*dilation[1])
        elif phase =='3D':
            assert len(kernelsize) == 3 and len(dilation) == 3
            self.Wkv = nn.Conv3d(video_fea_dim, 2*attention_dim, 1, 1)
            self.Wt = nn.Linear(text_fea_dim, attention_dim * kernelsize[0] * kernelsize[1] * kernelsize[2])
            self.padding = ((kernelsize[0]//2)*dilation[0], (kernelsize[1]//2)*dilation[1], (kernelsize[2]//2)*dilation[2])
        
    def forward(self, video_fea, text_fea, masks=None):
        b = video_fea.shape[0]

        kv = self.Wkv(video_fea)
        k, v = kv.chunk(2, dim=1)
        kernel = self.Wt(text_fea)

        if self.phase == '1D':
            kernel = repeat(kernel, 'b (g c k0) -> (b g) c k0', k0=self.kernel_size[0], g=self.groups)
            k = repeat(k, 'b c l0 -> n (b c) l0', n=1)
            att = F.conv1d(k, kernel, padding=self.padding, dilation=self.dilation[0], groups=b*self.groups)
            att = rearrange(att, 'n (b g c) l0 -> (n b) g c l0', b=b, g=self.groups)
            v = rearrange(v, 'b (g c) l0 -> b g c l0', g=self.groups)
        elif self.phase == '2D':
            kernel = repeat(kernel, 'b (g c k0 k1) -> (b g) c k0 k1', k0=self.kernel_size[0], k1=self.kernel_size[1], g=self.groups)
            k = repeat(k, 'b c l0 l1 -> n (b c) l0 l1', n=1)
            att = F.conv2d(k, kernel, padding=self.padding, dilation=self.dilation, groups=b*self.groups)
            att = rearrange(att, 'n (b g c) l0 l1 -> (n b) g c l0 l1', b=b, g=self.groups)
            v = rearrange(v, 'b (g c) l0 l1 -> b g c l0 l1', g=self.groups)
        elif self.phase == '3D':
            kernel = repeat(kernel, 'b (g c k0 k1 k2) -> (b g) c k0 k1 k2', k0=self.kernel_size[0], k1=self.kernel_size[1], k2=self.kernel_size[2], g=self.groups)
            k = repeat(k, 'b c l0 l1 l2 -> n (b c) l0 l1 l2', n=1)
            att = F.conv3d(k, kernel, padding=self.padding, dilation=self.dilation, groups=b*self.groups)
            att = rearrange(att, 'n (b g c) l0 l1 l2 -> (n b) g c l0 l1 l2', b=b, g=self.groups)
            v = rearrange(v, 'b (g c) l0 l1 l2 -> b g c l0 l1 l2', g=self.groups)
        active_map = att.mean(dim=1).sigmoid()
        out = v * torch.sigmoid(att)
        out = torch.flatten(out, 1, 2)

        if masks is not None:
            out = out * masks
            active_map = active_map * masks
        return out, active_map


class DynamicPyramidAttentionLayerDecomposed(nn.Module):
    def __init__(self, 
            text_fea_dim: int, 
            video_fea_dim: int, 
            attention_dim: int,
            dilations: List = [(1, 1), (1, 1), (2, 2), (4, 4)],
            kernels: List = [(1, 1), (3, 3), (3, 3), (3, 3)],
            decompose_type = 'weighted'
        ) -> None:
        super().__init__()

        self.decompose_type = decompose_type
        self.decompose = DynamicDecomposeBlock(video_fea_dim, decompose_type)
        self.compose = DynamicComposeBlock(attention_dim)
        assert attention_dim % len(kernels) == 0

        hidden_dim_group = attention_dim // len(kernels)
        video_dim_group = video_fea_dim // len(kernels)
        text_dim_group = text_fea_dim // len(kernels)

        self.att_tempo = RelevanceFilter(text_fea_dim, video_fea_dim, attention_dim, kernelsize=(3,), dilation=(1,), phase='1D')
        self.atts_ht = nn.ModuleList()
        self.atts_wt = nn.ModuleList()
        for i in range(len(kernels)):
            self.atts_ht.append(RelevanceFilter(
                text_dim_group, video_dim_group, hidden_dim_group, kernelsize=kernels[i], dilation=dilations[i], phase='2D'))
            self.atts_wt.append(RelevanceFilter(
                text_dim_group, video_dim_group, hidden_dim_group, kernelsize=kernels[i], dilation=dilations[i], phase='2D'))

    def group_wise_relevance_filter(self, feature: Tensor, filter: Tensor, 
                                    atts: nn.ModuleList, mask: Optional[Tensor] = None):
        """
        channel-wise group filter operation on input feature
        Args:
            feature: [b, c, *]
            filter: [b, c]
            mask: [b, 1, *]
        Return:
            out_feas: [b, c, *]
            out_maps: [b, g, *]
        """
        out_feas = []
        out_maps = []
        feature_splits = feature.chunk(len(atts), dim=1)
        filter_splits = filter.chunk(len(atts), dim=1)
        for att, fea, kernel in zip(atts, feature_splits, filter_splits):
            out_fea, response_map = att(fea, kernel, mask)
            out_feas.append(out_fea)
            out_maps.append(response_map)
        return torch.cat(out_feas, dim=1), torch.cat(out_maps, dim=1)
    
    def forward(self, video_fea: Tensor, text_fea: Tensor, masks: Optional[Tensor] = None):
        """
        Args:
            video_fea: [b, c, t, h, w]
            text_fea: [b, c]
            masks[Optional]: [b, 1, t, h, w] float   
        Return:
            A_obj: [b, 1, t, h, w]
            fea: [b, c, t, h, w]
        """
        if masks is not None:
            mask_th = masks.max(dim=-1)[0]  # [b, 1, t, h]
            mask_tw = masks.max(dim=-2)[0]  # [b, 1, t, w]
            mask_t = masks.flatten(-2).max(dim=-1)[0]   # [b, 1, t]
        else:
            mask_th, mask_tw, mask_t = None, None, None
        fea_obj, fea_th, fea_tw, A_obj = self.decompose(video_fea, masks)

        fea_th, map_th = self.group_wise_relevance_filter(fea_th, text_fea, self.atts_ht, mask_th)
        fea_tw, map_tw = self.group_wise_relevance_filter(fea_tw, text_fea, self.atts_wt, mask_tw)
        if self.decompose_type == 'weighted':
            cur_obj_embedding, map_temporal = self.att_tempo(fea_obj, text_fea, mask_t)
            fea = self.compose(fea_th, fea_tw, cur_obj_embedding, A_obj, masks)
        else:
            fea = self.compose(fea_th, fea_tw, mask=masks)
        return fea, A_obj


class DynamicPyramidAttentionLayer(nn.Module):
    def __init__(self, 
            text_fea_dim: int, 
            video_fea_dim: int, 
            attention_dim: int,
            dilations: List = [(1, 1, 1), (1, 1, 1), (2, 2, 2), (4, 4, 4)],
            kernels: List = [(1, 1, 1), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
        ) -> None:
        super().__init__()
        assert attention_dim % len(kernels) == 0

        hidden_dim_group = attention_dim // len(kernels)
        video_dim_group = video_fea_dim // len(kernels)
        text_dim_group = text_fea_dim // len(kernels)
        self.atts = nn.ModuleList()
        for i in range(len(kernels)):
            self.atts.append(RelevanceFilter(text_dim_group, video_dim_group, hidden_dim_group, kernelsize=kernels[i], dilation=dilations[i], phase='3D'))

    def group_wise_relevance_filter(self, feature: Tensor, filter: Tensor, 
                                    atts: nn.ModuleList, mask: Optional[Tensor] = None):
        """
        channel-wise group filter operation on input feature
        Args:
            feature: [b, c, *]
            filter: [b, c]
            mask: [b, 1, *]
        Return:
            out_feas: [b, c, *]
            out_maps: [b, g, *]
        """
        out_feas = []
        out_maps = []
        feature_splits = feature.chunk(len(atts), dim=1)
        filter_splits = filter.chunk(len(atts), dim=1)
        for att, fea, kernel in zip(atts, feature_splits, filter_splits):
            out_fea, response_map = att(fea, kernel, mask)
            out_feas.append(out_fea)
            out_maps.append(response_map)
        return torch.cat(out_feas, dim=1), torch.cat(out_maps, dim=1).mean(dim=1, keepdim=True)
    
    def forward(self, video_fea: Tensor, text_fea: Tensor, masks: Optional[Tensor] = None):
        """
        Args:
            video_fea: [b, c, t, h, w]
            text_fea: [b, c]
            masks[Optional]: [b, 1, t, h, w] float   
        Return:
            map: [b, 1, t, h, w]
            fea: [b, c, t, h, w]
        """
        fea, map = self.group_wise_relevance_filter(video_fea, text_fea, self.atts, masks)
        return fea, map

