import torch
import torch.nn.functional as F
import torch.nn as nn
from .anchor_utils import generate_proposals, generate_scores, generate_2d_gaussian
from einops import repeat, rearrange
from .utils import generate_anchor_scores, compute_temporal_reg_tar, segment_tiou
from .utils import generate_2d_gaussian as generate_2d_gaussian_new
import math

class SetCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.sigma = 1
        self.bce_loss = nn.BCELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def loss_spatial(self, outputs, targets, inter_idx):
        # CenterNet-based decoding
        inter_idx = inter_idx[0]
        h, w = outputs['spatial_map'].shape[-2:]
        box_gt = [targets[i]['boxes'] for i in range(len(targets))]
        box_gt = torch.cat(box_gt, dim=0)   # [k, 4], cxcywh
        size_gt = [targets[i]['size'] for i in range(len(targets))] # current input frame size
        size_gt = torch.stack(size_gt)  # [\sigma t_i, 2]
        padded_size_gt = torch.max(size_gt, dim=0)[0]    # [2]

        # generating 2D Gaussian groundtruth
        box_gt_unnormed = box_gt * torch.stack([size_gt[:, 1], size_gt[:, 0], size_gt[:, 1], size_gt[:, 0]], dim=-1)
        padded_box_gt = box_gt_unnormed / torch.stack([padded_size_gt[1], padded_size_gt[0], padded_size_gt[1], padded_size_gt[0]], dim=-1)[None]
        gaussian_gt = generate_2d_gaussian(padded_box_gt, w, h, delta=0.05)[:, None]  # [k, 1, h, w]
        wh_gt = (padded_box_gt[:, 2:] * torch.as_tensor([w, h])[None, :].to(box_gt.device))[..., None, None].repeat(1, 1, h, w)
        
        # calculating heatmap and regression loss
        pred_hm = outputs['spatial_map']    # [k, 1, h, w]
        pred_wh = outputs['spatial_wh']    # [k, 2, h, w]
        loss_hm = self.bce_loss(pred_hm, gaussian_gt)
        loss_wh = self.smooth_l1_loss(pred_wh*gaussian_gt, wh_gt*gaussian_gt)
        loss_map = 0
        for map in outputs['maps_spatial']:
            map = F.interpolate(map, (h, w), mode='bilinear', align_corners=True)
            loss_map += self.bce_loss(map, gaussian_gt)

        return {
            'spatial_hm_loss': loss_hm,
            'spatial_wh_loss': loss_wh,
            'spatial_map_loss': loss_map
        }

    def loss_temporal(self, outputs, durations, inter_idx, time_mask):
        device = outputs['spatial_map'].device
        seq_len = max(durations)
        b = len(durations)
        index = torch.as_tensor([i for i in range(seq_len)]).to(device)[None].repeat(b, 1) # [b, t]
        inter_idx_tensor = torch.as_tensor(inter_idx).float().to(device)  # [b, 2]
        inter_idx_expand = inter_idx_tensor[:, None].repeat(1, seq_len, 1) # [b, t, 2]
        action_gt = ((index >= inter_idx_expand[..., 0]) & (index <= inter_idx_expand[..., 1])).float()  # [b, t], 1 for moments when action happens, otherwise 0

        if self.cfg.temporal_decoder_type == 'anchor':
            # widely used anchor-based temporal grounding head
            pred_score = outputs['temporal_score']
            pred_offset = outputs['temporal_offset']

            proposals = generate_proposals(seq_len, self.cfg.temporal_window_width)[None].repeat(b, 1, 1).to(device)    # [b, t*n_window, 2]
            score_gt, score_mask = generate_anchor_scores(proposals, inter_idx_tensor, seq_len, self.cfg.temporal_score_thres)
            time_mask_expanded = repeat(time_mask, 'b t -> b (t n)', n=len(self.cfg.temporal_window_width))
            score_mask[time_mask_expanded] = True   # [b, t*n_window]

            pred_score = pred_score * score_mask.float()
            score_pos = (score_gt >= self.cfg.temporal_valid_thres).float()
            score_pos = score_pos.masked_fill(time_mask_expanded, 0.)
            reg_gt = inter_idx_tensor[:, None].repeat(1, proposals.shape[1], 1)
            refined_box = pred_offset + proposals   # [b, t*n_window, 2]
            loss_reg = self.smooth_l1_loss(refined_box*score_pos[..., None], reg_gt*score_pos[..., None])
            loss_cls = self.bce_loss(pred_score.masked_fill(time_mask_expanded, 0.), 
                                            score_gt.masked_fill(time_mask_expanded, 0.))
            result_dict = {
            'temporal_cls_loss': loss_cls,
            'temporal_align_loss': loss_reg}
        
        elif self.cfg.temporal_decoder_type == 'regression':
            # from "Dense Regression Network for Video Grounding" CVPR2020
            
            pred_start = index - outputs['temporal_reg'][:, :, 0]
            pred_end = index + outputs['temporal_reg'][:, :, 1]
            predictions = torch.stack([pred_start, pred_end], dim=-1) / seq_len
            predictions = torch.clamp(predictions, 0, 1)
            label_reg = compute_temporal_reg_tar(inter_idx_tensor, action_gt)
            label_iou = segment_tiou(predictions, inter_idx_tensor[:, None] / seq_len)
            iou_pos_ind = label_iou > 0.5
            pos_iou_target = label_iou[iou_pos_ind]
            pos_iou_pred = outputs['temporal_iou'][iou_pos_ind]
            loss_reg = self.smooth_l1_loss(
                outputs['temporal_reg'] * action_gt.unsqueeze(-1), label_reg)
            loss_score = self.bce_loss(outputs['temporal_score'], action_gt)
            if iou_pos_ind.sum().item() == 0:
                loss_iou = 0
            else:
                loss_iou = self.bce_loss(pos_iou_pred, pos_iou_target.detach())
            result_dict = {
            'temporal_score_loss': loss_score,
            'temporal_reg_loss': loss_reg,
            'temporal_iou_loss': loss_iou
            }
        
        elif self.cfg.temporal_decoder_type == 'distribution':
            # from "TubeDETR: Spatio-Temporal Video Grounding with Transformers" CVPR2022
            sted = outputs["temporal_dist"]
            # print(torch.isnan(sted).any())
            target_start = torch.tensor([x[0] for x in inter_idx], dtype=torch.long).to(
                sted.device
            )
            target_end = torch.tensor([x[1] for x in inter_idx], dtype=torch.long).to(
                sted.device
            )
            sted = sted.masked_fill(
                time_mask[:, :, None], -1e32
            )  # put very low probability on the padded positions before softmax
            eps = 1e-6  # avoid log(0) and division by 0

            sigma = self.sigma
            start_distrib = (
                -(
                    (
                        torch.arange(sted.shape[1])[None, :].to(sted.device)
                        - target_start[:, None]
                    )
                    ** 2
                )
                / (2 * sigma ** 2)
            ).exp()  # gaussian target
            start_distrib = F.normalize(start_distrib + eps, p=1, dim=1)
            pred_start_prob = (sted[:, :, 0]).softmax(1)
            loss_start = (
                pred_start_prob * ((pred_start_prob + eps) / start_distrib).log()
            )  # KL div loss
            loss_start = loss_start * (~time_mask)  # not count padded values in the loss

            end_distrib = (
                -(
                    (
                        torch.arange(sted.shape[1])[None, :].to(sted.device)
                        - target_end[:, None]
                    )
                    ** 2
                )
                / (2 * sigma ** 2)
            ).exp()  # gaussian target
            end_distrib = F.normalize(end_distrib + eps, p=1, dim=1)
            pred_end_prob = (sted[:, :, 1]).softmax(1)
            loss_end = (
                pred_end_prob * ((pred_end_prob + eps) / end_distrib).log()
            )  # KL div loss
            loss_end = loss_end * (~time_mask)  # do not count padded values in the loss

            loss_sted = loss_start + loss_end
            result_dict = {
               'temporal_dist_loss': loss_sted.mean()
            }
        else:
            raise NotImplementedError
        # loss_map = 0.0
        #for map in outputs['maps_temporal']:
         #   loss_map += self.bce_loss(map.squeeze(1).masked_fill(time_mask, 0.), action_gt.masked_fill(time_mask, 0.))
        #result_dict.update(
         #  {'temporal_map_loss': loss_map}
       # )#
        return result_dict

    def forward(self, outputs, durations, inter_idx, targets, time_mask):
        loss_dict = self.loss_temporal(outputs, durations, inter_idx, time_mask)
        loss_dict_s = self.loss_spatial(outputs, targets, inter_idx)
        loss_dict.update(loss_dict_s)
        return loss_dict

def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

        
