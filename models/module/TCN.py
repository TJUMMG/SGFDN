import torch
import torch.nn as nn
import torch.nn.functional as F
from models.module.attention import RelevanceFilter, DynamicPyramidAttentionLayerDecomposed, DynamicPyramidAttentionLayer
from models.utils import temporal_separate_to_stack, temporal_stacked_to_separate
import torchvision.ops as ops


class TCN(nn.Module):
    def __init__(self, text_dim, inchannel, hidden_channel, outchannel, layers=8, padding_type='circle', groups=8, dropout=0.1, decompose=True, decompose_type='weighted'):
        super(TCN, self).__init__()
        self.padding_type = padding_type
        self.decompose_type = decompose_type
        self.conv_time = nn.ModuleList()
        self.conv_spatial = nn.ModuleList()
        self.conv_convert = nn.ModuleList()
        ## [current]
        self.dropout1 = nn.ModuleList()
        self.dropout2 = nn.ModuleList()
        ## [current]
        self.dilations = []
        self.local_attention = nn.ModuleList()
        for i in range(layers):
            dilation = torch.pow(torch.tensor(2), i)
            dilation = int(dilation)
            self.dilations.append(dilation)
            if decompose:
                # self.local_attention.append(DynamicPyramidAttentionLayerDecomposed(text_dim, inchannel, inchannel, decompose_type=decompose_type, kernels=[(1, 1)], dilations=[(1, 1)]))
                self.local_attention.append(DynamicPyramidAttentionLayerDecomposed(text_dim, inchannel, inchannel, decompose_type=decompose_type))
            else:
                self.local_attention.append(DynamicPyramidAttentionLayer(text_dim, inchannel, inchannel))

            # self.conv_spatial.append(
            #     nn.Sequential(
            #         nn.Conv2d(inchannel, hidden_channel, 3, 1, 1, bias=False),
            #         nn.GroupNorm(4, hidden_channel),
            #         nn.ReLU(inplace=True)
            #     )
            # )
            self.conv_spatial.append(
                nn.Sequential(
                    nn.Conv3d(inchannel, hidden_channel, (1, 3, 3), 1, (0, 1, 1), bias=False),
                    nn.GroupNorm(4, hidden_channel),
                    nn.ReLU(inplace=True)
                )
            )
            
            self.conv_time.append(
                nn.Sequential(
                    nn.Conv3d(hidden_channel, hidden_channel, (3, 1, 1), (1, 1, 1), (0, 0, 0), (dilation, 1, 1), bias=False),
                    nn.GroupNorm(4, hidden_channel),
                    nn.ReLU(inplace=True)
                )
            )
            
            self.conv_convert.append(
                nn.Sequential(
                    nn.Conv3d(hidden_channel, outchannel, 1, 1, bias=False)
                )
            )
            self.dropout1.append(nn.Dropout(dropout))
            self.dropout2.append(nn.Dropout(dropout))
        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, fea, fea_text, frame_mask, durations):
        maps_spatial = []
        for i in range(len(self.conv_time)):
            fea1, maps = self.local_attention[i](fea, fea_text, frame_mask)
            maps = temporal_separate_to_stack(maps.transpose(1, 2), durations)
            maps_spatial.append(maps)
            fea = fea + self.dropout1[i](fea1)
            fea2 = self.conv_spatial[i](fea)
            
            if self.padding_type == 'circle':
                fea2 = circle_padding(self.dilations[i], fea2)
            elif self.padding_type == 'zero':
                fea2 = F.pad(fea2, (0, 0, 0, 0, self.dilations[i], self.dilations[i]), mode='constant', value=0)
            else:
                fea2 = F.pad(fea2, (0, 0, 0, 0, self.dilations[i], self.dilations[i]), mode='circular')
            fea2 = self.conv_time[i](fea2)  # B*C*T
            fea2 = self.conv_convert[i](fea2)
            fea = fea + self.dropout2[i](fea2)
        return fea, maps_spatial


def circle_padding(padding, feature):
    length_times = feature.shape[2]
    index = list(range(0, length_times)) + list(range(length_times - 2, 0, -1))
    total_num = 2 * padding + length_times
    num_c = padding // len(index)
    if num_c * len(index) < padding:
        num_c = num_c + 1
    expand_number = num_c * len(index) - padding
    index_f = []
    for n in range(num_c):
        index = index + index + index
    for i in range(expand_number, expand_number + total_num):
        index_f.append(index[i])

    feas = []
    for idf in index_f:
        feas.append(feature[:, :, idf, :, :].unsqueeze(2))
    feas = torch.cat(feas, dim=2)
    return feas
