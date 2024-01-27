import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.alpha = torch.rand(1)  # 可训练参数alpha，初始化为随机值
        self.beta = torch.rand(1)  # 可训练参数beta，初始化为随机值
        self.k = self.calculate_k(in_channels)
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=self.k, padding=(self.k // 2))

    def calculate_k(self, C, gamma=2, b=1):
        log_term = math.log2(C) / gamma
        k = abs(int((log_term + b / gamma) // 2) * 2 + 1)  # 使用公式计算k的值，确保k是奇数
        return k

    def adaptive_block(self, x):
        F_c_avg = self.avg_pool(x)
        F_c_max = self.max_pool(x)

        term1 = 0.5 * (F_c_avg + F_c_max)
        term2 = self.alpha * F_c_avg
        term3 = self.beta * F_c_max

        F_c_add = term1 + term2 + term3  # [1, 256, 1, 1]
        return F_c_add

    def get_A_C(self, x, F_c_add):
        F_c_add = torch.squeeze(F_c_add, dim=-1)   # [1, 256, 1]
        A_C = self.conv1d(F_c_add)
        A_C = torch.sigmoid(A_C)
        A_C = A_C.view(1, -1, 1, 1)  # [1, 256, 1, 1]
        return A_C

    def forward(self, x):
        F_c_add = self.adaptive_block(x)
        A_C = self.get_A_C(x, F_c_add)
        return x * A_C


class SpatialAttentionModule(nn.Module):
    def __init__(self, channel_groups, in_channels):
        super(SpatialAttentionModule, self).__init__()
        self.attention_map = torch.randn(1, in_channels, 1, 1)
        self.separation_rate = 0.5
        self.shared_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def channel_separation(self, feature_map, attention_map, separation_rate):
        C = feature_map.shape[1]
        C_im = int(C * separation_rate)
        C_im = C_im if C_im % 2 == 0 else C_im + 1
        C_subim = C - C_im

        _, important_indices = torch.flatten(attention_map).sort(descending=True)
        important_indices = important_indices[:C_im]
        important_mask = torch.zeros(C, dtype=torch.float32)
        subimportant_mask = torch.ones(C, dtype=torch.float32)
        important_mask[important_indices] = 1
        subimportant_mask[important_indices] = 0
        important_mask = important_mask.view(1, C, 1, 1)
        subimportant_mask = subimportant_mask.view(1, C, 1, 1)
        important_mask = important_mask.expand_as(feature_map)
        subimportant_mask = subimportant_mask.expand_as(feature_map)

        important_features = feature_map * important_mask
        subimportant_features = feature_map * subimportant_mask

        return important_features, subimportant_features

    def spatial_attention(self, F1, F2):
        F1_max = F1.max(dim=1, keepdim=True)[0]
        F1_avg = F1.mean(dim=1, keepdim=True)
        F2_max = F2.max(dim=1, keepdim=True)[0]
        F2_avg = F2.mean(dim=1, keepdim=True)

        F1_combined = torch.cat((F1_max, F1_avg), dim=1)
        F2_combined = torch.cat((F2_max, F2_avg), dim=1)

        A_s1 = self.shared_conv(F1_combined)
        A_s2 = self.shared_conv(F2_combined)
        A_s1 = self.sigmoid(A_s1)
        A_s2 = self.sigmoid(A_s2)

        return A_s1, A_s2

    def forward(self, x):
        important_features, subimportant_features = self.channel_separation(x, self.attention_map, self.separation_rate)
        A_s1, A_s2 = self.spatial_attention(important_features, subimportant_features)
        F1_prime = important_features * A_s1
        F2_prime = subimportant_features * A_s2
        return F1_prime + F2_prime


class HAM(nn.Module):
    def __init__(self, channels, channel_groups):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(in_channels=channels)
        self.spatial_attention = SpatialAttentionModule(channel_groups=channel_groups,in_channels=channels)

    def forward(self, x):
        F_refined = self.channel_attention(x)  # [1, 256, 64, 64]
        F_double_prime = self.spatial_attention(F_refined)
        return F_double_prime
