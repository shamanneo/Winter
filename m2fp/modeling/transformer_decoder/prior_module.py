import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class PriorModule(nn.Module) :  

    def __init__(self, num_queries, num_features) :
        super(PriorModule, self).__init__()
        self.num_queries = num_queries
        self.num_features = num_features

        self.conv_fusion = nn.Sequential(
            nn.Conv2d(num_features * 3, num_features, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(num_features)
        )
        self.conv_logit = nn.Sequential(
            nn.Conv2d(self.num_features, self.num_features, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(self.num_features, self.num_queries, kernel_size = 1, stride = 1, padding = 0)
        )

    def forward(self, x, mask_features) :
        """
        Args: 
            x: multi scale features
        """
        B, C, H, W = mask_features.size()
        x_resized = [F.interpolate(per_x, size = (H, W), mode = "bilinear", align_corners = True )for per_x in x]
        x_fused = self.conv_fusion(torch.cat(x_resized, dim = 1))
        logit = self.conv_logit(x_fused)
        logit = rearrange(logit, "b c h w -> b c (h w)")
        attn = F.softmax(logit, dim = -1)
        prior = torch.bmm(attn, rearrange(mask_features, "b c h w -> b (h w) c"))
        return prior

