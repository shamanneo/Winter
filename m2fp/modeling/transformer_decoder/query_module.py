import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from .prior_module import PriorModule
from .attention_module import AttentionModule


class QueryModule(nn.Module) :  

    def __init__(self, num_queries, num_features) :
        super(QueryModule, self).__init__()
        self.num_queries = num_queries
        self.num_features = num_features

        self.prior_module = PriorModule(num_queries, num_features)
        self.attention_module = AttentionModule(num_queries, num_features)

    def forward(self, x, mask_features, query_feat, query_embed) :
        """
        Args: 
            x: multi scale features
        """
        object_prior = self.prior_module(x, mask_features)
        object_query = self.attention_module(object_prior, query_feat)
        output = self.attention_module(object_prior, object_query + query_embed)
        return output
        
        




        





