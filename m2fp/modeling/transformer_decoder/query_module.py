import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from .prior_module import PriorModule


class QueryModule(nn.Module) :  

    def __init__(self, num_queries, num_features) :
        super(QueryModule, self).__init__()
        self.num_queries = num_queries
        self.num_features = num_features

        self.prior_module = PriorModule(num_queries, num_features)

    def forward(self, x) :
        object_prior = self.prior_module(x)
        return object_prior
        
        




        





