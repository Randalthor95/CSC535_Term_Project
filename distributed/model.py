'''
    Network structure taken from example:
    https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8
'''

feature_dim = 3243 # should be 3243 for the number of queries

from torch_geometric.nn import TopKPooling, SAGEConv, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import torch
import torch_geometric
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2

class GraphNetV1(torch.nn.Module):
    def __init__(self, convs=[(100,100), (100,100)], lin=[(100, 10), (10, 1)]):
        super(GraphNetV1, self).__init__()
        self.convolutions=torch.nn.ModuleList([GCNConv(*c) for c in convs])
        self.linear_layers=torch.nn.ModuleList([torch.nn.Linear(*l) for l in lin])
        self.activations=torch.nn.ModuleList([torch.nn.ReLU() for n in lin])
        
  
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Normalize values?
        nm = torch.norm(x).detach()
        x = x.div(nm.expand_as(x))
                
        x = torch.tensor(
            SelectKBest(chi2, k=100).fit_transform(data.x, data.current_y.t().squeeze()),
            dtype=torch.float
        )
        # Run graph convolutions
        for gcon in self.convolutions:
            x = gcon(x, edge_index)
        
        # Run linear layers
        i = 0
        for lin, act in zip(self.linear_layers, self.activations):
            x = lin(x)
            if i < len(self.linear_layers)-1:
                x = act(x)
            else:
                break
            i = i + 1
        
        # Transpose x on the return?
        return x.t()