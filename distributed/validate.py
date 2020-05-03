import matplotlib.pyplot as plt

# Data loader shit
import os.path as osp
import os
from datetime import datetime, timedelta
from torch_geometric.data import Data, Dataset, DataLoader

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

class COVIDSearchTerms(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(COVIDSearchTerms, self).__init__(root, transform, pre_transform)
        self.node_files = []
        self.target_files = []
        # self.processed_dir = '../processed'

    @property
    def raw_file_names(self):
        self.node_files = ['x/' + f for f in os.listdir('../raw/x/')]
        self.node_files.sort(
            key = lambda date: datetime.strptime(date.split('/')[-1].split('.')[0], '%Y-%m-%d')
        )
        # ensure that we only grab targets for dates we have
        self.target_files = [
            'y/' + f for f in
            list(set(os.listdir('../raw/y/')) & set(os.listdir('../raw/x/')))
        ]

        self.target_files.sort(
            key = lambda date: datetime.strptime(date.split('/')[-1].split('.')[0], '%Y-%m-%d')
        )

        return self.node_files + self.target_files

    @property
    def processed_file_names(self):
        dates = os.listdir('../raw/y/')
        return dates

    def download(self):
        pass
        # Download to `self.raw_dir`.

    def process(self):
        with open('../edge_list/edge_index.txt') as edge_file:
            edges = []
            for line in edge_file.readlines():
                u, v, d = line.split()
                edges.append([int(u),int(v)])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        i = 0
        for node_file in self.node_files:
            date = node_file.split('/')[-1].split('.')[0]
            week_forward = datetime.strptime(date, '%Y-%m-%d') + timedelta(weeks=1)
            x = torch.tensor(np.loadtxt('../raw/' + node_file).tolist(), dtype=torch.float)
            
            with open('../raw/y/' + date + '.csv') as fy:
                hk_data_arr = [int(line.split(',')[0]) for line in fy.readlines()]
            hk_data_arr = torch.tensor(hk_data_arr, dtype=torch.float).reshape([51, 1])
            
            x = torch.cat([x, hk_data_arr], dim=1)
            
            with open('../raw/y/' + week_forward.strftime('%Y-%m-%d') + '.csv') as fy:
                y_arr = [int(line.split(',')[0]) for line in fy.readlines()]
            y = torch.tensor(y_arr, dtype=torch.float).reshape([1, 51])
            edge_index = edge_index
            # Read data from `raw_path`.
            data = Data(x=x, y=y, edge_index=edge_index)
            data.current_y = hk_data_arr

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, osp.join(self.processed_dir, 'data-{}.pt'.format(i)))
            i = i + 1

    def len(self):
        return 57

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data-{}.pt'.format(idx)))
        return data

if __name__ == "__main__":

    dataset = COVIDSearchTerms('.')
    valid_data = dataset[50:]
    validation_loader = DataLoader(valid_data, batch_size=1)
    device = torch.device('cpu')
    model = GraphNetV1(
        convs=[],
        lin=[(100, 100), (100, 100), (100, 75), (75, 50), 
            (50, 50), (50, 25), (25, 10), (10, 5), (5, 1)]
    )
    state_dict = torch.load('trained_model.tmod')
    model.load_state_dict(state_dict)
    model.eval()

    for data in validation_loader:
        output = model(data).detach().numpy().squeeze()
        label = data.y.to(device).numpy().squeeze()
        top10_act = (-label).argsort()[:10]
        top10_pred = (-output).argsort()[:10]
        print(np.intersect1d(top10_act, top10_pred).shape)
        plt.scatter(np.arange(51), label, c="r", label="actual")
        plt.scatter(np.arange(51), output, c="b", label="predicted")
        plt.xlabel("States")
        plt.ylabel("Number of Cases a Week Later")
        plt.legend(loc="lower left")
        plt.show()