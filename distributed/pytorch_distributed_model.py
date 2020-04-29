import sys
import torch
import torch_geometric
import numpy as np

import torch.distributed as dist
from torch.multiprocessing import Process

import matplotlib.pyplot as plt

# Model shit
from torch_geometric.nn import TopKPooling, SAGEConv, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from sklearn.feature_selection import SelectKBest, chi2

# Data loader shit
import os.path as osp
import os
from datetime import datetime, timedelta
from torch_geometric.data import Data, Dataset, DataLoader


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

'''
    Network structure taken from example:
    https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8
'''

feature_dim = 3243 # should be 3243 for the number of queries

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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, model, criterion, optimizer, epoch):
    # losses = AverageMeter()
    device = torch.device('cpu')
    # switch to train mode
    model.train()
    loss_meter = AverageMeter()
    for data in train_loader:
        # Create non_blocking tensors for distributed training
        # input = data.cuda(device=device, non_blocking=True)
        # target = data.y.cuda(device=device, non_blocking=True)
        input = data.to(device)
        target = data.y.to(device)
        # compute output
        output = model(input).reshape((10, 51))
        loss = criterion(output, target)
        # losses.update(loss.item(), data.batch)
        loss_meter.update(loss)
        # compute gradients in a backward pass
        optimizer.zero_grad()
        loss.backward()
        # Call step of optimizer to update model params
        optimizer.step()       
    print("Epoch complete, loss:", loss_meter.avg)

def validate(validation_loader, model):
    i = 0
    device = torch.device('cpu')
    model.to(device)
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
        if i % 5 == 0:
            plt.savefig('graph_results-{}.png'.format(i))
        i = i + 1

def init_process(rank, world_size, backend='gloo'):
    """ Initialize the distributed environment. """
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '29500'
    os.environ['MASTER_ADDR'] = 'olympia.cs.colostate.edu'
    os.environ['MASTER_PORT'] = '60021'

    dist.init_process_group(backend, world_size=world_size, rank=rank)
    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    # torch.manual_seed(42)

    print('Model initiated')

    dataset = COVIDSearchTerms('..')
    train_data, valid_data = dataset[:50], dataset[50:]

    model = GraphNetV1(
        convs=[],
        lin=[(100, 100), (100, 100), (100, 75), (75, 50), 
            (50, 50), (50, 25), (25, 10), (10, 5), (5, 1)]
    )

    model = torch.nn.parallel.DistributedDataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    criterion = torch.nn.L1Loss()

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=10)
    valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=1)

    for epoch in range(5):
        print("Current epoch", epoch)
        train_sampler.set_epoch(epoch)
        train(train_loader, model, criterion, optimizer, epoch)

    if rank == 0:
        torch.save(model.state_dict, 'trained_model.tmod')
        validate(valid_loader, model)


if __name__ == "__main__":
    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])
    print('rank: {}, world_size: {}'.format(rank, world_size))
    init_process(rank, world_size)
