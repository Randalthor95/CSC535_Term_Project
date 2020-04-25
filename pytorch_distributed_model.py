import os.path as osp
import os
import sys
from datetime import datetime, timedelta
from torch_geometric.data import Dataset
from torch_geometric.data import DataLoader
from torch.nn import LSTM
from torch_geometric.nn import TopKPooling, SAGEConv, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import torch.distributed as dist
from torch.multiprocessing import Process


class COVIDSearchTerms(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(COVIDSearchTerms, self).__init__(root, transform, pre_transform)
        self.node_files = []
        self.target_files = []

    @property
    def raw_file_names(self):
        self.node_files = ['x/' + f for f in os.listdir('raw/x/')]
        self.node_files.sort(
            key = lambda date: datetime.strptime(date.split('/')[-1].split('.')[0], '%Y-%m-%d')
        )
        # ensure that we only grab targets for dates we have
        self.target_files = [
            'y/' + f for f in
            list(set(os.listdir('raw/y/')) & set(os.listdir('raw/x/')))
        ]

        self.target_files.sort(
            key = lambda date: datetime.strptime(date.split('/')[-1].split('.')[0], '%Y-%m-%d')
        )

        return self.node_files + self.target_files

    @property
    def processed_file_names(self):
        dates = os.listdir('raw/y/')
        return dates

    def download(self):
        pass
        # Download to `self.raw_dir`.

    def process(self):
        with open('edge_list/edge_index.txt') as edge_file:
            edges = []
            for line in edge_file.readlines():
                u, v, d = line.split()
                edges.append([int(u),int(v)])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        i = 0
        for node_file in self.node_files:
            date = node_file.split('/')[-1].split('.')[0]
            week_forward = datetime.strptime(date, '%Y-%m-%d') + timedelta(weeks=1)
            x = torch.tensor(np.loadtxt('raw/' + node_file).tolist(), dtype=torch.float)
            
            with open('raw/y/' + date + '.csv') as fy:
                hk_data_arr = [int(line.split(',')[0]) for line in fy.readlines()]
            hk_data_arr = torch.tensor(hk_data_arr, dtype=torch.float).reshape([51, 1])
            
            x = torch.cat([x, hk_data_arr], dim=1)
            
            with open('raw/y/' + week_forward.strftime('%Y-%m-%d') + '.csv') as fy:
                y_arr = [int(line.split(',')[0]) for line in fy.readlines()]
            y = torch.tensor(y_arr, dtype=torch.float).reshape([1, 51])
            edge_index = edge_index
            # Read data from `raw_path`.
            data = Data(x=x, y=y, edge_index=edge_index)

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
# feature_dim = data.num_node_features # should be 3243 for the number of queries
feature_dim = 3243

#         self.lin1 = torch.nn.Linear(10, 51)
#         self.act1 = torch.nn.ReLU()
#         self.lstm = LSTM(input_size=51, hidden_size=51, num_layers=3
#         x = self.lin1(x)
#         x = self.act1(x)
#         x, _ = self.lstm(x.view(1, 51, 51))

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 6 layers down convolutons?
        self.conv1 = GCNConv(feature_dim + 1, 2000)
        self.conv2 = GCNConv(2000, 1000)
        self.conv3 = GCNConv(1000, 500)
        self.conv4 = GCNConv(500, 100)
        self.conv5 = GCNConv(100, 10)
        # Reveals a [51, 1] tensor where the 2nd dimensions is the number of cases?
        self.conv6 = GCNConv(10, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        nm = torch.norm(x).detach()
        x = x.div(nm.expand_as(x))
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv6(x, edge_index)
        

        
        return x.t()



'''
Example from pytorch distributed on aws
https://github.com/pytorch/tutorials/blob/master/beginner_source/aws_distributed_training_tutorial.py#L218
'''

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
    losses = AverageMeter()
    device = torch.device('cpu')
    # switch to train mode
    model.train()
    end = time.time()
    for data in train_loader:
        # Create non_blocking tensors for distributed training
        input = data.cuda(device=device, non_blocking=True)
        target = data.y.cuda(device=device, non_blocking=True)
        # input = input.to(device)
        # target = target.to(device)
        # compute output
        output = model(input)
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))
        # compute gradients in a backward pass
        optimizer.zero_grad()
        loss.backward()
        # Call step of optimizer to update model params
        optimizer.step()

def init_process(rank, world_size, backend='gloo'):
    """ Initialize the distributed environment. """
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '29500'
    os.environ['MASTER_ADDR'] = 'olympia.cs.colostate.edu'
    os.environ['MASTER_PORT'] = '60021'

    dist.init_process_group(backend, world_size=world_size, rank=rank)
    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)

    dataset = COVIDSearchTerms('.')
    train_data = dataset

    model = Net()
    model = torch.nn.parallel.DistributedDataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    criterion = torch.nn.L1Loss()

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = DataLoader(train_data, batch_size=1)

    for epoch in range(10):
        train_sampler.set_epoch(epoch)
        train(train_loader, model, criterion, optimizer, epoch)

'''
    pip install torch
    pip install torch_geometric
    pip install torch-sparse
'''


if __name__ == "__main__":
    rank = int(sys.args[1])
    world_size = int(sys.args[2])
    init_process(rank, world_size)
