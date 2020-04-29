import os.path as osp
import os
import sys
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

# from distributed import GraphNetV1, COVIDSearchTerms
from model import GraphNetV1
from dataloader import COVIDSearchTerms

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
    for data in train_loader:
        # Create non_blocking tensors for distributed training
        # input = data.cuda(device=device, non_blocking=True)
        # target = data.y.cuda(device=device, non_blocking=True)
        input = data.to(device)
        target = data.y.to(device)
        # compute output
        output = model(input)
        loss = criterion(output, target)
        # losses.update(loss.item(), data.batch)
        print(loss)
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

    print('Model initiated')

    dataset = COVIDSearchTerms('..')
    train_data, valid_data = dataset[:50], dataset[50:]


    model = GraphNetV1(
        convs=[],
        lin=[(100, 100), (100, 100), (100, 75), (75, 50), 
            (50, 50), (50, 25), (25, 10), (10, 5), (5, 1)]
    ).to(device)

    model = torch.nn.parallel.DistributedDataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    criterion = torch.nn.L1Loss()

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    train_loader = DataLoader(train_data, batch_size=1)

    for epoch in range(10):
        train_sampler.set_epoch(epoch)
        train(train_loader, model, criterion, optimizer, epoch)

    if rank == 0:
        torch.save(model, 'trained_model.tmod')

if __name__ == "__main__":
    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])
    print('rank: {}, world_size: {}'.format(rank, world_size))
    init_process(rank, world_size)
