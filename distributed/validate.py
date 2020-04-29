import matplotlib.pyplot as plt

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

if __name__ == "__main__":

    validation_loader = DataLoader(valid_data, batch_size=1)
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