
import torch
import torch.nn.functional as F

from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import os.path as osp

import pytorch_lightning as pl

import random
from torch_geometric.data import Data
import re
from itertools import chain


class PairData(Data):  # pragma: no cover
    def __inc__(self, key, value, *args):
        if bool(re.search('index_s', key)):
            return self.x_s.size(0)
        if bool(re.search('index_t', key)):
            return self.x_t.size(0)
        else:
            return 0

class PairDataset(torch.utils.data.Dataset):
    r"""Combines two datasets, a source dataset and a target dataset, by
    building valid pairs between separate dataset examples.
    A pair is valid if each node class in the source graph also exists in the
    target graph.
    Args:
        dataset_s (torch.utils.data.Dataset): The source dataset.
        dataset_t (torch.utils.data.Dataset): The target dataset.
        sample (bool, optional): If set to :obj:`True`, will sample exactly
            one target example for every source example instead of holding the
            product of all source and target examples. (default: :obj:`False`)
    """
    def __init__(self, dataset_s, dataset_t, sample=False):
        self.dataset_s = dataset_s # equals length, each is a PascalVOC (or WILLOW) instance
        self.dataset_t = dataset_t # equals length
        self.sample = sample # True
        self.pairs, self.cumdeg = self.__compute_pairs__()

    def __compute_pairs__(self):
        num_classes = 0 # = max(so_keypoints each graph)
        for data in chain(self.dataset_s, self.dataset_t):
            num_classes = max(num_classes, data.y.max().item() + 1)

        y_s = torch.zeros((len(self.dataset_s), num_classes), dtype=torch.bool) # 400 x 10
        y_t = torch.zeros((len(self.dataset_t), num_classes), dtype=torch.bool)

        for i, data in enumerate(self.dataset_s):
            y_s[i, data.y] = 1
        for i, data in enumerate(self.dataset_t):
            y_t[i, data.y] = 1
            
        y_s = y_s.view(len(self.dataset_s), 1, num_classes) # 400 x 1 x 10
        y_t = y_t.view(1, len(self.dataset_t), num_classes) # 1 x 400 x 10

        pairs = ((y_s * y_t).sum(dim=-1) == y_s.sum(dim=-1)).nonzero() # n x 2
        
        cumdeg = pairs[:, 0].bincount().cumsum(dim=0)
        
        return pairs.tolist(), [0] + cumdeg.tolist()


    def __len__(self):
        return len(self.dataset_s) if self.sample else len(self.pairs)

    def __getitem__(self, idx):
        if self.sample:
            data_s = self.dataset_s[idx]
            i = random.randint(self.cumdeg[idx], self.cumdeg[idx + 1] - 1)
            data_t = self.dataset_t[self.pairs[i][1]]
        else:
            data_s = self.dataset_s[self.pairs[idx][0]]
            data_t = self.dataset_t[self.pairs[idx][1]]

        y = data_s.y.new_full((data_t.y.max().item() + 1, ), -1)
        y[data_t.y] = torch.arange(data_t.num_nodes)
        y = y[data_s.y]

        return PairData(
            x_s=data_s.x,
            edge_index_s=data_s.edge_index,
            edge_attr_s=data_s.edge_attr,
            name_s = data_s.name,
            x_t=data_t.x,
            edge_index_t=data_t.edge_index,
            edge_attr_t=data_t.edge_attr,
            name_t = data_t.name,
            y=y,
            num_nodes=None,
        )

    def __repr__(self):
        return '{}({}, {}, sample={})'.format(self.__class__.__name__,
                                              self.dataset_s, self.dataset_t,
                                              self.sample)


class PascalVOCModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = cfg.dataset.datadir
        self.batch_size = cfg.train.batch_size
        self.batch_size_test = cfg.train.batch_size_test
        self.transform = T.Compose([
                        T.Delaunay(),
                        T.FaceToEdge(),
                        T.Cartesian(),
                    ])
        
        self.pre_filter = lambda data: data.pos.size(0) > 0
        self.pre_setup()
        self.setup()

    def pre_setup(self):
        self.pascal_test__ = list()
        
    def setup(self):
        self.pascalvoc_train = list() 
        self.pascalvoc_val = list()

        path = self.data_dir 

        for category in PascalVOC.categories:
            dataset = PascalVOC(path, category, train=True, transform=self.transform, pre_filter=self.pre_filter)
            self.pascalvoc_train += [PairDataset(dataset, dataset, sample=True)]
            dataset = PascalVOC(self.data_dir, category, train=False, transform=self.transform, pre_filter=self.pre_filter)
            self.pascal_test__ += [PairDataset(dataset, dataset, sample=True)]
            # dataset = PascalVOC(path, category, train=False, transform=self.transform, pre_filter=self.pre_filter)
            # self.pascalvoc_test_ += [PairDataset(dataset, dataset, sample=True)]

        self.pascal_test_ = [DataLoader(dataset, self.batch_size_test, shuffle=False, follow_batch=['x_s', 'x_t']) for dataset in self.pascal_test__]

        self.pascalvoc_train = torch.utils.data.ConcatDataset(self.pascalvoc_train)
        self.pascalvoc_test = torch.utils.data.ConcatDataset(self.pascal_test__)
        self.pascalvoc_val = self.pascalvoc_test


    def visual_dataloader_train(self, shuffle, size):
        return DataLoader(self.pascalvoc_train, batch_size=size, shuffle=shuffle, follow_batch=['x_s', 'x_t'])
        
    def visual_dataloader_test(self, shuffle, size,):
        return DataLoader(self.pascalvoc_test, batch_size=size, shuffle=shuffle, follow_batch=['x_s', 'x_t'])

    def train_dataloader(self):
        return DataLoader(self.pascalvoc_train, batch_size=self.batch_size, shuffle=True, follow_batch=['x_s', 'x_t'])
    
    def val_dataloader(self):
        return DataLoader(self.pascalvoc_val, batch_size=self.batch_size_test, shuffle=False, follow_batch=['x_s', 'x_t'])
    
    def test_dataloader(self):
        return DataLoader(self.pascalvoc_test, batch_size=self.batch_size_test, shuffle=False, follow_batch=['x_s', 'x_t'])
    

