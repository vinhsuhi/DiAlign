import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import Metric, MetricCollection
import time
import wandb

class AlignAcc(Metric):
    higher_is_better: bool = True
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        assert reduction in ['mean', 'sum']

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update_old(self, S: torch.Tensor, y: torch.Tensor):
        r"""Computes the accuracy of correspondence predictions.
        Args:
            S (Tensor): Sparse or dense correspondence matrix of shape
                :obj:`[batch_size * num_nodes, num_nodes]`.
            y (LongTensor): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]`.
            reduction (string, optional): Specifies the reduction to apply to
                the output: :obj:`'mean'|'sum'`. (default: :obj:`'mean'`)
        """
        pred = S[y[0]].argmax(dim=1)
        
        self.correct += torch.sum(pred == y[1])
        self.total += y[1].numel()
        
    def update(self, S: torch.Tensor, y: torch.Tensor):
        r"""Computes the accuracy of correspondence predictions.
        Args:
            S (Tensor): Sparse or dense correspondence matrix of shape
                :obj:`[batch_size * num_nodes, num_nodes]`.
            y (LongTensor): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]`.
            reduction (string, optional): Specifies the reduction to apply to
                the output: :obj:`'mean'|'sum'`. (default: :obj:`'mean'`)
        """
        
        pred = S.argmax(dim=1)
        target = y.argmax(dim=1)
        self.correct += torch.sum(pred == target)
        self.total += len(target)
        
    
    def compute(self):
        if self.reduction == "mean":
            return self.correct.float() / self.total
        else:
            return self.correct.float()        
    
