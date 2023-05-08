import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

from src import utils
from src.diffusion import diffusion_utils
from src.diffusion.layers import SinusoidalPosEmb

from torch.nn import Linear as Lin
from torch_geometric.nn import SplineConv
from torch_geometric.utils import to_dense_batch



def masked_softmax(src, mask, dim=-1):
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)
    return out


class GraphTransformerMatching(nn.Module):
    def __init__(self, scalar_dim=20, num_layers=3, ori_feat_dim=1024, embed_dim=128, cat=True, lin=True, dropout=0.0, use_time=True):
        super().__init__()
        self.positional_encoding = SinusoidalPosEmb(dim=scalar_dim)
        self.scalar_dim = scalar_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_time = use_time
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(ori_feat_dim + scalar_dim * 2, embed_dim))
        for _ in range(num_layers - 1):
            self.linears.append(nn.Linear(embed_dim + scalar_dim * 2, embed_dim))
        self.cat = cat 
        self.lin = lin 

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = SplineConv(embed_dim * 2, embed_dim, 2, kernel_size=5)
            self.convs.append(conv)

        final_in_dim = ori_feat_dim + embed_dim * num_layers
        if not self.cat:
            final_in_dim = embed_dim
        
        self.final = nn.Linear(final_in_dim, embed_dim)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.lin:
            self.final.reset_parameters()


    def forward(self, noisy_data, s_mask, t_mask, graph_s_data, graph_t_data):
        """
        dict_keys(['t_int', 't', 'beta_t', 'alpha_sb', 'alpha_tb', 'Xt', 'mask_align'])
        what is the input?
        what is the output? - new alignment matrix
        """
        device = s_mask.device
        if self.scalar_dim > 1:
            time_emb = self.positional_encoding(noisy_data['t'].to(device))

            num_aligned_src = self.positional_encoding(noisy_data['Xt'].sum(dim=-1)[s_mask] / 10)
            num_aligned_trg = self.positional_encoding(noisy_data['Xt'].transpose(1, 2).sum(dim=-1)[t_mask] / 10)
        else:
            time_emb = noisy_data['t'].to(device).reshape(-1, 1) # bs x 20
            num_aligned_src = noisy_data['Xt'].sum(dim=-1)[s_mask] / 10
            num_aligned_trg = noisy_data['Xt'].transpose(1, 2).sum(dim=-1)[t_mask] / 10
            num_aligned_src = num_aligned_src.reshape(-1, 1)
            num_aligned_trg = num_aligned_trg.reshape(-1, 1)

        if not self.use_time:
            time_emb = time_emb * 0

        bnn_src = s_mask.sum(dim=1)
        bnn_trg = t_mask.sum(dim=1)
        # nhung features nao minh muon 
        time_features_src = list()
        time_features_trg = list()
        for i in range(len(bnn_src)):
            time_features_src.append(time_emb[i].repeat(bnn_src[i], 1))
            time_features_trg.append(time_emb[i].repeat(bnn_trg[i], 1))

        time_features_src = torch.cat(time_features_src, dim=0)
        time_features_trg = torch.cat(time_features_trg, dim=0)

        xs_src = [graph_s_data['x']]
        xs_trg = [graph_t_data['x']]

        for i in range(self.num_layers):
            x_input_src = self.linears[i](torch.cat((xs_src[-1], num_aligned_src, time_features_src), dim=-1))
            x_input_trg = self.linears[i](torch.cat((xs_trg[-1], num_aligned_trg, time_features_trg), dim=-1))

            x_padded_src, _ = to_dense_batch(x_input_src, graph_s_data['batch'], fill_value=0)
            x_padded_trg, _ = to_dense_batch(x_input_trg, graph_t_data['batch'], fill_value=0)

            target_messages = noisy_data['Xt'].float() @ x_padded_trg
            source_messages = noisy_data['Xt'].float().transpose(1, 2) @ x_padded_src

            diff_src_ = x_padded_src - target_messages
            diff_trg_ = x_padded_trg - source_messages

            diff_src = diff_src_[s_mask]
            diff_trg = diff_trg_[t_mask]

            cat_src = torch.cat((x_input_src, diff_src), dim=-1) # gut
            cat_trg = torch.cat((x_input_trg, diff_trg), dim=-1) # gut


            xs_src += [torch.tanh(self.convs[i](cat_src, graph_s_data['edge_index'], graph_s_data['edge_attr']))]
            xs_trg += [torch.tanh(self.convs[i](cat_trg, graph_t_data['edge_index'], graph_t_data['edge_attr']))]


        x_src = F.dropout(torch.cat(xs_src, dim=-1) if self.cat else xs_src[-1], p=self.dropout, training=self.training)
        x_trg = F.dropout(torch.cat(xs_trg, dim=-1) if self.cat else xs_trg[-1], p=self.dropout, training=self.training)

        x_src = self.final(x_src) if self.lin else x_src 
        x_trg = self.final(x_trg) if self.lin else x_trg 

        x_src_, _ = to_dense_batch(x_src, graph_s_data['batch'], fill_value=0)
        x_trg_, _ = to_dense_batch(x_trg, graph_t_data['batch'], fill_value=0)

        # TODO: Is this a naive way of coputing similarity matrix?
        similarity_matrix = x_src_ @ x_trg_.transpose(-1, -2)
        similarity_matrix = masked_softmax(similarity_matrix, noisy_data['mask_align'])[s_mask]

        return similarity_matrix
    