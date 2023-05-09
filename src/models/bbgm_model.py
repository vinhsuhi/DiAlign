import torch

from src.bbgm_utils import backbone
from src.models.affinity_layer import InnerProductWithWeightsAffinity
from src.models.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
#from lpmp_py import GraphMatchingModule
#from lpmp_py import MultiGraphMatchingModule
# from src.bbgm_utils.config import cfg
from src.bbgm_utils.feature_align import feature_align
from src.bbgm_utils.utils import lexico_iter
# from utils.visualization import easy_visualize
from torch_geometric.utils import to_dense_batch
from src.diffusion.layers import SinusoidalPosEmb
import torch.nn.functional as F


import torch
from torch.nn import Linear as Lin
from torch_geometric.nn import GINConv

# from .mlp import MLP


class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, batch_norm=False,
                 cat=True, lin=True):
        super(GIN, self).__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.cat = cat
        self.lin = lin

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP(in_channels, out_channels, 2, batch_norm, dropout=0.0)
            self.convs.append(GINConv(mlp, train_eps=True))
            in_channels = out_channels

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.final = Lin(in_channels, out_channels)
        else:
            self.out_channels = in_channels

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index, *args):
        """"""
        xs = [x]

        for conv in self.convs:
            xs += [conv(xs[-1], edge_index)]

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = self.final(x) if self.lin else x
        return x

    def __repr__(self):
        return ('{}({}, {}, num_layers={}, batch_norm={}, cat={}, '
                'lin={})').format(self.__class__.__name__, self.in_channels,
                                  self.out_channels, self.num_layers,
                                  self.batch_norm, self.cat, self.lin)


def masked_softmax(src, mask, dim=-1):
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)
    return out


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)


class Net(backbone.VGG16_bn):
    def __init__(self):
        super(Net, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=1024, out_dim=1024)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.global_state_dim = 1024
        self.scalar_dim = 20
        # self.verification = GIN()
        self.positional_encoding = SinusoidalPosEmb(dim=self.scalar_dim)
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim, self.message_pass_node_features.out_dim)
        
        self.time_linear1 = torch.nn.Linear(20, 10)
        self.act = torch.nn.ReLU()
        self.time_linear2 = torch.nn.Linear(10, 1)
        # self.edge_affinity = InnerProductWithWeightsAffinity(
        #     self.global_state_dim,
        #     self.build_edge_features_from_node_features.num_edge_features)

        # def forward(
        #     self,
        #     images,
        #     points,
        #     graphs,
        #     n_points,
        #     perm_mats,
        #     visualize_flag=False,
        #     visualization_params=None,
        # ):


    def forward(self, noisy_data, s_mask, data, update_info, pad=False):
        images = data['images']
        points = data['Ps']
        n_points = data['ns']
        graphs = data['edges']
        
        global_list = []
        processed_graphs = []
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            # extract feature
            nodes = self.node_layers(image)
            edges = self.edge_layers(nodes)
            
            global_list.append(self.final_layers(edges)[0].reshape((nodes.shape[0], -1)))
            nodes = normalize_over_channels(nodes) #
            edges = normalize_over_channels(edges)

            # arrange features
            U = concat_features(feature_align(nodes, p, n_p, (256, 256)), n_p)
            F = concat_features(feature_align(edges, p, n_p, (256, 256)), n_p)
            node_features = torch.cat((U, F), dim=-1)
            graph.x = node_features

            graph = self.message_pass_node_features(graph)
            processed_graphs.append(graph)
        
        src_graph, trg_graph = processed_graphs
        src_x, trg_x = src_graph.x, trg_graph.x
        
        src_x_, _ = to_dense_batch(src_x, src_graph.batch, fill_value=0)
        trg_x_, _ = to_dense_batch(trg_x, trg_graph.batch, fill_value=0)
        
        S_emb = src_x_ @ trg_x_.transpose(-1, -2) # this is 
        S_proposed = noisy_data['Xt']
        time_emb = self.positional_encoding(noisy_data['t'].to(S_emb.device)) # bs x 1
        time_scalar = self.time_linear2(self.act(self.time_linear1(time_emb)))
        time_weight = torch.nn.functional.softplus(time_scalar)
        S_hat = S_emb + S_proposed * time_weight.unsqueeze(-1)
        S_hat = masked_softmax(S_hat, update_info['mask_align'])
        
        if pad:
            return S_hat
        return S_hat[s_mask]
        
