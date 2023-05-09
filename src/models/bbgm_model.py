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


    def forward(self, noisy_data, s_mask, t_mask, data, update_info, pad=False):
        images = data['images']
        points = data['Ps']
        n_points = data['ns']
        graphs = data['edges']
        
        bs = update_info['bs']
        
        global_list = []
        orig_graph_list = []
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
        
        
        '''
        hat_{S} = S_emb + S_proposed * f(t)
        '''
        
        # bs, n, d = src_x_.shape
        # bs, m, d = trg_x_.shape
        # exp_src_x = src_x_.unsqueeze(1).expand(bs, m, n, d)
        # exp_trg_x = trg_x_.unsqueeze(2).expand(bs, m, n, d)
        
        # diff = exp_src_x - exp_trg_x
        S_emb = src_x_ @ trg_x_.transpose(-1, -2) # this is 
        
        S_proposed = noisy_data['Xt']
        time_emb = self.positional_encoding(noisy_data['t'].to(S_emb.device)) # bs x 1
        time_scalar = self.time_linear2(self.act(self.time_linear1(time_emb)))
        
        S_hat = S_emb + S_proposed * torch.nn.functional.softplus(time_scalar).unsqueeze(-1)
        
        S_hat = masked_softmax(S_hat, update_info['mask_align'])
        
        if pad:
            return S_hat
        return S_hat[s_mask]
        
        # initialize source colour
        source_colour = None 
        # colouring target nodes using source colour space and the transfer matrix 
        
        # import pdb; pdb.set_trace()
        
        
        similarity_matrix = masked_softmax(similarity_matrix, update_info['mask_align'])[s_mask]
        
        return similarity_matrix
        
        # import pdb; pdb.set_trace()
        
        # import pdb; pdb.set_trace()
        
        
        #import pdb; pdb.set_trace()
        # concatenate global source and global target feature
        # global_weights_list = [
        #     torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
        # ]
        # global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        # unary_costs_list = [
        #     self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
        #     for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        # ]
        
        # unary_costs_list = list()
        # for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list):
        #     first_xs = list()
        #     second_xs = list()
        #     for item in g_1:
        #         print(item)
        #         import pdb; pdb.set_trace()
        #         first_xs.append(item)
        #     for item in g_2:
        #         second_xs.append(item)
                
        
        
        unary_costs_list = [torch.softmax(ele, dim=1) for ele in unary_costs_list[0]]  

        # Similarities to costs
        #unary_costs_list = [[-x for x in unary_costs] for unary_costs in unary_costs_list]
        
        return unary_costs_list


        if self.training:
            unary_costs_list = [
                [
                    x + 1.0*gt[:dim_src, :dim_tgt]  # Add margin with alpha = 1.0
                    for x, gt, dim_src, dim_tgt in zip(unary_costs, perm_mat, ns_src, ns_tgt)
                ]
                for unary_costs, perm_mat, (ns_src, ns_tgt) in zip(unary_costs_list, perm_mats, lexico_iter(n_points))
            ]

        quadratic_costs_list = [
            self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        # Aimilarities to costs
        quadratic_costs_list = [[-0.5 * x for x in quadratic_costs] for quadratic_costs in quadratic_costs_list]

        # if cfg.BB_GM.solver_name == "lpmp":
        #     all_edges = [[item.edge_index for item in graph] for graph in orig_graph_list]
        #     gm_solvers = [
        #         GraphMatchingModule(
        #             all_left_edges,
        #             all_right_edges,
        #             ns_src,
        #             ns_tgt,
        #             cfg.BB_GM.lambda_val,
        #             cfg.BB_GM.solver_params,
        #         )
        #         for (all_left_edges, all_right_edges), (ns_src, ns_tgt) in zip(
        #             lexico_iter(all_edges), lexico_iter(n_points)
        #         )
        #     ]
        #     matchings = [
        #         gm_solver(unary_costs, quadratic_costs)
        #         for gm_solver, unary_costs, quadratic_costs in zip(gm_solvers, unary_costs_list, quadratic_costs_list)
        #     ]
        # elif cfg.BB_GM.solver_name == "multigraph":
        #     all_edges = [[item.edge_index for item in graph] for graph in orig_graph_list]
        #     gm_solver = MultiGraphMatchingModule(
        #         all_edges, n_points, cfg.BB_GM.lambda_val, cfg.BB_GM.solver_params)
        #     matchings = gm_solver(unary_costs_list, quadratic_costs_list)
        # else:
        #     raise ValueError(f"Unknown solver {cfg.BB_GM.solver_name}")

        # if visualize_flag:
        #     easy_visualize(
        #         orig_graph_list,
        #         points,
        #         n_points,
        #         images,
        #         unary_costs_list,
        #         quadratic_costs_list,
        #         matchings,
        #         **visualization_params,
        #     )

        return None
