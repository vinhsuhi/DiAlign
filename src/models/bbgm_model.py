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
    def __init__(self, complex=False, without_diff=False):
        super(Net, self).__init__()
        out_dim = 512
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=1024, out_dim=out_dim)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.global_state_dim = 1024
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim, out_dim)
        self.edge_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim,
            self.build_edge_features_from_node_features.num_edge_features)
        self.positional_encoding = SinusoidalPosEmb(dim=20)
        
        self.message_dim_reduction = torch.nn.Linear(1024 * 2 + 20, 1024)
        self.complex = complex
        self.without_diff = without_diff

    def forward(self, noisy_data, data, update_info, pad=False, train=False):
        '''
        check if noisy data is valid
        '''
        s_mask = update_info['s_mask']
        t_mask = update_info['t_mask']
        device = s_mask.device
        images = data['images']
        points = data['Ps']
        n_points = data['ns']
        graphs = data['edges']
        Xt = noisy_data['Xt']
        time_emb = self.positional_encoding(noisy_data['t'].to(device)) 
        num_aligned_src = self.positional_encoding(Xt.sum(dim=-1)[s_mask] / 10)
        num_aligned_trg = self.positional_encoding(Xt.transpose(1, 2).sum(dim=-1)[t_mask] / 10)
        
        ##### change network here!
        global_list = []
        orig_graph_list = []
        all_node_features = []
        
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            # extract feature
            nodes = self.node_layers(image)
            edges = self.edge_layers(nodes)

            global_list.append(self.final_layers(edges)[0].reshape((nodes.shape[0], -1)))
            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            U = concat_features(feature_align(nodes, p, n_p, (256, 256)), n_p)
            F = concat_features(feature_align(edges, p, n_p, (256, 256)), n_p)
            node_features = torch.cat((U, F), dim=-1)
            
            graph.x = node_features
            # if train:
            #     import pdb; pdb.set_trace()
            orig_graph_list.append(graph)

        src_feat = orig_graph_list[0].x 
        trg_feat = orig_graph_list[1].x 
        

        if self.without_diff:
            reduce_feat_src = src_feat
            reduce_feat_trg = trg_feat
        else:
            x_padded_src, _ = to_dense_batch(src_feat, orig_graph_list[0].batch, fill_value=0)
            x_padded_trg, _ = to_dense_batch(trg_feat, orig_graph_list[1].batch, fill_value=0)
            
            target_messages = noisy_data['Xt'].float() @ x_padded_trg # sum trg1 + trg2 + ... + message from target nodes to source nodes
            source_messages = noisy_data['Xt'].float().transpose(1, 2) @ x_padded_src
            
            time_emb_src = time_emb.unsqueeze(1).repeat(1, x_padded_src.shape[1], 1)
            time_emb_trg = time_emb.unsqueeze(1).repeat(1, x_padded_trg.shape[1], 1)
            
            cat_feat_src = torch.cat((x_padded_src, target_messages, time_emb_src), dim=-1)
            cat_feat_trg = torch.cat((x_padded_trg, source_messages, time_emb_trg), dim=-1)
            reduce_feat_src = self.message_dim_reduction(cat_feat_src)[s_mask]
            reduce_feat_trg = self.message_dim_reduction(cat_feat_trg)[t_mask]
        
        src_graph, trg_graph = orig_graph_list
        src_graph.x = reduce_feat_src
        trg_graph.x = reduce_feat_trg
        
        src_graph = self.message_pass_node_features(src_graph)
        trg_graph = self.message_pass_node_features(trg_graph)
        
        processed_feat_src, _ = to_dense_batch(src_graph.x, src_graph.batch, fill_value=0)
        processed_feat_trg, _ = to_dense_batch(trg_graph.x, trg_graph.batch, fill_value=0)

        global_weights_list = [
            torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
        ]
        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        # import pdb; pdb.set_trace()

        unary_costs_list = [
            self.vertex_affinity(processed_feat_src, processed_feat_trg, global_weights, self.complex)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]
        prob = unary_costs_list[0] # too confidence

        if pad:
            return masked_softmax(prob, update_info['mask_align'])
        return masked_softmax(prob, update_info['mask_align'])[s_mask]
    
