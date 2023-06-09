import torch.nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import SplineConv


class SConv(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(SConv, self).__init__()

        self.in_channels = input_features
        self.num_layers = 2
        self.convs = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            conv = SplineConv(input_features, output_features, dim=2, kernel_size=5, aggr="max")
            self.convs.append(conv)
            input_features = output_features

        input_features = output_features
        self.out_channels = input_features
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        xs = [x]
        for conv in self.convs[:-1]:
            xs += [F.relu(conv(xs[-1], edge_index, edge_attr))]

        xs += [self.convs[-1](xs[-1], edge_index, edge_attr)]
        return xs[-1]


class SiameseSConvOnNodes(torch.nn.Module):
    def __init__(self, input_node_dim, out_dim=1024):
        super(SiameseSConvOnNodes, self).__init__()
        self.num_node_features = input_node_dim
        self.out_dim = out_dim
        self.lin_tran = torch.nn.Linear(self.num_node_features, 700)
        self.act = torch.nn.Tanh()
        
        self.lin_tran2 = torch.nn.Linear(700, out_dim)
        self.mp_network = SConv(
            input_features=self.num_node_features, output_features=self.out_dim)

    def forward(self, graph):
        old_features = self.lin_tran2(self.act(self.lin_tran(graph.x)))
        result = self.mp_network(graph)
        # if self.out_dim < self.num_node_features:
        graph.x = old_features + 0.1 * result
        # else:
        #graph.x = result 
        return graph


class SiameseNodeFeaturesToEdgeFeatures(torch.nn.Module):
    def __init__(self, total_num_nodes):
        super(SiameseNodeFeaturesToEdgeFeatures, self).__init__()
        self.num_edge_features = total_num_nodes

    def forward(self, graph):
        orig_graphs = graph.to_data_list()
        orig_graphs = [self.vertex_attr_to_edge_attr(graph) for graph in orig_graphs]
        return orig_graphs

    def vertex_attr_to_edge_attr(self, graph):
        """Assigns the difference of node features to each edge"""
        flat_edges = graph.edge_index.transpose(0, 1).reshape(-1)
        vertex_attrs = torch.index_select(graph.x, dim=0, index=flat_edges)

        new_shape = (graph.edge_index.shape[1], 2, vertex_attrs.shape[1])
        vertex_attrs_reshaped = vertex_attrs.reshape(new_shape).transpose(0, 1)
        new_edge_attrs = vertex_attrs_reshaped[0] - vertex_attrs_reshaped[1]
        graph.edge_attr = new_edge_attrs
        return graph
