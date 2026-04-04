import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, in_dim=131, num_hidden=64, out_dim=64, num_layers=3, num_heads=6):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation = F.elu

        self.input_layer = GATConv(
            in_dim,
            num_hidden,
            heads=self.num_heads,
            concat=True,
            add_self_loops=False,
        )
        self.hidden_layers = nn.ModuleList()

        for _ in range(1, self.num_layers):
            self.hidden_layers.append(
                GATConv(
                    num_hidden * self.num_heads,
                    num_hidden,
                    heads=self.num_heads,
                    concat=True,
                    add_self_loops=False,
                )
            )

        self.output_layer = GATConv(
            num_hidden * self.num_heads,
            out_dim,
            heads=self.num_heads,
            concat=False,
            add_self_loops=False,
        )

    def forward(self, graph_data, nfeat=None):
        edge_index = graph_data.edge_index if hasattr(graph_data, 'edge_index') else graph_data
        h = graph_data.x if nfeat is None and hasattr(graph_data, 'x') else nfeat
        if h is None:
            raise ValueError("GMEL.GAT.forward expects node features via graph_data.x or nfeat.")

        h = self.activation(self.input_layer(h, edge_index))
        for layer in self.hidden_layers:
            h = self.activation(layer(h, edge_index))

        embeddings = self.output_layer(h, edge_index)
        return embeddings


class GMEL(nn.Module):
    def __init__(self):
        super(GMEL, self).__init__()

        self.gat_in = GAT()
        self.gat_out = GAT()

        self.linear_in = nn.Linear(64, 1)
        self.linear_out = nn.Linear(64, 1)
        self.bilinear = nn.Bilinear(64, 64, 1)

    def forward(self, graph_data, nfeat=None):
        h_in = self.gat_in(graph_data, nfeat)
        flow_in = self.linear_in(h_in)

        h_out = self.gat_out(graph_data, nfeat)
        flow_out = self.linear_out(h_out)

        w = self.bilinear.weight.squeeze(0)
        flow = (h_out @ w) @ h_in.transpose(0, 1)
        if self.bilinear.bias is not None:
            flow = flow + self.bilinear.bias.view(1, 1)

        return flow_in, flow_out, flow, h_in, h_out
