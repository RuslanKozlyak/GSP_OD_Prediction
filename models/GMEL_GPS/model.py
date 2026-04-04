import torch
import torch.nn as nn

from models.GPS.model import GPSEncoder


class GMEL_GPS(nn.Module):
    """GMEL with GPS encoders instead of GAT.

    Replaces the two GAT encoders from GMEL with two independent
    PyG-based GPSEncoder instances (one for inflow, one for outflow),
    while keeping the rest of GMEL's architecture unchanged:
      - linear_in / linear_out  (hidden_dim → 1)
      - bilinear                (hidden_dim × hidden_dim → 1)
    Returns the same tuple as GMEL.forward: (flow_in, flow_out, flow, h_in, h_out)
    """

    def __init__(self, input_dim, edge_dim, hidden_dim=64, pe_dim=8,
                 n_layers=3, n_heads=4, dropout=0.1,
                 pe_type='rwpe', norm_type='batch_norm'):
        super().__init__()
        self.gps_in = GPSEncoder(
            input_dim, hidden_dim, pe_dim, edge_dim,
            n_layers, n_heads, dropout, pe_type, norm_type,
        )
        self.gps_out = GPSEncoder(
            input_dim, hidden_dim, pe_dim, edge_dim,
            n_layers, n_heads, dropout, pe_type, norm_type,
        )
        self.linear_in  = nn.Linear(hidden_dim, 1)
        self.linear_out = nn.Linear(hidden_dim, 1)
        self.bilinear   = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(self, graph_data):
        """
        Args:
            graph_data: torch_geometric.data.Data with x, edge_index, edge_attr, pe

        Returns:
            flow_in  (N, 1)  — inflow prediction per node
            flow_out (N, 1)  — outflow prediction per node
            flow     (N, 1)  — bilinear OD signal (same semantics as GMEL)
            h_in     (N, hidden_dim)
            h_out    (N, hidden_dim)
        """
        h_in  = self.gps_in(graph_data)          # (N, hidden_dim)
        h_out = self.gps_out(graph_data)          # (N, hidden_dim)
        flow_in  = self.linear_in(h_in)           # (N, 1)
        flow_out = self.linear_out(h_out)          # (N, 1)
        w = self.bilinear.weight.squeeze(0)
        flow = (h_out @ w) @ h_in.transpose(0, 1)
        if self.bilinear.bias is not None:
            flow = flow + self.bilinear.bias.view(1, 1)
        return flow_in, flow_out, flow, h_in, h_out
