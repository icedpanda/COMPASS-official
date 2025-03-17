import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, TransformerConv, SAGEConv


class RGCN(nn.Module):
    def __init__(self,
                 num_relations: int,
                 in_dim: int = 256,
                 hid_dim: int = 256,
                 out_dim: int = 256,
                 num_bases: int = 8,
                 dropout: float = 0.1,
                 ):
        super().__init__()

        self.dropout = dropout
        self.conv1 = RGCNConv(in_dim, out_dim, num_relations, num_bases).to(torch.bfloat16)
        # self.conv2 = RGCNConv(hid_dim, out_dim, num_relations, num_bases)
        self.out_dim = out_dim

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training, p=self.dropout)
        # x = self.conv2(x, edge_index, edge_type)

        return x


class GraphSage(nn.Module):
    def __init__(self, in_dim: int = 256, hid_dim: int = 256, out_dim: int = 256, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)

        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(dim * 3, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, res):
        gate_input = torch.cat((x, res, x - res), dim=-1)
        gate = self.gate_proj(gate_input)
        return x * gate + res * (1 - gate)


class Feedforward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim if out_dim is None else out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GraphTransformer(nn.Module):
    def __init__(self,
                 n_layers=4,
                 in_channels=256,
                 n_heads=6,
                 concat=False,
                 edge_dim=256,
                 norm_edges=False,
                 ):
        super().__init__()

        node_hidden_dim = in_channels * 4
        edge_hidden_dim = edge_dim * 4
        self.out_dim = in_channels
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    PreNorm(in_channels, TransformerConv(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        concat=concat,
                        edge_dim=edge_dim,
                        heads=n_heads,
                        beta=True,
                    )),
                    nn.LayerNorm(edge_dim) if norm_edges else nn.Identity()
                ]),
                nn.ModuleList([
                    PreNorm(in_channels, Feedforward(in_channels, node_hidden_dim)),
                    GatedResidual(in_channels),
                    PreNorm(edge_dim, Feedforward(edge_dim, edge_hidden_dim)),
                    GatedResidual(edge_dim)
                ])
            ])
            for _ in range(n_layers)
        ])

    def forward(self, nodes, edge_index, edge_attr):
        for attn_blocks, ffn_block in self.layers:
            node_atten, edge_norm = attn_blocks

            edge_attr = edge_norm(edge_attr)  # pre-norm
            nodes = node_atten(nodes, edge_index=edge_index, edge_attr=edge_attr)

            node_pre_norm_ffn, node_gate_residual, edge_pre_norm_ffn, edge_gate_residual = ffn_block

            nodes = node_gate_residual(node_pre_norm_ffn(nodes), nodes)
            edge_attr = edge_gate_residual(edge_pre_norm_ffn(edge_attr), edge_attr)

        return nodes
