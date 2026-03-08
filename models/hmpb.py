import torch
import torch.nn as nn
import torch_geometric.nn as pygnn
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import dense_to_sparse

from .transformer_layers import FEATT


class LTIABlock(nn.Module):
    """Local + global edge-aware message passing block used by GraphBrep."""

    def __init__(
        self,
        dim_h,
        local_gnn_type,
        global_model_type,
        num_heads,
        temb_dim=None,
        act=None,
        dropout=0.0,
        attn_dropout=0.0,
    ):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.local_gnn_type = local_gnn_type
        self.global_model_type = global_model_type
        self.act = nn.ReLU() if act is None else act

        if temb_dim is not None:
            self.t_node = nn.Linear(temb_dim, dim_h)
            self.t_edge = nn.Linear(temb_dim, dim_h)

        if local_gnn_type == "None":
            self.local_model = None
        elif local_gnn_type == "GINE":
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h), nn.ReLU(), Linear_pyg(dim_h, dim_h))
            self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == "GAT":
            self.local_model = pygnn.GATConv(
                in_channels=dim_h,
                out_channels=dim_h // num_heads,
                heads=num_heads,
                edge_dim=dim_h,
            )
        elif local_gnn_type == "LocalTrans_1":
            self.local_model = FEATT(dim_h, dim_h // num_heads, num_heads, edge_dim=dim_h)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")

        if global_model_type == "None":
            self.self_attn = None
        elif global_model_type == "FullTrans_1":
            self.self_attn = FEATT(dim_h, dim_h // num_heads, num_heads, edge_dim=dim_h)
        else:
            raise ValueError(f"Unsupported global x-former model: {global_model_type}")

        self.norm1_local = nn.GroupNorm(num_groups=min(dim_h // 4, 32), num_channels=dim_h, eps=1e-6)
        self.norm1_attn = nn.GroupNorm(num_groups=min(dim_h // 4, 32), num_channels=dim_h, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.norm2_node = nn.GroupNorm(num_groups=min(dim_h // 4, 32), num_channels=dim_h, eps=1e-6)

        self.ff_linear3 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear4 = nn.Linear(dim_h * 2, dim_h)
        self.norm2_edge = nn.GroupNorm(num_groups=min(dim_h // 4, 32), num_channels=dim_h, eps=1e-6)

    def _ff_block_node(self, x):
        x = self.dropout(self.act(self.ff_linear1(x)))
        return self.dropout(self.ff_linear2(x))

    def _ff_block_edge(self, x):
        x = self.dropout(self.act(self.ff_linear3(x)))
        return self.dropout(self.ff_linear4(x))

    def forward(self, x, edge_index, dense_edge, dense_index, node_mask, adj_mask, temb=None):
        b, n, _, _ = dense_edge.shape
        h_in1 = x
        h_in2 = dense_edge

        h_edge = dense_edge
        h = x

        h_out_list = []
        if self.local_model is not None:
            edge_attr = h_edge[dense_index]
            h_local = self.local_model(h, edge_index, edge_attr) * node_mask.reshape(-1, 1)
            h_local = h_in1 + self.dropout(h_local)
            h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        if self.self_attn is not None:
            if "FullTrans" in self.global_model_type:
                dense_index_full = adj_mask.squeeze(-1).nonzero(as_tuple=True)
                edge_index_full, _ = dense_to_sparse(adj_mask.squeeze(-1))
                edge_attr_full = h_edge[dense_index_full]
                h_attn = self.self_attn(h, edge_index_full, edge_attr_full)
            else:
                raise ValueError("Unsupported global transformer layer")
            h_attn = h_in1 + self.dropout(h_attn)
            h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        assert len(h_out_list) > 0
        h = sum(h_out_list) * node_mask.reshape(-1, 1)
        h_dense = h.reshape(b, n, -1)
        h_edge = h_dense.unsqueeze(1) + h_dense.unsqueeze(2)

        h = h + self._ff_block_node(h)
        h = self.norm2_node(h) * node_mask.reshape(-1, 1)

        h_edge = h_in2 + self._ff_block_edge(h_edge)
        h_edge = (h_edge + h_edge.transpose(1, 2)) / 2
        h_edge = self.norm2_edge(h_edge.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * adj_mask

        return h, h_edge
