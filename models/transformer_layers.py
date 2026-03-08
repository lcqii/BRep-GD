import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax


class FEATT(MessagePassing):
    """Edge-gated attention message passing used by LTIABlock."""

    _alpha: OptTensor

    def __init__(
        self,
        x_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.lin_key = Linear(self.in_channels, heads * out_channels, bias=bias)
        self.lin_query = Linear(self.in_channels, heads * out_channels, bias=bias)
        self.lin_value = Linear(self.in_channels, heads * out_channels, bias=bias)

        self.lin_edge0 = Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_edge1 = Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_edge2 = Linear(edge_dim, heads * out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge0.reset_parameters()
        self.lin_edge1.reset_parameters()
        self.lin_edge2.reset_parameters()

    def forward(self, x: OptTensor, edge_index: Adj, edge_attr: OptTensor = None) -> Tensor:
        h, c = self.heads, self.out_channels
        query = self.lin_query(x).view(-1, h, c)
        key = self.lin_key(x).view(-1, h, c)
        value = self.lin_value(x).view(-1, h, c)
        return self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr)

    def message(self, query_i, key_j, value_j, edge_attr, index, ptr, size_i):
        edge_attn = self.lin_edge0(edge_attr).view(-1, self.heads, self.out_channels)
        edge_attn = torch.tanh(edge_attn)
        alpha = (query_i * key_j * edge_attn).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = value_j
        msg = msg * torch.tanh(self.lin_edge1(edge_attr).view(-1, self.heads, self.out_channels))
        msg = msg * alpha.view(-1, self.heads, 1)
        msg = msg.view(-1, self.heads * self.out_channels)
        msg = msg + self.lin_edge2(edge_attr)
        return msg

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})"
