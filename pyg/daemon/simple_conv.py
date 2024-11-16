from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import add_self_loops

from .base import activation_layer


class SimpleConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_normalize: bool = False,
        add_self_loop: bool = True,
        activation: str = "relu",
        aggr: Literal["sum", "mean"] = "sum",
        bias: bool = True,
        **kwargs,
    ):
        """
        The SimpleConv operator from the `"DAEMON" <https://www.amazon.science/publications/recommending-related-products-using-graph-neural-networks-in-directed-graphs>` at row 4 or 5 in Algorithm 1.

        Args:
            in_channels: Size of each input sample.
            out_channels: Size of each output sample.
            output_normalize: If set to `True`, output features will be normalized to unit length.
            add_self_loop: If set to `True`, will include a root node weight.
            project: If set to `True`, will include a linear projection.
            activation: The activation function to use.
            bias: If set to `False`, the layer will not learn an additive bias.
            **kwargs: Additional arguments of `torch_geometric.nn.conv.MessagePassing`.

        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_normalize = output_normalize
        self.add_self_loop = add_self_loop

        super().__init__(aggr=aggr, **kwargs)

        self.lin = Linear(in_channels, out_channels, bias=bias)
        self.act = activation_layer(activation)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tensor:
        """Forward pass

        Args:
            x: node features, shape = (num_nodes, in_channels)
            edge_index: edge indices, shape = (2, num_edges)

        Returns:
            node features, shape = (num_nodes, out_channels)
        """
        if self.add_self_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        # message and aggregate, then update
        out = self.propagate(edge_index, x=x)
        out = self.act(out)
        if self.output_normalize:
            out = F.normalize(out, p=2.0, dim=-1)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        """j -> iに対するメッセージ関数

        Args:
            x_j: jの特徴量

        Returns:
            j -> iのメッセージ
        """
        return x_j

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, aggr={self.aggr})"
        )
