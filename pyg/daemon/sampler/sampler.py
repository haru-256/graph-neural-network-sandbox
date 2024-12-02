import math
from collections import deque
from typing import Callable, Optional, overload

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.sampler import (
    BaseSampler,
    EdgeSamplerInput,
    NegativeSampling,
    NumNeighbors,
    SamplerOutput,
)
from torch_geometric.utils import mask_to_index

from .utils import remove_edges


class DAEMONNeighborSampler(BaseSampler):
    def __init__(
        self,
        data: Data,
        num_neighbors: list[int],
        weight_attr: Optional[str] = None,
        replace: bool = False,
        seed: int = 1026,
    ):
        if data.num_nodes is None:
            raise ValueError("data.num_nodes is required for sampling")
        if weight_attr is not None:
            raise NotImplementedError("Weighted sampling is not implemented yet")
        if replace:
            raise NotImplementedError("Replacement is not implemented yet")

        self.data = data
        self.num_neighbors = NumNeighbors(values=num_neighbors)
        self.weight_attr = weight_attr
        self.num_nodes = data.num_nodes
        self.rng = torch.Generator().manual_seed(seed)

    def sample_from_edges(
        self, inputs: EdgeSamplerInput, neg_sampling: Optional[NegativeSampling] = None
    ) -> SamplerOutput:
        """
        sampling from edges
        Args:
            index: EdgeSamplerInput.
            neg_sampling: neg_sampling method.
        Returns:
            SamplerOutput.
        """
        if neg_sampling is None:
            raise ValueError("neg_sampling is required for edge sampling")
        out = edge_sample(
            inputs=inputs,
            sample_fn=self._sample,
            num_nodes=self.num_nodes,
            neg_sampling=neg_sampling,
        )
        return out

    def _sample(self, src: Tensor, dst_pos: Tensor, dst_neg: Tensor) -> SamplerOutput:
        """
        Implements neighbor sampling by calling either :obj:`pyg-lib` (if
        installed) or :obj:`torch-sparse` (if installed) sampling routines.
        Args:
            src: source nodes. shape: (num_src, )
            dst_pos: positive destination nodes. shape: (num_src, )
            dst_neg: negative destination nodes. shape: (num_src, num_neg)
        Raises:
            ValueError: _description_
            ImportError: _description_
        Returns:
            out: SamplerOutput. サンプリングされたノード、エッジ、バッチの情報を含む。out.node の最初のnum_pos個がsrcに、次のnum_pos個がpos_dstに、残りがneg_dstに対応していることを仮定している
        """
        assert src.size() == dst_pos.size()
        assert dst_neg.ndim == 2 and dst_neg.size(0) == src.size(0)

        seed = torch.cat([src, dst_pos, dst_neg.flatten()], dim=0).unique()
        assert seed.ndim == 1, f"Seed tensor must be one-dimensional: {seed.size()=}"

        edge_index = self.data.edge_index
        edge_index_type = self.data.edge_index_type
        num_neighbors = self.num_neighbors.get_values()
        assert (
            edge_index is not None and edge_index_type is not None
        ), f"{edge_index=}, {edge_index_type=}"

        sampled_nodes = seed  # samplingされたノードを格納するtensor
        sampled_edge_indices = torch.tensor([], dtype=torch.long).reshape(
            2, 0
        )  # samplingされたエッジを格納するtensor
        sampled_edge_index_ptrs = torch.tensor(
            [], dtype=torch.long
        )  # samplingされたエッジのindexを格納するtensor

        # 深さ優先探索で近傍nodeをサンプリングする
        seed_queue = deque(seed)  # samplingするseedを格納するqueue
        layer_queue = deque([0] * seed.numel())  # seedの層を格納するqueue
        remained_edge_index = edge_index
        remained_edge_index_ptr = torch.arange(edge_index.size(1))
        while seed_queue:
            seed = seed_queue.popleft()
            layer = layer_queue.popleft()
            # layerがnum_neighborsの長さを超えた場合(= 指定されたhop数を超えた場合)はスキップ
            if layer >= len(num_neighbors):
                continue
            # seedを起点に近傍nodeをサンプリングする
            (
                sampled_node,  # shape: (N,)
                sampled_edge_index,  # shape: (2, N)
                sampled_edge_index_ptr,  # shape: (N, 1)
                remained_edge_index,  # shape: (2, num_edges - N)
                remained_edge_index_ptr,  # shape: (num_edges - N, 1)
            ) = sample_one_hop_neighbors(
                seed=int(seed.item()),
                edge_index=remained_edge_index,
                num_neighbor=num_neighbors[layer],
                edge_attr=remained_edge_index_ptr.reshape(-1, 1),
                generator=self.rng,
            )
            sampled_edge_index_ptr = sampled_edge_index_ptr.squeeze(1)
            remained_edge_index_ptr = remained_edge_index_ptr.squeeze(1)
            # すでにsamplingされたことがあるノードは再度サンプリングしないため、除外する
            filtered_sampled_node = torch.tensor(
                list(filter(lambda x: x not in sampled_nodes, sampled_node))
            )
            # queueに追加
            seed_queue.extend(filtered_sampled_node)
            layer_queue.extend([layer + 1] * len(filtered_sampled_node))
            # samplingされたノード、エッジを登録
            sampled_nodes = torch.hstack([sampled_nodes, filtered_sampled_node])
            sampled_edge_indices = torch.hstack(
                [sampled_edge_indices, sampled_edge_index]
            )
            sampled_edge_index_ptrs = torch.hstack(
                [sampled_edge_index_ptrs, sampled_edge_index_ptr]
            )

        # sampled_edge_indexを0 ~ num_nodes - 1にindexを振り直す
        node_wo_isolated, sampled_edge_index = torch.unique(
            sampled_edge_indices, return_inverse=True
        )
        # sampled_nodesにおける孤立ノードをnodeに追加する
        isolated_nodes = torch.tensor(
            list(filter(lambda x: x not in node_wo_isolated, sampled_nodes))
        )
        node = torch.hstack([node_wo_isolated, isolated_nodes]).long()
        row = sampled_edge_index[0]
        col = sampled_edge_index[1]
        edge = sampled_edge_index_ptrs.long()
        # src, dst_pos, dst_negのindexを取得する
        src_index, dst_pos_index, dst_neg_index = find_index(
            node=node, src=src, dst_pos=dst_pos, dst_neg=dst_neg
        )
        input_id = None
        seed_time = None

        return SamplerOutput(
            node=node,  # n_id相当, originalのnode_indexのindex
            row=row,  # src相当
            col=col,  # dst相当
            edge=edge,  # e_id相当, originalのedge_indexのindex
            batch=None,
            num_sampled_nodes=None,
            num_sampled_edges=None,
            metadata=(
                input_id,
                src_index,
                dst_pos_index,
                dst_neg_index,
                seed_time,
            ),  # NOTE: LinkLoaderのfilter_fnで使用するための情報。この順番である必要がある
        )


def find_index(
    node: Tensor, src: Tensor, dst_pos: Tensor, dst_neg: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Find the index of the node w,r.t src, dst_pos and dst_neg.
    Args:
        node: all nodes. shape: (num_nodes, )
        src: source nodes. shape: (num_src, )
        dst_pos: positive destination nodes. shape: (num_src, )
        dst_neg: negative destination nodes. shape: (num_src, num_neg)
    Returns:
        index: index of src, dst_pos, dst_neg. each shape: (num_src, ), (num_pos, ), (num_src, num_neg)
    """
    assert src.size() == dst_pos.size()
    assert dst_neg.ndim == 2 and dst_neg.size(0) == src.size(0)

    src_index = torch.tensor([], dtype=torch.long)
    dst_pos_index = torch.tensor([], dtype=torch.long)
    dst_neg_index_: list[Tensor] = []
    for src_, dst_pos_, dst_neg_ in zip(src, dst_pos, dst_neg):
        # src
        src_mask = torch.tensor([n in src_ for n in node], dtype=torch.bool)
        assert sum(src_mask) == 1, f"{src_mask=}"
        src_index = torch.hstack([src_index, mask_to_index(src_mask)])
        # dst_pos
        dst_pos_mask = torch.tensor([n in dst_pos_ for n in node], dtype=torch.bool)
        assert sum(dst_pos_mask) == 1, f"{dst_pos_mask=}"
        dst_pos_index = torch.hstack([dst_pos_index, mask_to_index(dst_pos_mask)])
        # dst_neg
        dst_neg_index_per_src = torch.tensor([], dtype=torch.long)
        for neg_ in dst_neg_:
            dst_neg_mask = torch.tensor([n in neg_ for n in node], dtype=torch.bool)
            assert sum(dst_neg_mask) == 1, f"{dst_neg_mask=}"
            dst_neg_index_per_src = torch.hstack(
                [dst_neg_index_per_src, mask_to_index(dst_neg_mask)]
            )
        dst_neg_index_.append(dst_neg_index_per_src)
    dst_neg_index = torch.vstack(dst_neg_index_)

    return src_index, dst_pos_index, dst_neg_index


@overload
def sample_one_hop_neighbors(
    seed: int,
    edge_index: Tensor,
    num_neighbor: int,
    edge_attr: None,
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, Tensor, None, Tensor, None]: ...


@overload
def sample_one_hop_neighbors(
    seed: int,
    edge_index: Tensor,
    num_neighbor: int,
    edge_attr: Tensor,
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: ...


def sample_one_hop_neighbors(
    seed: int,
    edge_index: Tensor,
    num_neighbor: int,
    edge_attr: Optional[Tensor],
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
    """
    sampling one-hop neighbors

    Args:
        seed: seed node. shape: (1, )
        edge_index: edge index. shape: (2, num_edges)
        edge_attr: edge attributes. shape: (num_edges, num_edge_attrs)
        remained_edge_index: remained edge index. shape: (2, num_remained_edges)
        num_neighbor: number of neighbors to sample
        sampled_nodes: すでにサンプリングされたノード, shape: (num_sampled_nodes, )
        generator: random number generator

    Returns:
        sampled: sampled nodes: shape=(N,), sampled edges: shape=(2, N), sampled edge attr: shape=(N, D), remained edges: shape=(2. num_edges - N), remained edge attr: shape=(num_edges - N, D)
    """
    # seedを起点にedge_indexからnum_neighbor個の近傍nodeをサンプリングする
    # 1. seedがsrc or dstとなるedgeをサンプリングする
    mask = (edge_index == seed).any(dim=0)  # shape: (num_edges, )
    target_edge_index = edge_index[:, mask]
    target_edge_attr = edge_attr[mask] if edge_attr is not None else None
    # 2. seedがsrcとなるedgeに限定せずedgeをサンプリングする
    # TODO: 重み付きsamplingの実装
    indices = torch.randperm(target_edge_index.size(1), generator=generator)[
        :num_neighbor
    ]
    sampled_edge_index = target_edge_index[:, indices]
    sampled_edge_attr = (
        target_edge_attr[indices] if target_edge_attr is not None else None
    )
    # 3. edge_indexからサンプリングされたedgeを削除する
    if sampled_edge_attr is None:
        remained_edge_index = remove_edges(
            edge_index=edge_index,
            removed_edge_index=sampled_edge_index,
        )
        remained_edge_attr = None
    else:
        remained_edge_index, remained_edge_attr = remove_edges(
            edge_index=edge_index,
            edge_attr=edge_attr,
            removed_edge_index=sampled_edge_index,
        )
    # 4. edgeからseed以外のnodeを取得する
    sampled_nodes = sampled_edge_index[sampled_edge_index != seed].flatten()
    assert sampled_nodes.numel() == sampled_nodes.unique().numel()

    return (
        sampled_nodes,
        sampled_edge_index,
        sampled_edge_attr,
        remained_edge_index,
        remained_edge_attr,
    )


def edge_sample(
    inputs: EdgeSamplerInput,
    sample_fn: Callable[[Tensor, Tensor, Tensor], SamplerOutput],
    num_nodes: int,
    neg_sampling: NegativeSampling,
) -> SamplerOutput:
    """
    Performs sampling from an edge sampler input, leveraging a sampling
    function of the same signature as `node_sample`.
    Args:
        inputs: EdgeSamplerInput.
        sample_fn: Sampling Neighbor function., returns a SamplerOutput. src, dst_pos, dst_negを引数にそのseedの近傍をサンプリングする関数
        num_nodes: Number of nodes.
        neg_sampling: Negative sampling method
    Raises:
        ValueError: If `neg_sampling` is binary.
    Returns:
        _description_
    """
    src = inputs.row  # shape: (num_src,)
    dst_pos = inputs.col  # shape: (num_pos,)
    edge_label = inputs.label
    assert src.size() == dst_pos.size()
    if edge_label is not None:
        raise ValueError(
            "Support only triplet negative sampling. So edge_label should be None."
        )
    # Negative Sampling #######################################################
    # When we are doing negative sampling, we append negative information
    # of nodes/edges to `src`, `dst`, `src_time`, `dst_time`.
    # Later on, we can easily reconstruct what belongs to positive and
    # negative examples by slicing via `num_pos`.
    num_pos = src.numel()
    if neg_sampling.is_binary():
        raise ValueError("Binary negative sampling is not supported for edge sampling")
    # In the "triplet" case, we randomly sample negative destinations.
    # TODO: Support non-false negative negative sampling for sources.
    dst_neg = approx_neg_sample(
        num_pos=num_pos, neg_sampling=neg_sampling, num_nodes=num_nodes
    )
    # sampling the neighbors from the seed
    out = sample_fn(src, dst_pos, dst_neg.reshape(src.size(0), -1))
    return out


def neg_sample(
    seed: Tensor,
    neg_sampling: NegativeSampling,
    num_nodes: int,
) -> Tensor:
    """
    Negative samplingする関数。neg_samplingのamountに基づいて、seedの近傍からnum_neg個のノードをサンプリングする。NegativeSamplingの性質上、偽陰性をサンプリングする可能性がある
    Args:
        seed: seed tensor.
        neg_sampling: NegativeSampling.
        num_nodes: Number of nodes.
    Returns:
        negative sampled nodes.
    """
    num_neg = math.ceil(seed.numel() * neg_sampling.amount)
    # TODO: Do not sample false negatives.
    return neg_sampling.sample(num_neg, num_nodes)


def approx_neg_sample(
    num_pos: int,
    neg_sampling: NegativeSampling,
    num_nodes: int,
) -> Tensor:
    """
    Negative samplingする関数。neg_samplingのamountに基づいて、seedの近傍からnum_neg個のノードをサンプリングする。random_samplingのため、偽陰性をサンプリングする可能性がある
    Args:
        num_pos: Number of positive samples.
        neg_sampling: NegativeSampling.
        num_nodes: Number of nodes.
    Returns:
        negative sampled nodes. shape: (num_neg,)
    """
    num_neg = math.ceil(num_pos * neg_sampling.amount)
    return neg_sampling.sample(num_neg, endpoint="dst", num_nodes=num_nodes)
