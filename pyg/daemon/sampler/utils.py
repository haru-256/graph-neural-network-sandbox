from typing import Optional, overload

import torch
from torch_geometric.utils import coalesce

from .cython.cython_fn import get_connected_edges as _get_connected_edges_v2
from .cython.cython_fn import remove_edges as _remove_edges_v2


@overload
def remove_edges(
    edge_index: torch.Tensor,
    removed_edge_index: torch.Tensor,
    edge_attr: None = None,
) -> torch.Tensor: ...


@overload
def remove_edges(
    edge_index: torch.Tensor,
    removed_edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]: ...


def remove_edges(
    edge_index: torch.Tensor,
    removed_edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
):
    """Remove edges from edge_index.

    Args:
        edge_index: edge index, shape=(2, n) n: number of edges
        edge_attr: edge attributes, shape=(n, d) d: number of edge attributes
        removed_edge_index: target edge index to remove, shape=(2, m) m: number of edges to remove, default=None
    Returns:
        remaining_edge_index: edge index after removing edges
        remaining_edge_attrs: edge attributes after removing edges
    """
    # shape: (2, n + m)
    all_edge_index = torch.cat([edge_index, removed_edge_index], dim=1)
    # mark removed edges as 1 and 0 otherwise, shape: (n + m, 1)
    all_edge_removed_flg = (
        torch.cat(
            [
                torch.zeros(edge_index.size(1)),
                torch.ones(removed_edge_index.size(1)),
            ]
        )
        .long()
        .reshape(-1, 1)
        .to(all_edge_index.device)
    )
    # maxを取ることでedge_index_typeを保持するため、removed_edge_indexのedge_index_typeは0にする
    if edge_attr is not None:
        all_edge_attrs_ = torch.cat(
            [
                edge_attr,
                torch.zeros(removed_edge_index.size(1), edge_attr.size(1)),
            ],
            dim=0,
        ).to(all_edge_index.device)  # shape: (n + m, d)
        all_edge_attrs = torch.cat(
            [all_edge_removed_flg, all_edge_attrs_], dim=1
        )  # shape: (n + m, d + 1)
    else:
        all_edge_attrs = all_edge_removed_flg
    all_edge_index, all_edge_attrs = coalesce(
        all_edge_index, all_edge_attrs, reduce="max"
    )
    # remove edges indicated by 1
    if edge_attr is not None:
        mask = all_edge_attrs[:, 0] == 0
        remaining_edge_index = all_edge_index[:, mask]
        remaining_edge_attrs = all_edge_attrs[mask, 1:]
        return remaining_edge_index, remaining_edge_attrs
    else:
        mask = all_edge_attrs.squeeze() == 0
        remaining_edge_index = all_edge_index[:, mask]
        return remaining_edge_index


def remove_edges_v2(
    edge_index: torch.Tensor,
    removed_edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
):
    """Remove edges from edge_index.

    Args:
        edge_index: edge index, shape=(2, n) n: number of edges
        edge_attr: edge attributes, shape=(n, d) d: number of edge attributes
        removed_edge_index: target edge index to remove, shape=(2, m) m: number of edges to remove, default=None
    Returns:
        remaining_edge_index: edge index after removing edges
        remaining_edge_attrs: edge attributes after removing edges
    """
    if edge_attr is None:
        raise NotImplementedError("edge_attr is required")
    edge_index, edge_attr = _remove_edges_v2(
        edge_index=edge_index.numpy(),
        edge_attr=edge_attr.numpy(),
        removed_edge_index=removed_edge_index.numpy(),
    )
    return torch.tensor(edge_index), torch.tensor(edge_attr)


def check_has_edge(
    edge_index: torch.Tensor, edges_to_check: torch.Tensor
) -> torch.Tensor:
    """Check if `edges_to_check` is in edge_index.

    Args:
        edge_index: edge index, (2, n), n: number of edges
        edges_to_check: target edge index to check, (2, m), m: number of edges to check
    Returns:
        bool tensor indicating if edges_to_check is in edge_index
    """
    # shape: (1, n, 2) vs (m, 1, 2)
    edge_exists = (
        (edge_index.T.unsqueeze(0) == edges_to_check.T.unsqueeze(1))
        .all(dim=-1)
        .any(dim=-1)
    )
    return edge_exists


def get_connected_edges(
    seed: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    get connected edges from seed node.

    Args:
        seed: seed node. shape: (1, )
        edge_index: edge index. shape: (2, num_edges)
        edge_attr: edge attributes. shape: (num_edges, num_edge_attrs)

    Returns:
        connected_edges: connected edges. shape: (2, num_connected_edges)
        connected_edge_attrs: connected edge attributes. shape: (num_connected_edges, num_edge_attrs)
    """
    mask = (edge_index == seed).any(dim=0)  # shape: (num_edges, )
    target_edge_index = edge_index[:, mask]
    target_edge_attr = edge_attr[mask] if edge_attr is not None else None
    return target_edge_index, target_edge_attr


def get_connected_edges_v2(
    seed: int,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    get connected edges from seed node.

    Args:
        seed: seed node.
        edge_index: edge index. shape: (2, num_edges)
        edge_attr: edge attributes. shape: (num_edges, num_edge_attrs)

    Returns:
        connected_edges: connected edges. shape: (2, num_connected_edges)
        connected_edge_attrs: connected edge attributes. shape: (num_connected_edges, num_edge_attrs)
    """
    if edge_attr is None:
        raise NotImplementedError("edge_attr is required")
    connected_edge_index, connected_edge_attr = _get_connected_edges_v2(
        seed=seed,
        edge_index=edge_index.numpy(),
        edge_attr=edge_attr.numpy(),
    )
    return torch.tensor(connected_edge_index, dtype=edge_index.dtype), torch.tensor(
        connected_edge_attr, dtype=edge_attr.dtype
    )
