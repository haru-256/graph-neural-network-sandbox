import torch

from sampler.utils import get_connected_edges, get_connected_edges_v2, remove_edges_v2


def test_remove_edges_v2():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    edge_attr = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    removed_edge_index = torch.tensor([[0, 2], [1, 3]])
    remaining_edge_index, remaining_edge_attr = remove_edges_v2(
        edge_index=edge_index,
        edge_attr=edge_attr,
        removed_edge_index=removed_edge_index,
    )
    assert torch.equal(remaining_edge_index, torch.tensor([[1, 3], [2, 0]]))
    assert torch.equal(remaining_edge_attr, torch.tensor([[3, 4], [7, 8]]))


def test_get_connected_edges():
    seed = 0
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    edge_attr = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    target_edge_index, target_edge_attr = get_connected_edges(
        seed=seed,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    assert torch.equal(target_edge_index, torch.tensor([[0, 3], [1, 0]]))
    assert torch.equal(target_edge_attr, torch.tensor([[1, 2], [7, 8]]))


def test_get_connected_edges_v2():
    seed = 0
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    edge_attr = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    target_edge_index, target_edge_attr = get_connected_edges_v2(
        seed=seed,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    assert torch.equal(target_edge_index, torch.tensor([[0, 3], [1, 0]]))
    assert torch.equal(target_edge_attr, torch.tensor([[1, 2], [7, 8]]))

    # no connected edges
    seed = 0
    edge_index = torch.tensor([[1, 2, 3], [2, 3, 1]])
    edge_attr = torch.tensor([[3, 4], [5, 6], [7, 8]])
    target_edge_index, target_edge_attr = get_connected_edges_v2(
        seed=seed,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    assert target_edge_index.numel() == 0 and target_edge_attr.numel() == 0
