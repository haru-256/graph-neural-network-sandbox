import torch

from sampler.utils import remove_edges_v2


def test_remove_edges_v2():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    edge_attr = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    removed_edge_index = torch.tensor([[0, 2], [1, 3]])
    # breakpoint()
    remaining_edge_index, remaining_edge_attr = remove_edges_v2(
        edge_index=edge_index,
        edge_attr=edge_attr,
        removed_edge_index=removed_edge_index,
    )
    assert torch.equal(remaining_edge_index, torch.tensor([[1, 3], [2, 0]]))
    assert torch.equal(remaining_edge_attr, torch.tensor([[3, 4], [7, 8]]))
