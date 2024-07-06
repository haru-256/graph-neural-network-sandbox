import timeit

import torch
from torch_geometric.utils import coalesce


def my_way() -> torch.Tensor:
    remaining_edge_index_list = [
        (src, dst)
        for src, dst in edge_index.T
        if not (src in source_index and dst in target_index)
    ]
    remaining_edge_index = torch.tensor(remaining_edge_index_list).T
    return remaining_edge_index


def better_way() -> torch.Tensor:
    removed_edge_index = torch.stack([source_index, target_index], dim=0)
    all_edge_index = torch.cat([edge_index, removed_edge_index], dim=1)
    # mark removed edges as 1 and 0 otherwise
    all_edge_weights = torch.cat(
        [torch.zeros(edge_index.size(1)), torch.ones(removed_edge_index.size(1))]
    )
    all_edge_index, all_edge_weights = coalesce(all_edge_index, all_edge_weights)
    # remove edges indicated by 1
    remaining_edge_index = all_edge_index[:, all_edge_weights == 0]
    return remaining_edge_index


n = 1000
src_index = torch.arange(n)
target_index = torch.arange(n) + 1
edge_index = torch.stack([src_index, target_index], dim=0)
source_index = torch.tensor([1, 2, 3, 4, 5])
target_index = source_index + 1
duration1 = timeit.timeit("my_way()", globals=globals(), number=1000)
duration2 = timeit.timeit("better_way()", globals=globals(), number=1000)
print(
    f"my_way: {duration1:.6f}, better_way: {duration2:.6f}, speedup: {duration1/duration2:.2f}x"
)
