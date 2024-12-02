import pytest
import torch
from torch_geometric.data import Data


@pytest.fixture()
def data():
    # graph follows the below image.
    # https://github.com/pyg-team/pytorch_geometric/discussions/9816#discussion-7583341
    x = torch.tensor([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    edge_index = torch.tensor(
        [[0, 4], [0, 5], [1, 0], [2, 0], [3, 0], [1, 6]]
    ).T.contiguous()
    edge_index_type = torch.tensor([0, 1, 0, 1, 0])
    edge_label_index = torch.tensor([[0, 4], [1, 6]]).T.contiguous()
    return Data(
        x=x,
        edge_index=edge_index,
        edge_index_type=edge_index_type,
        edge_label_index=edge_label_index,
    )
