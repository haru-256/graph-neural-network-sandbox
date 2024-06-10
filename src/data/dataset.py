from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub


class MyKarateClub:
    def __init__(self):
        self.dataset = KarateClub()
        self.data = self.__create_graph()
        self.node_index_map = self.__build_node_index_map()

    def __build_node_index_map(self) -> dict[int, int]:
        node_index_map = {
            node: i for i, node in enumerate(self.data.x.squeeze().tolist())
        }
        return node_index_map

    def __create_graph(self) -> Data:
        data = self.dataset._data
        data.x = torch.arange(data.num_nodes).reshape(-1, 1)
        return data

    def get_summary(self) -> Any:
        return self.dataset.get_summary()
