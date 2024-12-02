from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling

from .sampler import DAEMONNeighborSampler


def daemon_link_neighbor_loader_factory(
    data: Data,
    num_neighbors: list[int],
    batch_size: int,
    shuffle: bool = True,
    num_neg_samples: int = 1,
    seed: int = 1026,
) -> LinkNeighborLoader:
    """Create a LinkNeighborLoader for DAEMON.

    Args:
        data: Graph data.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data. Defaults to True.
        num_neg_samples: number of negative sample size. Defaults to 1.
        seed: random seed for DAEMONNeighborSampler. Defaults to 1026.

    Returns:
        _description_
    """
    neg_sampling = NegativeSampling(mode="triplet", amount=num_neg_samples)
    loader = LinkNeighborLoader(
        data=data,
        edge_label_index=data.edge_label_index,
        edge_label=None,
        batch_size=batch_size,
        shuffle=shuffle,
        num_neighbors=num_neighbors,
        neighbor_sampler=DAEMONNeighborSampler(
            data, num_neighbors=num_neighbors, seed=seed
        ),
        neg_sampling=neg_sampling,
    )

    return loader
