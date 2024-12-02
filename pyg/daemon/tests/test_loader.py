from torch_geometric.data import Data

from sampler.loader import daemon_link_neighbor_loader_factory


def test_daemon_link_neighbor_loader(data: Data):
    x = data.x
    edge_index = data.edge_index
    assert x is not None and edge_index is not None
    num_neighbors = [2, 2]

    loader = daemon_link_neighbor_loader_factory(
        data,
        batch_size=1,
        num_neighbors=num_neighbors,
        shuffle=False,
        num_neg_samples=1,
        seed=1026,
    )
    batch = next(iter(loader))
    print(batch)
