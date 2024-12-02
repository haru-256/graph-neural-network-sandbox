import torch
from torch_geometric.data import Data

from sampler.loader import daemon_link_neighbor_loader_factory


def test_daemon_link_neighbor_loader(data: Data):
    x = data.x
    edge_index = data.edge_index
    assert x is not None and edge_index is not None

    num_neighbors = [2, 2]
    num_neg_samples = 3
    batch_size = 1
    loader = daemon_link_neighbor_loader_factory(
        data,
        batch_size=batch_size,
        num_neighbors=num_neighbors,
        shuffle=False,
        num_neg_samples=num_neg_samples,
        seed=1026,
    )
    batch = next(iter(loader))
    node_index = batch.x.squeeze(1)
    assert torch.equal(node_index[batch.edge_index], edge_index[:, batch.e_id])
    assert torch.equal(batch.edge_index_type, data.edge_index_type[batch.e_id])
    assert (
        batch.src_index.size() == batch.dst_pos_index.size() == torch.Size([batch_size])
    )
    assert batch.dst_neg_index.size() == torch.Size([batch_size, num_neg_samples])

    num_neighbors = [2, 2]
    num_neg_samples = 3
    batch_size = 2
    loader = daemon_link_neighbor_loader_factory(
        data,
        batch_size=batch_size,
        num_neighbors=num_neighbors,
        shuffle=False,
        num_neg_samples=num_neg_samples,
        seed=1026,
    )
    batch = next(iter(loader))
    node_index = batch.x.squeeze(1)
    assert torch.equal(node_index[batch.edge_index], edge_index[:, batch.e_id])
    assert torch.equal(batch.edge_index_type, data.edge_index_type[batch.e_id])
    assert (
        batch.src_index.size() == batch.dst_pos_index.size() == torch.Size([batch_size])
    )
    assert batch.dst_neg_index.size() == torch.Size([batch_size, num_neg_samples])
