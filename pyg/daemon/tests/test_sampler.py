import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.sampler import EdgeSamplerInput, NegativeSampling

from sampler.sampler import (
    DAEMONNeighborSampler,
    approx_neg_sample,
    find_index,
    sample_one_hop_neighbors,
)


def test_approx_neg_sample():
    # negative sample: 1
    num_pos = 5
    num_neg = 1
    neg_sampling = NegativeSampling(mode="triplet", amount=num_neg)
    num_nodes = 10
    expected_size = torch.Size([num_pos])
    expected_range = set(range(num_nodes))
    actual = approx_neg_sample(num_pos, neg_sampling, num_nodes)
    assert (
        actual.size() == expected_size
    ), f"The shape of the output is incorrect. {actual.size()}"
    assert (
        set(actual.tolist()) <= expected_range
    ), f"The output contains an invalid node id. {actual}"

    # negative sample: 2
    num_pos = 5
    num_neg = 2
    neg_sampling = NegativeSampling(mode="triplet", amount=2)
    num_nodes = 10
    expected_size = torch.Size([num_pos * num_neg])
    expected_range = set(range(num_nodes))
    actual = approx_neg_sample(num_pos, neg_sampling, num_nodes)
    assert (
        actual.size() == expected_size
    ), f"The shape of the output is incorrect. {actual.size()}"
    assert (
        set(actual.tolist()) <= expected_range
    ), f"The output contains an invalid node id. {actual}"


def test_find_index():
    # neg sample size = 1
    node = torch.tensor([1, 2, 3, 4, 5, 10, 11, 12])
    src = torch.tensor([1, 2])
    dst_pos = torch.tensor([3, 4])
    dst_neg = torch.tensor([[10], [11]])
    exp_src_index = torch.tensor([0, 1])
    exp_dst_pos_index = torch.tensor([2, 3])
    exp_dst_neg_index = torch.tensor([[5], [6]])
    actual_src_index, actual_dst_pos_index, actual_dst_neg_index = find_index(
        node, src, dst_pos, dst_neg
    )
    assert torch.equal(
        actual_src_index, exp_src_index
    ), f"Source index is incorrect. {actual_src_index}"
    assert torch.equal(
        actual_dst_pos_index, exp_dst_pos_index
    ), f"Positive destination index is incorrect. {actual_dst_pos_index}"
    assert torch.equal(
        actual_dst_neg_index, exp_dst_neg_index
    ), f"Negative destination index is incorrect. {actual_dst_neg_index}"

    # neg sample size = 2
    node = torch.tensor([1, 2, 3, 4, 5, 10, 11, 12])
    src = torch.tensor([1, 2])
    dst_pos = torch.tensor([3, 4])
    dst_neg = torch.tensor([[10, 11], [11, 12]])
    exp_src_index = torch.tensor([0, 1])
    exp_dst_pos_index = torch.tensor([2, 3])
    exp_dst_neg_index = torch.tensor([[5, 6], [6, 7]])
    actual_src_index, actual_dst_pos_index, actual_dst_neg_index = find_index(
        node, src, dst_pos, dst_neg
    )
    assert torch.equal(
        actual_src_index, exp_src_index
    ), f"Source index is incorrect. {actual_src_index}"
    assert torch.equal(
        actual_dst_pos_index, exp_dst_pos_index
    ), f"Positive destination index is incorrect. {actual_dst_pos_index}"
    assert torch.equal(
        actual_dst_neg_index, exp_dst_neg_index
    ), f"Negative destination index is incorrect. {actual_dst_neg_index}"


def test_sample_one_hop():
    seed = 0
    edge_index = torch.tensor(
        [
            [0, 4],
            [0, 5],
            [1, 0],
            [2, 0],
            [3, 0],
        ],
        dtype=torch.long,
    ).T.contiguous()
    edge_attr = None
    num_neighbor = 3
    generator = torch.Generator()
    generator.manual_seed(1124)
    exp_nodes = torch.tensor([1, 3, 5])
    exp_edge_index = torch.tensor(
        [
            [1, 0],
            [0, 5],
            [3, 0],
        ],
        dtype=torch.long,
    ).T.contiguous()
    exp_remained_edge_index = torch.tensor(
        [
            [0, 4],
            [2, 0],
        ],
        dtype=torch.long,
    ).T.contiguous()
    (
        actual_nodes,
        actual_edge_index,
        actual_edge_attr,
        actual_remained_edge_index,
        actual_remained_edge_attr,
    ) = sample_one_hop_neighbors(
        seed=seed,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_neighbor=num_neighbor,
        generator=generator,
    )
    assert torch.equal(actual_nodes, exp_nodes)
    assert torch.equal(actual_edge_index, exp_edge_index)
    assert actual_edge_attr is None
    assert torch.equal(actual_remained_edge_index, exp_remained_edge_index)
    assert actual_remained_edge_attr is None

    # edge_attr is not None
    seed = 0
    edge_index = torch.tensor(
        [
            [0, 4],
            [0, 5],
            [1, 0],
            [2, 0],
            [3, 0],
        ],
        dtype=torch.long,
    ).T.contiguous()
    edge_attr = torch.tensor(
        [
            [0.1],
            [0.2],
            [0.3],
            [0.4],
            [0.5],
        ]
    )
    num_neighbor = 3
    generator = torch.Generator()
    generator.manual_seed(1124)
    exp_nodes = torch.tensor([1, 3, 5])
    exp_edge_index = torch.tensor(
        [
            [1, 0],
            [0, 5],
            [3, 0],
        ],
        dtype=torch.long,
    ).T.contiguous()
    exp_edge_attr = torch.tensor(
        [
            [0.3],
            [0.2],
            [0.5],
        ]
    )
    exp_remained_edge_index = torch.tensor(
        [
            [0, 4],
            [2, 0],
        ],
        dtype=torch.long,
    ).T.contiguous()
    exp_remained_edge_attr = torch.tensor(
        [
            [0.1],
            [0.4],
        ]
    )
    (
        actual_nodes,
        actual_edge_index,
        actual_edge_attr,
        actual_remained_edge_index,
        actual_remained_edge_attr,
    ) = sample_one_hop_neighbors(
        seed=seed,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_neighbor=num_neighbor,
        generator=generator,
    )
    assert torch.equal(actual_nodes, exp_nodes)
    assert torch.equal(actual_edge_index, exp_edge_index)
    assert torch.equal(actual_edge_attr, exp_edge_attr)
    assert torch.equal(actual_remained_edge_index, exp_remained_edge_index)
    assert torch.equal(actual_remained_edge_attr, exp_remained_edge_attr)


@pytest.fixture()
def data():
    # graph follows the below image.
    # https://github.com/pyg-team/pytorch_geometric/discussions/9816#discussion-7583341
    x = torch.tensor([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    edge_index = torch.tensor(
        [[0, 4], [0, 5], [1, 0], [2, 0], [3, 0], [1, 6]]
    ).T.contiguous()
    edge_index_type = torch.tensor([0, 1, 0, 1, 0])
    return Data(x=x, edge_index=edge_index, edge_index_type=edge_index_type)


class TestDAEMONNeighborSampler:
    def test_sample(self, data: Data):
        x = data.x
        edge_index = data.edge_index
        assert x is not None and edge_index is not None

        # num_src = 1
        # num_neg_sample_size = 1
        src = torch.tensor([0])
        dst_pos = torch.tensor([4])
        dst_neg = torch.tensor([[5]])
        sampler = DAEMONNeighborSampler(data, num_neighbors=[2], seed=1026)
        out = sampler._sample(src, dst_pos, dst_neg)
        exp_node_range = set(x.flatten().tolist())
        assert set(out.node.tolist()) <= exp_node_range
        actual_edge_index = torch.vstack([out.row, out.col])
        actual_e_id = out.edge
        assert torch.equal(out.node[actual_edge_index], edge_index[:, actual_e_id])
        actual_src_index, actual_dst_pos_index, actual_dst_neg_index = out.metadata
        assert torch.equal(out.node[actual_src_index], src)
        assert torch.equal(out.node[actual_dst_pos_index], dst_pos)
        assert torch.equal(out.node[actual_dst_neg_index], dst_neg)

        # num_src = 1
        # num_neg_sample_size = 2
        src = torch.tensor([0])
        dst_pos = torch.tensor([4])
        dst_neg = torch.tensor([[5, 6]])
        sampler = DAEMONNeighborSampler(data, num_neighbors=[2], seed=1026)
        out = sampler._sample(src, dst_pos, dst_neg)
        exp_node_range = set(x.flatten().tolist())
        assert set(out.node.tolist()) <= exp_node_range
        actual_edge_index = torch.vstack([out.row, out.col])
        actual_e_id = out.edge
        assert torch.equal(out.node[actual_edge_index], edge_index[:, actual_e_id])
        actual_src_index, actual_dst_pos_index, actual_dst_neg_index = out.metadata
        assert torch.equal(out.node[actual_src_index], src)
        assert torch.equal(out.node[actual_dst_pos_index], dst_pos)
        assert torch.equal(out.node[actual_dst_neg_index], dst_neg)

        # num_src = 2
        # num_neg_sample_size = 1
        # num_neg_sample_size = 1
        src = torch.tensor([0, 1])
        dst_pos = torch.tensor([4, 2])
        dst_neg = torch.tensor([[5], [3]])
        sampler = DAEMONNeighborSampler(data, num_neighbors=[2], seed=1026)
        out = sampler._sample(src, dst_pos, dst_neg)
        exp_node_range = set(x.flatten().tolist())
        assert set(out.node.tolist()) <= exp_node_range
        actual_edge_index = torch.vstack([out.row, out.col])
        actual_e_id = out.edge
        assert torch.equal(out.node[actual_edge_index], edge_index[:, actual_e_id])
        actual_src_index, actual_dst_pos_index, actual_dst_neg_index = out.metadata
        assert torch.equal(out.node[actual_src_index], src)
        assert torch.equal(out.node[actual_dst_pos_index], dst_pos)
        assert torch.equal(out.node[actual_dst_neg_index], dst_neg)

        # num_src = 2
        # num_neg_sample_size = 1
        # num_neg_sample_size = 1
        # num_neighbor_layer = 2
        src = torch.tensor([0, 1])
        dst_pos = torch.tensor([4, 2])
        dst_neg = torch.tensor([[5], [3]])
        sampler = DAEMONNeighborSampler(data, num_neighbors=[2, 2], seed=1026)
        out = sampler._sample(src, dst_pos, dst_neg)
        exp_node_range = set(x.flatten().tolist())
        assert set(out.node.tolist()) <= exp_node_range
        actual_edge_index = torch.vstack([out.row, out.col])
        actual_e_id = out.edge
        assert torch.equal(out.node[actual_edge_index], edge_index[:, actual_e_id])
        actual_src_index, actual_dst_pos_index, actual_dst_neg_index = out.metadata
        assert torch.equal(out.node[actual_src_index], src)
        assert torch.equal(out.node[actual_dst_pos_index], dst_pos)
        assert torch.equal(out.node[actual_dst_neg_index], dst_neg)

    def test_sample_from_edges(self, data: Data):
        x = data.x
        edge_index = data.edge_index
        assert x is not None and edge_index is not None

        # num_src = 1
        # num_neg_sample_size = 1
        src = torch.tensor([0])
        dst_pos = torch.tensor([4])
        inputs = EdgeSamplerInput(input_id=None, row=src, col=dst_pos)
        sampler = DAEMONNeighborSampler(data, num_neighbors=[2], seed=1026)
        neg_sampling = NegativeSampling(mode="triplet", amount=1)
        out = sampler.sample_from_edges(inputs, neg_sampling)
        exp_node_range = set(x.flatten().tolist())
        assert set(out.node.tolist()) <= exp_node_range
        actual_edge_index = torch.vstack([out.row, out.col])
        actual_e_id = out.edge
        assert torch.equal(out.node[actual_edge_index], edge_index[:, actual_e_id])
        actual_src_index, actual_dst_pos_index, _ = out.metadata
        assert torch.equal(out.node[actual_src_index], src)
        assert torch.equal(out.node[actual_dst_pos_index], dst_pos)