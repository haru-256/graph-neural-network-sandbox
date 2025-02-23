# distutils: language = c++

import numpy as np
cimport numpy as cnp
cimport cython
from libcpp.deque cimport deque
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef remove_edges(
    cnp.int64_t[:, ::1] edge_index,
    cnp.int64_t[:, ::1] edge_attr,
    cnp.int64_t[:, ::1] removed_edge_index,
):
    """Remove edges from edge_index.

    Args:
        edge_index: edge index, shape=(2, n) n: number of edges
        edge_attr: edge attributes, shape=(n, d) d: number of edge attributes
        removed_edge_index: target edge index to remove, shape=(2, m) m: number of edges to remove
    Returns:
        remaining_edge_index: edge index after removing edges, shape=(2, n_remaining)
        remaining_edge_attrs: edge attributes after removing edges, shape=(n_remaining, d)
    """
    cdef int num_edges = edge_index.shape[1]
    cdef int num_removed_edges = removed_edge_index.shape[1]
    cdef int d = edge_attr.shape[1]
    cdef int i, j, counter, flag, remaining
    cdef cnp.ndarray[cnp.int64_t, ndim=2] rem_edge_index, rem_edge_attr
    cdef cnp.int64_t* rem_e0
    cdef cnp.int64_t* rem_e1
    cdef int k

    # Get C-level pointers for faster access; the memoryviews are required to be contiguous in the last axis.
    cdef cnp.int64_t* e0 = &edge_index[0, 0]
    cdef cnp.int64_t* e1 = &edge_index[1, 0]
    cdef cnp.int64_t* r0 = &removed_edge_index[0, 0]
    cdef cnp.int64_t* r1 = &removed_edge_index[1, 0]

    # First pass: count how many edges are not removed.
    remaining = 0
    for i in range(num_edges):
        flag = 1
        for j in range(num_removed_edges):
            if e0[i] == r0[j] and e1[i] == r1[j]:
                flag = 0
                break
        if flag:
            remaining += 1

    # Allocate output arrays
    rem_edge_index = np.empty((2, remaining), dtype=np.int64)
    rem_edge_attr = np.empty((remaining, d), dtype=np.int64)

    rem_e0 = &rem_edge_index[0, 0]
    rem_e1 = &rem_edge_index[1, 0]
    counter = 0

    # Second pass: fill output arrays.
    for i in range(num_edges):
        flag = 1
        for j in range(num_removed_edges):
            if e0[i] == r0[j] and e1[i] == r1[j]:
                flag = 0
                break
        if flag:
            rem_e0[counter] = e0[i]
            rem_e1[counter] = e1[i]
            # Copy the corresponding row from edge_attr.
            for k in range(d):
                rem_edge_attr[counter, k] = edge_attr[i, k]
            counter += 1

    return rem_edge_index, rem_edge_attr

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef find_index(
    cnp.int64_t[:] node,
    cnp.int64_t[:] src,
    cnp.int64_t[:] dst_pos,
    cnp.int64_t[:, ::1] dst_neg,
):
    """
    Find the index of the node w,r.t src, dst_pos and dst_neg.
    Args:
        node: all nodes. shape: (num_nodes, )
        src: source nodes. shape: (num_src, )
        dst_pos: positive destination nodes. shape: (num_src, )
        dst_neg: negative destination nodes. shape: (num_src, num_neg)
    Returns:
        index: index of src, dst_pos, dst_neg. each shape: (num_src, ), (num_pos, ), (num_src, num_neg)
    """

    cdef int num_nodes = node.shape[0]
    cdef int num_src = src.shape[0]
    cdef int num_pos = num_src
    cdef int num_neg_per_pos = dst_neg.shape[1]
    cdef int i, j, k
    cdef cnp.ndarray[cnp.int64_t, ndim=1] index_src, index_dst_pos
    cdef cnp.ndarray[cnp.int64_t, ndim=2] index_dst_neg
    cdef int src_node, pos_node, neg_node
    cdef int num_found_src_nodes, num_found_pos_nodes, num_found_neg_nodes

    index_src = np.empty(num_src, dtype=np.int64)
    index_dst_pos = np.empty(num_pos, dtype=np.int64)
    index_dst_neg = np.empty((num_src, num_neg_per_pos), dtype=np.int64)

    num_found_src_nodes = 0
    num_found_pos_nodes = 0
    num_found_neg_nodes = 0
    for i in range(num_nodes):
        # src
        if num_found_src_nodes < num_src: # src nodeが見つかっていない場合
            for j in range(num_src):
                if src[j] == node[i]:
                    index_src[j] = i
                    num_found_src_nodes += 1
        # dst_pos
        if num_found_pos_nodes < num_pos: # pos nodeが見つかっていない場合
            for j in range(num_pos):
                if dst_pos[j] == node[i]:
                    index_dst_pos[j] = i
                    num_found_pos_nodes += 1
        # dst_neg
        if num_found_neg_nodes < num_src * num_neg_per_pos: # pos nodeが見つかっていない場合
            for j in range(num_src):
                for k in range(num_neg_per_pos):
                    if dst_neg[j, k] == node[i]:
                        index_dst_neg[j, k] = i
                        num_found_neg_nodes += 1

    return index_src, index_dst_pos, index_dst_neg


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple get_connected_edges(
    int seed,
    cnp.int64_t[:, ::1] edge_index,
    cnp.int64_t[:, ::1] edge_attr,
):
    """
    get connected edges from seed node.

    Args:
        seed: seed node. shape: (1, )
        edge_index: edge index. shape: (2, num_edges)
        edge_attr: edge attributes. shape: (num_edges, num_edge_attrs)

    Returns:
        connected_edge_index: connected edges. shape: (2, num_connected_edges)
        connected_edge_attr: connected edge attributes. shape: (num_connected_edges, num_edge_attrs)
    """
    cdef int num_edges = edge_index.shape[1]
    cdef int num_edge_attrs = edge_attr.shape[1]
    cdef int i, j, counter, num_connected_edges
    cdef cnp.int64_t[:, ::1] connected_edge_index, connected_edge_attr

    # Obtain C-level pointers for fast access.
    cdef cnp.int64_t* e0 = &edge_index[0, 0]
    cdef cnp.int64_t* e1 = &edge_index[1, 0]
    cdef cnp.int64_t* attr = &edge_attr[0, 0]

    # 1. Count the number of connected edges.
    num_connected_edges = 0
    for i in range(num_edges):
        if e0[i] == seed or e1[i] == seed:
            num_connected_edges += 1

    # If no connected edges are found, return empty arrays.
    if num_connected_edges == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, num_edge_attrs), dtype=np.int64)

    # 2. Allocate result arrays.
    connected_edge_index = np.empty((2, num_connected_edges), dtype=np.int64)
    connected_edge_attr = np.empty((num_connected_edges, num_edge_attrs), dtype=np.int64)
    cdef cnp.int64_t* ce0 = &connected_edge_index[0, 0]
    cdef cnp.int64_t* ce1 = &connected_edge_index[1, 0]

    # 3. Fill result arrays using C-level pointers.
    counter = 0
    for i in range(num_edges):
        if e0[i] == seed or e1[i] == seed:
            ce0[counter] = e0[i]
            ce1[counter] = e1[i]
            # Copy attributes using pointer arithmetic. Assumes edge_attr is contiguous.
            for j in range(num_edge_attrs):
                connected_edge_attr[counter, j] = attr[i * num_edge_attrs + j]
            counter += 1

    return connected_edge_index, connected_edge_attr

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple sample_one_hop_neighbors(
    int seed,
    cnp.int64_t[:, ::1] edge_index,
    cnp.int64_t[:, ::1] edge_attr,
    int num_neighbors,
):
    cdef cnp.int64_t[:] indices
    cdef cnp.int64_t* p_indices
    cdef cnp.int64_t[:, ::1] target_edge_index, target_edge_attr
    cdef int num_edge_attr, num_target_edges, num_sampled_neighbors
    cdef cnp.int64_t* target_e0
    cdef cnp.int64_t* target_e1
    cdef cnp.int64_t* sample_e0
    cdef cnp.int64_t* sample_e1
    cdef cnp.int64_t* target_attr
    cdef cnp.int64_t* sample_attr
    cdef cnp.int64_t[:] sampled_nodes
    cdef cnp.int64_t[:, ::1] sampled_edge_index, sampled_edge_attr
    cdef int i, j, idx, count

    # 1. Retrieve connected edges quickly.
    target_edge_index, target_edge_attr = get_connected_edges(seed, edge_index, edge_attr)
    num_target_edges = target_edge_index.shape[1]
    num_edge_attr = target_edge_attr.shape[1]
    if num_target_edges == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, num_edge_attr), dtype=np.int64),
            np.asarray(edge_index),
            np.asarray(edge_attr),
        )

    # 2. Perform neighbor sampling.
    if num_neighbors == -1:
        # Use all connected edges.
        sampled_edge_index = target_edge_index
        sampled_edge_attr = target_edge_attr
    else:
        num_sampled_neighbors = num_neighbors if num_neighbors < num_target_edges else num_target_edges
        sampled_edge_index = np.empty((2, num_sampled_neighbors), dtype=np.int64)
        sampled_edge_attr = np.empty((num_sampled_neighbors, num_edge_attr), dtype=np.int64)
        indices = np.random.choice(num_target_edges, num_sampled_neighbors, replace=False)
        p_indices = &indices[0]

        # Get C pointers for target and destination arrays.
        target_e0 = &target_edge_index[0, 0]
        target_e1 = &target_edge_index[1, 0]
        sample_e0 = &sampled_edge_index[0, 0]
        sample_e1 = &sampled_edge_index[1, 0]
        target_attr = &target_edge_attr[0, 0]
        sample_attr = &sampled_edge_attr[0, 0]

        # Copy selected edges and their attributes using pointer arithmetic.
        for i in range(num_sampled_neighbors):
            idx = p_indices[i]
            sample_e0[i] = target_e0[idx]
            sample_e1[i] = target_e1[idx]
            for j in range(num_edge_attr):
                sample_attr[i * num_edge_attr + j] = target_attr[idx * num_edge_attr + j]

    # 3. Remove sampled edges from the full graph.
    remained_edge_index, remained_edge_attr = remove_edges(
        edge_index=edge_index,
        edge_attr=edge_attr,
        removed_edge_index=sampled_edge_index,
    )

    # 4. Extract nodes from sampled edges (non-seed).
    cdef int num_sampled_edges = sampled_edge_index.shape[1]
    count = 0
    for i in range(num_sampled_edges):
        if sampled_edge_index[0, i] != seed:
            count += 1
        elif sampled_edge_index[1, i] != seed:
            count += 1
        else:
            raise ValueError(f"sampled_edge_index should be connected with seed node. {seed=}, {np.asarray(sampled_edge_index)[:, i]=}")
    sampled_nodes = np.empty(count, dtype=np.int64)
    count = 0
    for i in range(num_sampled_edges):
        if sampled_edge_index[0, i] != seed:
            sampled_nodes[count] = sampled_edge_index[0, i]
            count += 1
        elif sampled_edge_index[1, i] != seed:
            sampled_nodes[count] = sampled_edge_index[1, i]
            count += 1
        else:
            raise ValueError("sampled_edge_index should be from seed node to other node.")

    # return sampled_nodes, sampled_edge_index, sampled_edge_attr, remained_edge_index, remained_edge_attr
    return (
        np.asarray(sampled_nodes),
        np.asarray(sampled_edge_index),
        np.asarray(sampled_edge_attr),
        np.asarray(remained_edge_index),
        np.asarray(remained_edge_attr)
    )

cdef cnp.ndarray[cnp.int64_t, ndim=1] flatten_2d_array(cnp.ndarray[cnp.int64_t, ndim=2] arr):
    """
    Flatten 2D array to 1D array.

    Args:
        arr: 2D array. shape: (n, m)

    Returns:
        flat_arr: 1D array. shape: (n * m, )
    """
    cdef int n = arr.shape[0]
    cdef int m = arr.shape[1]
    cdef int i, j, counter
    cdef cnp.int64_t* a = &arr[0, 0]
    cdef cnp.ndarray[cnp.int64_t, ndim=1] flat_arr

    flat_arr = np.empty(n * m, dtype=np.int64)
    cdef cnp.int64_t* flat = &flat_arr[0]

    counter = 0
    for i in range(n):
        for j in range(m):
            flat[counter] = a[i * m + j]
            counter += 1

    return flat_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple neighbor_sampling_by_dfs(
    cnp.int64_t[:] seed, cnp.int64_t[:, ::1] edge_index, cnp.int64_t[:] num_neighbors
):
    """
    深さ優先探索で近傍nodeをサンプリングする (Optimized via an unordered_set for visited nodes)

    Args:
        seed: seed node. shape: (N, )
        edge_index: edge index. shape: (2, num_edges)
        num_neighbors: number of neighbors to sample per layer (hop)

    Returns:
        sampled_nodes: sampled nodes. shape=(N,)
        sampled_edge_indices: sampled edges. shape=(2, num_sampled_edges)
        sampled_edge_index_ptrs: indices of sampled edges. shape=(num_sampled_edges,)
    """
    cdef:
        int i, layer, max_layer, seed_node, node_val, num_sampled_edges, num_sampled, num_remained
        deque[int] seed_queue
        deque[int] layer_queue
        vector[int] sampled_edge_index_ptrs = vector[int]()
        vector[int] sampled_nodes_v = vector[int]()
        unordered_set[int] visited_nodes  # fast lookup for visited nodes

        cnp.int64_t[:, ::1] sampled_edge_index
        cnp.ndarray[cnp.int64_t, ndim=1] _sampled_nodes, _sampled_edge_index_ptr, _remained_edge_index_ptr
        cnp.ndarray[cnp.int64_t, ndim=2] _sampled_edge_index, _remained_edge_index, _sampled_edge_index_ptr_2d, _remained_edge_index_ptr_2d
        cnp.ndarray[cnp.int64_t, ndim=1] sampled_nodes_np, sampled_edge_index_ptrs_np
        cnp.int64_t* sampled_ptr
        cnp.int64_t* remained_ptr

    max_layer = num_neighbors.shape[0]

    # Initialize: add seed nodes into the queue and mark as visited.
    for i in range(seed.shape[0]):
        seed_queue.push_back(seed[i])
        layer_queue.push_back(0)
        if visited_nodes.find(seed[i]) == visited_nodes.end():
            visited_nodes.insert(seed[i])
            sampled_nodes_v.push_back(seed[i])

    # Initialize remained_edge_index and an auxiliary pointer array.
    _remained_edge_index = np.asarray(edge_index)
    _remained_edge_index_ptr = np.arange(edge_index.shape[1], dtype=np.int64)

    # Depth-first like neighbor sampling with layer tracking.
    while seed_queue.size() > 0:
        seed_node = seed_queue.front()
        seed_queue.pop_front()
        layer = layer_queue.front()
        layer_queue.pop_front()

        # layerがnum_neighborsの長さを超えた場合(= 指定されたhop数を超えた場合)はスキップ
        if layer >= max_layer:
            continue

        (
            _sampled_nodes,
            _sampled_edge_index,
            _sampled_edge_index_ptr_2d,
            _remained_edge_index,
            _remained_edge_index_ptr_2d
        ) = sample_one_hop_neighbors(
            seed=seed_node,
            edge_index=_remained_edge_index,
            edge_attr=_remained_edge_index_ptr.reshape(-1, 1),
            num_neighbors=num_neighbors[layer],
        )

        if _sampled_nodes.size == 0:  # samplingされたnodeがない場合
            continue
        if _remained_edge_index.size == 0: # 今後samplingされうるedgeがない場合
            break

        _sampled_edge_index_ptr = cnp.PyArray_Flatten(_sampled_edge_index_ptr_2d, cnp.NPY_CORDER)
        # _sampled_edge_index_ptr = _sampled_edge_index_ptr_2d.reshape(-1)
        # _sampled_edge_index_ptr = flatten_2d_array(_sampled_edge_index_ptr_2d)
        _remained_edge_index_ptr = cnp.PyArray_Flatten(_remained_edge_index_ptr_2d, cnp.NPY_CORDER)
        # _remained_edge_index_ptr = _remained_edge_index_ptr_2d.reshape(-1)
        # _remained_edge_index_ptr = flatten_2d_array(_remained_edge_index_ptr_2d)

        # Use unordered_set to quickly check and register new nodes.
        for i in range(_sampled_nodes.shape[0]):
            node_val = _sampled_nodes[i]
            if visited_nodes.find(node_val) == visited_nodes.end(): # すでに訪れたノードは再度訪れない
                visited_nodes.insert(node_val)
                seed_queue.push_back(node_val)
                layer_queue.push_back(layer + 1)
                sampled_nodes_v.push_back(node_val)

        for i in range(_sampled_edge_index_ptr.shape[0]):
            sampled_edge_index_ptrs.push_back(_sampled_edge_index_ptr[i])

    # Allocate sampled_edge_index using the collected edge indices.
    num_sampled_edges = sampled_edge_index_ptrs.size()
    sampled_edge_index = np.empty((2, num_sampled_edges), dtype=np.int64)
    for i in range(num_sampled_edges):
        sampled_edge_index[0, i] = edge_index[0, sampled_edge_index_ptrs[i]]
        sampled_edge_index[1, i] = edge_index[1, sampled_edge_index_ptrs[i]]

    # Convert the vector of sampled nodes to a NumPy array.
    sampled_nodes_np = np.empty(sampled_nodes_v.size(), dtype=np.int64)
    for i in range(sampled_nodes_v.size()):
        sampled_nodes_np[i] = sampled_nodes_v[i]
    sampled_edge_index_ptrs_np = np.empty(sampled_edge_index_ptrs.size(), dtype=np.int64)
    for i in range(sampled_edge_index_ptrs.size()):
        sampled_edge_index_ptrs_np[i] = sampled_edge_index_ptrs[i]

    return sampled_nodes_np, sampled_edge_index, sampled_edge_index_ptrs_np
