import numpy as np
cimport numpy as cnp
cimport cython

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
    cdef cnp.ndarray[cnp.int64_t, ndim=1] index_src, index_pos
    cdef cnp.ndarray[cnp.int64_t, ndim=2] index_neg
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

