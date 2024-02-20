from ecnf.utils.numerical import safe_norm


def get_edge_vectors_and_lengths(positions,   # (n_nodes, dim)
                                 edge_index,  # (2, n_edges)
                                 shifts,      # (n_edges, dim)
                                 normalize=False,
                                 eps=1e-9):
    senders, receivers = edge_index
    vectors = positions[receivers] - positions[senders] + shifts  # (n_edges, dim)
    lengths = safe_norm(vectors, axis=-1, keepdims=True)  # (n_edges, 1)
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths