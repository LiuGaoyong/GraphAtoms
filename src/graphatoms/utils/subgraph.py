import array_api_extra as xpx

from graphatoms.arrayapi import Array, ArrayNamespace, get_namespace


def maybe_num_nodes(edge_index: Array, num_nodes: int | None = None) -> int:
    if num_nodes is not None:
        return num_nodes
    xp: ArrayNamespace = get_namespace(edge_index)
    return int(xp.max(edge_index)) + 1


def index_to_mask(index: Array, size: int) -> Array:
    xp: ArrayNamespace = get_namespace(index)
    mask = xp.zeros((size,), dtype=bool)
    return xpx.at(mask, index).set(True)  # type: ignore


def map_index(
    src: Array,
    index: Array,
    max_index: int | Array | None = None,
    inclusive: bool = False,
) -> tuple[Array, Array | None]:
    r"""Maps indices in :obj:`src` to the positional value of their
    corresponding occurrence in :obj:`index`.
    Indices must be strictly non-negative.

    This implementation uses only Array API standard functions, making it
    backend-agnostic (NumPy, PyTorch, JAX, CuPy, etc.).

    Args:
        src (Array): The source array to map.
        index (Array): The index array that denotes the new mapping.
        max_index (int, optional): The maximum index value.
            (default :obj:`None`)
        inclusive (bool, optional): If set to :obj:`True`, it is assumed that
            every entry in :obj:`src` has a valid entry in :obj:`index`.
            Can speed-up computation. (default: :obj:`False`)

    :rtype: (:class:`Array`, :class:`Array`)

    Examples:
        >>> src = torch.tensor([2, 0, 1, 0, 3])
        >>> index = torch.tensor([3, 2, 0, 1])
        >>> map_index(src, index)
        (tensor([1, 2, 3, 2, 0]), tensor([True, True, True, True, True]))
    """
    xp: ArrayNamespace = get_namespace(src, index)

    # 1. Type checks
    if xp.isdtype(src.dtype, "real floating"):
        raise ValueError(f"Expected 'src' to be an index (got '{src.dtype}')")
    if xp.isdtype(index.dtype, "real floating"):
        raise ValueError(
            f"Expected 'index' to be an index (got '{index.dtype}')"
        )

    # 2. Determine max_index
    if max_index is None:
        max_val = xp.max(xp.stack([xp.max(src), xp.max(index)]))
        max_index = int(max_val)  # scalar integer
    else:
        max_index = int(max_index)  # convert if it's a 0‑D array

    # 3. Build lookup table via direct indexing (hash‑map style)
    assoc = xp.full((max_index + 1,), -1, dtype=src.dtype)
    assoc[index] = xp.arange(index.shape[0], dtype=src.dtype)  # type: ignore

    # 4. Map src indices
    out = assoc[src]

    # 5. Handle inclusive flag and mask
    if inclusive:
        if xp.any(out == -1):
            raise ValueError(
                "Found invalid entries in 'src' that do not have "
                "a corresponding entry in 'index'. Set `inclusive=False` "
                "to ignore these entries."
            )
        return out, None
    else:
        mask = out != -1
        return out[mask], mask


def subgraph(
    subset: Array | list[int],
    edge_index: Array,
    edge_attr: Array | None = None,
    relabel_nodes: bool = False,
    num_nodes: int | None = None,
    *,
    return_edge_mask: bool = False,
) -> tuple[Array, Array | None, Array | None]:
    r"""Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.

    Args:
        subset (LongTensor, BoolTensor or [int]): The nodes to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max(edge_index) + 1`. (default: :obj:`None`)
        return_edge_mask (bool, optional): If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:
        >>> import numpy as xp
        >>> edge_index = xp.array([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        ...                        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5]])
        >>> edge_attr = xp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        >>> subset = xp.array([3, 4, 5])
        >>> subgraph(subset, edge_index, edge_attr)
        (array([[3, 4, 4, 5],
                [4, 3, 5, 4]]),
        array([ 7,  8,  9, 10]))

        >>> subgraph(subset, edge_index, edge_attr, return_edge_mask=True)
        (array([[3, 4, 4, 5],
                [4, 3, 5, 4]]),
        array([ 7,  8,  9, 10]),
        array([False, False, False, False, False, False,  True,
                True,  True,  True,  False, False]))
    """
    xp: ArrayNamespace = get_namespace(edge_index)
    assert 2 in xp.shape(edge_index), "edge_index must be 2D"  # type: ignore
    if xp.shape(edge_index)[0] != 2:  # type: ignore
        edge_index = edge_index.T
        _CONVERT = True
    else:
        _CONVERT = False

    if isinstance(subset, (list, tuple)):
        subset = xp.asarray(subset, dtype=xp.int64)

    if subset.dtype != xp.bool:  # type: ignore
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        node_mask = index_to_mask(subset, size=num_nodes)  # type: ignore
    else:
        num_nodes = subset.shape[0]  # type: ignore
        node_mask = subset
        subset = xp.where(node_mask)[0]  # type: ignore

    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        edge_index_flat, _ = map_index(
            xp.reshape(edge_index, (-1,)),
            subset,  # type: ignore
            max_index=num_nodes,
            inclusive=True,
        )
        edge_index = xp.reshape(edge_index_flat, (2, -1))

    if _CONVERT:
        edge_index = edge_index.T
    if return_edge_mask:
        return edge_index, edge_attr, edge_mask  # type: ignore
    else:
        return edge_index, edge_attr, None


if __name__ == "__main__":
    import numpy as xp

    edge_index = xp.array(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5],
        ]
    )
    edge_attr = xp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    subset = xp.array([3, 4, 5])
    edge_index, edge_attr, edge_mask = subgraph(
        subset,  # type: ignore
        edge_index,  # type: ignore
        edge_attr,  # type: ignore
        relabel_nodes=True,
        return_edge_mask=True,
    )
    print(edge_index)
    print(edge_attr)
    print(edge_mask)
