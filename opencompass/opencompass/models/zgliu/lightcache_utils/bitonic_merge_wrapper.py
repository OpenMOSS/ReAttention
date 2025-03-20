import triton
import triton.language as tl


@triton.jit
def _compare_and_swap(x, x_indices, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = tl.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    # actual compare-and-swap
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    cond = (left < right)
    ix = x.to(idtype, bitcast=True)
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    ret = ret.to(x.dtype, bitcast=True)

    # argsort
    y_indices = tl.reshape(x_indices, shape)
    mask = tl.arange(0, 2)[None, :, None]
    indices_left = tl.broadcast_to(tl.sum(y_indices * (1 - mask), 1)[:, None, :], shape).to(y_indices.dtype)
    indices_right = tl.broadcast_to(tl.sum(y_indices * mask, 1)[:, None, :], shape).to(y_indices.dtype)
    indices_left = tl.reshape(indices_left, x_indices.shape)
    indices_right = tl.reshape(indices_right, x_indices.shape)
    ret_indices = x_indices ^ tl.where(cond, indices_left ^ indices_right, tl.zeros_like(x_indices))
    return ret, ret_indices


@triton.jit
def bitonic_merge_wrapper(x,
            x_indices,
            a_shape_1: tl.constexpr):
    """
    li ruixiao from double sparse argsort
    Merge a 2-dim tensor along the last dimension, and return argsort tesnor.

    :param x: The input tensor to be sorted.
    :type x: Tensor
    :param x_shape_0: The first dimension of the input tensor, aka x.shape[0]
    :type x_shape_0: int
    :param x_shape_1: The second dimension of the input tensor, aka x.shape[1]. This is the dimension to be sorted.
    :type x_shape_1: int
    :param descending: If set to True, the tensor is sorted in descending order. If set to False, the tensor is sorted in ascending order.
    :type descending: bool, optional
    """
    tl.static_assert(a_shape_1 == 4, "a.shape[1] must be 4")
    n_dims: tl.constexpr = tl.standard._log2(a_shape_1)
    
    for i in tl.static_range(n_dims):
        x, x_indices = _compare_and_swap(x, x_indices, i, n_dims) 

    return x, x_indices
