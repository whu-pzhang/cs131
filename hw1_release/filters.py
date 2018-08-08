import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # YOUR CODE HERE
    for i in range(Hk // 2, Hi - Hk // 2):
        for j in range(Wk // 2, Wi - Wk // 2):
            for k in range(Hk):
                for l in range(Wk):
                    out[i, j] += kernel[k, l] * \
                        image[i + Hk // 2 - k, j + Wk // 2 - l]
    # END YOUR CODE

    return out


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    # YOUR CODE HERE
    out = np.pad(image, ((pad_height, pad_height), (pad_width,
                                                    pad_width)), "constant", constant_values=(0))
    # END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # YOUR CODE HERE
    # 若要和conv_nested的结果做比较的话，这里不应该使用zero_pad函数！
    # 因为conv_nested中并没有pad操作
    # image = zero_pad(image, Hk // 2, Wk // 2)  # padding image
    kernel = np.flip(np.flip(kernel, axis=0), axis=1)  # flip kernel
    half_Hk = Hk // 2
    half_Wk = Wk // 2
    for i in range(half_Hk, Hi - half_Hk):
        for j in range(half_Wk, Wi - half_Wk):
            out[i, j] = np.sum(
                image[i - half_Hk:i + half_Hk + 1, j - half_Wk:j + half_Wk + 1] * kernel)

    # END YOUR CODE

    return out


def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # YOUR CODE HERE
    kernel = np.flip(np.flip(kernel, axis=0), axis=1)  # flip kernel
    half_Hk = Hk // 2
    half_Wk = Wk // 2
    # 将相乘相加操作变为向量乘法
    for i in range(half_Hk, Hi - half_Hk):
        for j in range(half_Wk, Wi - half_Wk):
            out[i, j] = np.ravel(image[i - half_Hk:i + half_Hk + 1, j -
                                       half_Wk:j + half_Wk + 1]) @ np.ravel(kernel)
    # END YOUR CODE

    return out


def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    # YOUR CODE HERE
    g = np.flip(np.flip(g, axis=0), axis=1)
    # change dimension of g to be an odd number
    out = conv_fast(f, g[:-1, :])
    # END YOUR CODE

    return out


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    # YOUR CODE HERE
    g = g - np.mean(g)
    out = cross_correlation(f, g)
    # END YOUR CODE

    return out


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    # YOUR CODE HERE
    g = g[:-1, :]  # Make the dimension of g is odd number

    g_normalized = (g - np.mean(g)) / np.std(g)

    Hi, Wi = f.shape
    Hk, Wk = g.shape
    half_Hk = Hk // 2
    half_Wk = Wk // 2
    out = np.zeros((Hi, Wi))

    for i in range(half_Hk, Hi - half_Hk):
        for j in range(half_Wk, Wi - half_Wk):
            f_patch = f[i - half_Hk:i + half_Hk +
                        1, j - half_Wk:j + half_Wk + 1]
            out[i, j] = np.sum(
                (f_patch - np.mean(f_patch)) / np.std(f_patch) * g_normalized)
    # END YOUR CODE

    return out
