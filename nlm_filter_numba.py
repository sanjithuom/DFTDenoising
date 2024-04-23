from numba import jit
import numpy as np


@jit(nopython=True)
def make_kernel(f):
    kernel = np.zeros((2 * f + 1, 2 * f + 1))
    for d in range(1, f + 1):
        value = 1 / (2 * d + 1) ** 2
        for i in range(-d, d + 1):
            for j in range(-d, d + 1):
                kernel[f - i, f - j] += value
    kernel /= f
    return kernel


@jit(nopython=True)
def pad_image(input_image, pad_width):
    m, n = input_image.shape
    padded_image = np.zeros((m + 2 * pad_width, n + 2 * pad_width))
    # Copy original image into center
    padded_image[pad_width:-pad_width, pad_width:-pad_width] = input_image
    # Top border
    padded_image[:pad_width, pad_width:-pad_width] = input_image[:1, :]
    # Bottom border
    padded_image[-pad_width:, pad_width:-pad_width] = input_image[-1:, :]
    # Left border
    padded_image[:, :pad_width] = padded_image[:, pad_width:pad_width + 1]
    # Right border
    padded_image[:, -pad_width:] = padded_image[:, -pad_width - 1:-pad_width]
    return padded_image


@jit(nopython=True)
def fast_NLmeansfilter(input_image, t=3, f=5, h=10):
    m, n = input_image.shape
    output = np.zeros((m, n))
    padded_input = pad_image(input_image, f)
    kernel = make_kernel(f)
    h = h * h

    for i in range(m):
        for j in range(n):
            i1, j1 = i + f, j + f
            W1 = padded_input[i1 - f:i1 + f + 1, j1 - f:j1 + f + 1]
            wmax = 0
            average = 0
            sweight = 0

            rmin = max(i1 - t, f)
            rmax = min(i1 + t, m + f - 1)
            smin = max(j1 - t, f)
            smax = min(j1 + t, n + f - 1)

            for r in range(rmin, rmax + 1):
                for s in range(smin, smax + 1):
                    if r == i1 and s == j1:
                        continue
                    W2 = padded_input[r - f:r + f + 1, s - f:s + f + 1]
                    if W2.shape != (2 * f + 1, 2 * f + 1):
                        continue
                    d = np.sum(kernel * (W1 - W2) ** 2)
                    w = np.exp(-d / h)

                    if w > wmax:
                        wmax = w
                    sweight += w
                    average += w * padded_input[r, s]

            average += wmax * padded_input[i1, j1]
            sweight += wmax

            if sweight > 0:
                output[i, j] = average / sweight
            else:
                output[i, j] = input_image[i, j]
    return output
