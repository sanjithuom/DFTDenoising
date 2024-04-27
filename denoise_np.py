import numpy as np


def post_process_image(image):
    """Post processing to display DFTs
    Parameters:
        image (numpy.ndarray): Input image (grayscale).

    Returns:
        numpy.ndarray: post processed image.
    """
    rows, cols = image.shape
    # Log compressed image
    log_compressed_image = np.zeros((rows, cols), dtype=np.uint8)
    compression_factor = 10

    # do log compression for each pixel
    for i in range(rows):
        for j in range(cols):
            log_compressed_image[i][j] = compression_factor * np.log(1 + image[i][j])

    stretched_image = log_compressed_image

    # find min max value
    min_value = np.min(stretched_image)
    max_value = np.max(stretched_image)

    # Stretch the image to [0,255]
    stretched_image = ((stretched_image - min_value) / (max_value - min_value)) * 255
    stretched_image = np.clip(stretched_image, 0, 255)

    return stretched_image.astype(np.uint8)


def stretch_image(image):
    stretched_image = image

    # find min max value
    min_value = np.min(stretched_image)
    max_value = np.max(stretched_image)

    # Stretch the image to [0,255]
    stretched_image = ((stretched_image - min_value) / (max_value - min_value)) * 255.0
    stretched_image = np.clip(stretched_image, 0, 255)

    return stretched_image


def denoise_by_np(image, sigma=50, radius=10, alpha=0.05):
    """
       Denoises the input image using frequency domain filtering with NumPy using circular mask.

       Parameters:
           image (numpy.ndarray): The input image.
           sigma: Adjust sigma to control the smoothness of the decline of mask. Default is 50
           radius: Radius near the zero frequency to emphasize. Default is 10.
           alpha: Factor by which region near zero freq should be magnified. Alpha is 0.05.
       Returns:
           numpy.ndarray: The denoised image.
       """
    # Compute the dft using fft
    dft = np.fft.fft2(image)

    # Shift the zero frequency component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)

    # Get the dimensions of the image and the center coordinates
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Calculate the radius of the circular mask
    r = min(crow, ccol)

    # Create a mask to filter out high-frequency noise using a Gaussian function
    mask = np.zeros((rows, cols), dtype=np.float32)
    y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
    # Define circular region within the mask
    mask_area = x ** 2 + y ** 2 <= r ** 2
    mask_value = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))  # Gaussian function
    mask[mask_area] = mask_value[mask_area]
    # non - circular region
    mask[~mask_area] = 0.05

    # add more importance to closer to zero frequency
    zero_freq_neighborhood = x ** 2 + y ** 2 <= radius ** 2
    mask[zero_freq_neighborhood] *= (1 + alpha)

    # Apply the mask to the shifted FFT
    dft_shift[:, :] *= mask

    # dft_shift = stretch_image(dft_shift)
    # Inverse shift to bring zero frequency components back to the corners
    dft_ishift = np.fft.ifftshift(dft_shift)

    # Compute the inverse FFT to get the denoised image in the spatial domain
    denoised_image = np.fft.ifft2(dft_ishift)

    # Calculate the magnitude of the complex numbers to get the final denoised image
    denoised_image = np.abs(denoised_image)

    return denoised_image
