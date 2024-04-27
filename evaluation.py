import numpy as np


def psnr(original_image, denoised_image):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        original_image (numpy.ndarray): The original image (must be in the range [0, 255]).
        denoised_image (numpy.ndarray): The denoised image (must be in the range [0, 255]).

    Returns:
        float: The PSNR value.
    """
    # Convert images to float64 to ensure accurate calculations
    original_image = original_image.astype(np.float64)
    denoised_image = denoised_image.astype(np.float64)

    # Calculate Mean Squared Error (MSE) between the two images
    mse = np.mean((original_image - denoised_image) ** 2)

    # Maximum possible pixel value
    max_pixel_value = 255.0

    # If MSE is zero, images are identical, PSNR is infinity
    if mse == 0:
        return float('inf')

    # Calculate PSNR value
    psnr_value = 10 * np.log10(max_pixel_value ** 2 / mse)

    return psnr_value
