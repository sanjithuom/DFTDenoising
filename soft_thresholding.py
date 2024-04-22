import numpy as np
import cv2


def soft_threshold(x, threshold):
    """
    Apply soft thresholding to the input array 'x' with the specified threshold.

    Parameters:
        x (numpy.ndarray): Input array to be thresholded.
        threshold (float): Threshold value for soft thresholding.

    Returns:
        numpy.ndarray: Thresholded array.
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def soft_thresholding_dft(image, threshold):
    """
    Denoise an image using soft thresholding in the frequency domain with the Discrete Fourier Transform (DFT).

    Parameters:
        image (numpy.ndarray): Input image (grayscale).
        threshold (float): Threshold value for soft thresholding.

    Returns:
        numpy.ndarray: Denoised image.
    """
    # Apply the Discrete Fourier Transform (DFT) to the image
    dft_image = np.fft.fft2(image)

    # Apply soft thresholding in the frequency domain
    dft_image_thresholded = soft_threshold(dft_image, threshold)

    # Reconstruct the denoised image by applying the inverse DFT
    denoised_image = np.fft.ifft2(dft_image_thresholded)

    # Ensure the denoised image is real-valued and clip values to valid intensity range [0, 255]
    denoised_image = np.real(denoised_image)
    denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)

    return denoised_image