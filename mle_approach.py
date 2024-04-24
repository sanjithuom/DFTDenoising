import numpy as np


def ml_denoising_dft(noisy_image, patch_size=3, window_size=3, sigma=1000):

    # Initialize denoised image
    denoised_image = np.zeros_like(noisy_image)

    # Define boundaries for patches
    h, w = noisy_image.shape
    h_limit = h - patch_size + 1
    w_limit = w - patch_size + 1

    # Iterate over patches
    for i in range(0, h_limit, window_size):
        for j in range(0, w_limit, window_size):
            # Extract the patch
            patch = noisy_image[i:i + patch_size, j:j + patch_size]

            # Compute ML estimate for the patch
            ml_estimate = compute_ml_estimate(patch, sigma)

            # Fill the corresponding part in the denoised image
            denoised_image[i:i + patch_size, j:j + patch_size] = ml_estimate

    return denoised_image


def compute_ml_estimate(patch, sigma):
    # Compute ML estimate for the patch using DFT
    patch_dft = np.fft.fft2(patch)
    weights_dft = np.exp(-((np.abs(patch_dft) ** 2) / (2 * sigma ** 2)))
    ml_estimate_dft = patch_dft * weights_dft
    ml_estimate = np.fft.ifft2(ml_estimate_dft)
    return np.real(ml_estimate)
