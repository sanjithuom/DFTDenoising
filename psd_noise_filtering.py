import numpy as np


def wiener_filter_dft(noisy_image, noise_power, signal_power):
    """
    Denoise an image using Wiener filtering in the frequency domain with the Discrete Fourier Transform (DFT).

    Parameters:
        noisy_image (numpy.ndarray): Noisy input image (grayscale).
        noise_power (float): Power of the additive noise.
        signal_power (float): Power of the original signal (before adding noise).

    Returns:
        numpy.ndarray: Denoised image.
    """
    # Compute the DFT of the noisy image
    dft_noisy_image = np.fft.fft2(noisy_image)

    # Compute the power spectral density (PSD) of the noisy image
    noisy_image_psd = np.abs(dft_noisy_image) ** 2

    # Compute the filter based on the Wiener filter equation
    filter = signal_power / (signal_power + noise_power)

    # Normalize the filter to preserve brightness
    filter /= np.sum(filter)

    # Apply a minimum threshold to the filter
    filter = np.maximum(filter, 0.1)

    # Apply the filter to the DFT of the noisy image
    dft_denoised_image = dft_noisy_image * filter

    # Reconstruct the denoised image by applying the inverse DFT
    denoised_image = np.fft.ifft2(dft_denoised_image)

    # Ensure the denoised image is real-valued and clip values to valid intensity range [0, 255]
    denoised_image = np.real(denoised_image)
    denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)

    return denoised_image



def estimate_noise_power(noisy_image):
    noise_power = np.var(noisy_image)
    return noise_power


def estimate_signal_power(clean_image):
    signal_power = np.var(clean_image)
    return signal_power
