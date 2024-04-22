import numpy as np
import scipy.signal as signal


def remove_noise(image):
    noise_power = estimate_noise_power(image)
    # Compute the power spectral density (PSD) of the image using Welch's method
    f, Pxx = signal.welch(image, nperseg=len(image))

    # Compute the filter based on the ratio of image and noise PSD
    filter = np.sqrt(Pxx / (Pxx + noise_power))

    # Apply the filter to the image in the frequency domain
    filtered_image = np.fft.ifft2(np.fft.fft2(image) * filter)

    # Ensure the filtered image is real-valued
    filtered_image = np.real(filtered_image)

    return filtered_image


def wiener_filter_dft(image, noise_power, signal_power):
    """
    Denoise an image using Wiener filtering in the frequency domain with the Discrete Fourier Transform (DFT).

    Parameters:
        image (numpy.ndarray): Input image (grayscale).
        noise_power (float): Power of the additive noise.
        signal_power (float): Power of the original signal (before adding noise).

    Returns:
        numpy.ndarray: Denoised image.
    """
    # Compute the power spectral density (PSD) of the noisy image
    noisy_image_fft = np.fft.fft2(image)
    noisy_image_psd = np.abs(noisy_image_fft) ** 2

    # Compute the Wiener filter
    filter = signal_power / (signal_power + noise_power)

    # Apply the Wiener filter in the frequency domain
    denoised_image_fft = noisy_image_fft * filter

    # Reconstruct the denoised image by applying the inverse DFT
    denoised_image = np.fft.ifft2(denoised_image_fft)

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
