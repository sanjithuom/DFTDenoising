import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn.decomposition import MiniBatchDictionaryLearning
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float, img_as_ubyte


def wavelet_sparse_nlm_denoising(image):
    # Convert image to float for processing
    image = img_as_float(image)

    # Estimate noise sigma from the original image
    sigma_est = np.mean(estimate_sigma(image))  # Assume correct version of estimate_sigma()

    # Set threshold relative to estimated noise
    threshold = sigma_est * 0.5

    # Step 1: Wavelet Decomposition
    coeffs = pywt.wavedec2(image, 'db1', level=2)
    coeffs_thresh = [(cH.copy(), cV.copy(), cD.copy()) for cH, cV, cD in coeffs[1:]]

    # Step 2: Sparse Coding of Wavelet Coefficients
    dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, max_iter=500)
    for i, (cH, cV, cD) in enumerate(coeffs_thresh):
        for j, coeff in enumerate([cH, cV, cD]):
            X = coeff.reshape(-1, 1)
            dico.fit(X)
            code = dico.transform(X)
            # Apply threshold based on code magnitude without reshaping
            mask = (np.abs(code) > threshold).flatten()
            coeffs_thresh[i][j] = coeff.flatten() * mask
            coeffs_thresh[i][j] = coeffs_thresh[i][j].reshape(coeff.shape)

    # Replace detailed coefficients with thresholded ones
    coeffs[1:] = coeffs_thresh

    # Step 3: Reconstruction from Wavelet Coefficients
    reconstructed_image = pywt.waverec2(coeffs, 'db1')

    # Step 4: Non-Local Means Denoising
    sigma_est = np.mean(estimate_sigma(reconstructed_image))
    denoised_image = denoise_nl_means(reconstructed_image, h=1.15 * sigma_est, fast_mode=True,
                                      patch_size=5, patch_distance=6)

    # Clip values to valid range and convert to 8-bit format
    denoised_image = np.clip(denoised_image, 0, 1)
    denoised_image = img_as_ubyte(denoised_image)

    return denoised_image


def pad_image_to_power_of_two(image):
    """Pad an image up to the nearest dimensions that are powers of two."""
    m, n = image.shape
    M, N = 2**np.ceil(np.log2(m)).astype(int), 2**np.ceil(np.log2(n)).astype(int)
    padded_image = np.zeros((M, N), dtype=image.dtype)
    padded_image[:m, :n] = image
    return padded_image, (m, n)

def crop_image_to_original(image, original_dim):
    """Crop the padded image back to the original dimensions."""
    m, n = original_dim
    return image[:m, :n]

def soft_threshold(coeff, value):
    magnitude = np.abs(coeff)
    with np.errstate(divide='ignore', invalid='ignore'):  # Ignore warnings in this context
        thresholded = np.where(magnitude > value, (1 - value/magnitude) * coeff, 0)
    return thresholded

def w_denoise_image(image, wavelet='db3', level=2, mode='hard'):
    np.nan_to_num(image, copy=False)  # Handle NaN and Inf
    original_dim = image.shape
    image, _ = pad_image_to_power_of_two(image)  # Assume padding function is defined elsewhere

    coeffs = pywt.wavedec2(image, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1][-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(image.size)) * 0.8

    new_coeffs = [coeffs[0]]
    if mode == 'soft':
        for detail_level in coeffs[1:]:
            new_detail_level = tuple(soft_threshold(coeff, threshold) for coeff in detail_level)
            new_coeffs.append(new_detail_level)
    else:
        for detail_level in coeffs[1:]:
            new_detail_level = tuple(pywt.threshold(coeff, value=threshold, mode=mode) for coeff in detail_level)
            new_coeffs.append(new_detail_level)

    denoised_image = pywt.waverec2(new_coeffs, wavelet)
    denoised_image = crop_image_to_original(denoised_image,
                                            original_dim)  # Assume cropping function is defined elsewhere

    return denoised_image