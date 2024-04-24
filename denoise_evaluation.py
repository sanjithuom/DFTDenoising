import cv2
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import random_noise, img_as_ubyte, img_as_float

from nlm_filter_numba import fast_NLmeansfilter
from denoise_cv2 import denoise_by_cv2
from denoise_np import denoise_by_np, post_process_image
from evaluation import psnr
from mle_approach import ml_denoising_dft
from psd_noise_filtering import wiener_filter_dft, estimate_noise_power, estimate_signal_power
from soft_thresholding import soft_thresholding_dft

denois_methods = [denoise_by_cv2, denoise_by_np, wiener_filter_dft, soft_thresholding_dft, ml_denoising_dft,
                  denoise_nl_means, fast_NLmeansfilter]
estimate_signal_power, estimate_noise_power

original_image = cv2.imread('images/t1_original.png', cv2.IMREAD_GRAYSCALE)
# noisy_image = cv2.imread('images/t1_with_noise_0.75.png', cv2.IMREAD_GRAYSCALE)

patch_kw = dict(
    patch_size=5, patch_distance=6
)


def show_fft(image, title):
    fftshift = np.fft.fftshift(np.fft.fft2(image))
    cv2.imshow(title, post_process_image(np.abs(fftshift)))


# convert image array to float
original_image_as_float = img_as_float(original_image)

# add noise to image
sigma = 0.1
noisy_image_as_float = random_noise(original_image, var=sigma ** 2)
noisy_image = img_as_ubyte(noisy_image_as_float)

# psnr value for noisy image
psnr_noisy_value = psnr(original_image=original_image, denoised_image=noisy_image)

# parameters for different algorithms to be used
sigma_est = np.mean(estimate_sigma(noisy_image_as_float))
signal_power = np.mean(estimate_sigma(original_image_as_float))

# show frequency domain of the image
show_fft(original_image_as_float, 'Original Image FFT')
show_fft(noisy_image_as_float, 'Noisy Image FFT')

# state of the art function NLM filter
state_of_the_art_function = lambda image: denoise_nl_means(
    noisy_image_as_float, h=0.6 * sigma_est, sigma=sigma_est, fast_mode=False, **patch_kw
)

# denoise_function = lambda image: denoise_by_cv2(image)

denoise_function = lambda image: denoise_by_np(image)

# denoise_function = lambda image: ml_denoising_dft(image, patch_size=4, window_size=4, sigma=sigma_est * (10 ** 5))
# denoise_function = lambda image: wiener_filter_dft(image, estimate_noise_power(image),
#                                                    estimate_signal_power(original_image_as_float))
# denoise_function = lambda image: soft_thresholding_dft(image, threshold=0.6 * sigma_est)

# denoise_function = lambda image: fast_NLmeansfilter(image, t=5, f=7, h=0.8 * sigma_est)

# denoise_function = lambda image: nlm_filter_dft.nlm_denoise(image, t=5, f=7, h=0.6 * sigma_est)
# denoise_function = lambda image: nlm_denoise.nlm_denoise(image, t=5, f=7, h=0.6 * sigma_est)

# denoised_image = denoise_function_additional(denoise_function(noisy_image_as_float))
denoised_image = denoise_function(noisy_image_as_float)
# denoised_image = denoise_function(noisy_image_as_float)
denoised_image_sota = state_of_the_art_function(noisy_image_as_float)


show_fft(denoised_image, 'Denoised Image FFT')
show_fft(denoised_image_sota, 'State of the art denoised FFT')

psnr_denoised_value = psnr(original_image=original_image_as_float, denoised_image=denoised_image)
psnr_denoised_value_sota = psnr(original_image=original_image_as_float, denoised_image=denoised_image_sota)

print(f'psnr noisy value: {psnr_noisy_value}')
print(f'psnr denoised value: {psnr_denoised_value}')
print(f'psnr denoised value state of the art: {psnr_denoised_value_sota}')
cv2.imshow('Original Image', original_image)
cv2.imshow('Noisy Image', noisy_image)
cv2.imshow('Denoised Image', denoised_image)
cv2.imshow('Denoised Image State of the Art', denoised_image_sota)
cv2.waitKey(0)
cv2.destroyAllWindows()
