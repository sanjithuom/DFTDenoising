import cv2
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import random_noise, img_as_ubyte, img_as_float

from denoise_cv2 import denoise_by_cv2
from denoise_np import denoise_by_np, post_process_image
from evaluation import psnr
from mle_approach import ml_denoising_dft
from nlm_filter_numba import fast_NLmeansfilter
from psd_noise_filtering import wiener_filter_dft, estimate_noise_power, estimate_signal_power
from soft_thresholding import soft_thresholding_dft

denois_methods = [denoise_by_cv2, denoise_by_np, wiener_filter_dft, soft_thresholding_dft, ml_denoising_dft,
                  denoise_nl_means, fast_NLmeansfilter]
estimate_signal_power, estimate_noise_power

image_id = '53'

original_image = cv2.imread(f'images/t1_original_{image_id}.png', cv2.IMREAD_GRAYSCALE)

patch_kw = dict(
    patch_size=5, patch_distance=6
)


def show_dft(image, title):
    """Show the DFT image
    Parameters:
        image (numpy.ndarray): Input image (grayscale).
        title (str): Title for the image window
    """
    # Compute the dft and shift the zero frequency component to the center of the spectrum
    fftshift = np.fft.fftshift(np.fft.fft2(image))

    processed_dft = post_process_image(np.abs(fftshift))
    # show the image
    cv2.imshow(title, processed_dft)

    # save images
    image_file = title.lower().replace('image ', '').replace(' ', '_').replace('(', '').replace(')', '')
    cv2.imwrite(f"images/{image_id}/t1_{image_file}_{image_id}.png", processed_dft)


def show_and_save_denoised_image(image, title, filename):
    """Show and save the denoised image
    Parameters:
        image (numpy.ndarray): Input image (grayscale).
        title (str): Title for the image window
        filename (str): Filename to save the image
    """
    scaled_image = img_as_ubyte(np.clip(image, -1, 1))
    cv2.imshow(title, scaled_image)
    cv2.imwrite(filename, scaled_image)


# convert image array to float
original_image_as_float = img_as_float(original_image)

# add noise to image
sigma = 0.1
noisy_image_as_float = random_noise(original_image, var=sigma ** 2)
noisy_image = img_as_ubyte(noisy_image_as_float)

# calculate psnr value for noisy image
psnr_noisy_value = psnr(original_image=original_image, denoised_image=noisy_image)

# parameters for different algorithms to be used
sigma_est = np.mean(estimate_sigma(noisy_image_as_float))
signal_power = np.mean(estimate_sigma(original_image_as_float))

# state of the art function NLM filter
state_of_the_art_function = lambda image: denoise_nl_means(
    noisy_image_as_float, h=0.6 * sigma_est, sigma=sigma_est, fast_mode=False, **patch_kw
)

# denoise functions
denoise_function = lambda image: denoise_by_np(image, sigma=65)
denoise_function_nlm_custom = lambda image: fast_NLmeansfilter(image, t=5, f=7, h=0.8 * sigma_est)

# call denoise function to get the denoised image
denoised_image = denoise_function(noisy_image_as_float)
denoised_image_nlm_custom = denoise_function_nlm_custom(noisy_image_as_float)
# call denoise function using state of the art method to get the denoised image
denoised_image_sota = state_of_the_art_function(noisy_image_as_float)

# calculate the psnr value of noisy image, denoised image, denoised image using state of the art method
psnr_denoised_value = psnr(original_image=original_image_as_float, denoised_image=denoised_image)
psnr_denoised_value_nlm_custom = psnr(original_image=original_image_as_float, denoised_image=denoised_image_nlm_custom)
psnr_denoised_value_sota = psnr(original_image=original_image_as_float, denoised_image=denoised_image_sota)

print(f'================== Image id: {image_id} ===========================')
print(f'psnr noisy value: {psnr_noisy_value}')
print(f'psnr denoised value (DFT Masking): {psnr_denoised_value}')
print(f'psnr denoised value (NLM Custom): {psnr_denoised_value_nlm_custom}')
print(f'psnr denoised value (State of the art): {psnr_denoised_value_sota}')
print(f'===========================================================')
# show all the images
cv2.imshow('Original Image', original_image)
cv2.imshow('Noisy Image', noisy_image)
# cv2.imshow('Denoised Image (DFT Masking)', denoised_image)
# cv2.imshow('Denoised Image (NLM Custom)', denoised_image_nlm_custom)
# cv2.imshow('Denoised Image (State of the Art)', denoised_image_sota)

cv2.imwrite(f'images/{image_id}/t1_noisy_{image_id}.png', noisy_image)

show_and_save_denoised_image(denoised_image, title='Denoised Image (DFT Masking)',
                             filename=f'images/{image_id}/t1_denoised_dft_masking_{image_id}.png')
show_and_save_denoised_image(denoised_image_nlm_custom, title='Denoised Image (NLM Custom)',
                             filename=f'images/{image_id}/t1_denoised_nlm_custom_{image_id}.png')
show_and_save_denoised_image(denoised_image_sota, title='Denoised Image (State of the Art)',
                             filename=f'images/{image_id}/t1_denoised_state_of_the_art_{image_id}.png')

# show frequency domain of all the images
show_dft(original_image_as_float, 'Original Image DFT')
show_dft(noisy_image_as_float, 'Noisy Image DFT')
show_dft(denoised_image, 'Denoised Image DFT (DFT Masking)')
show_dft(denoised_image_nlm_custom, 'Denoised Image DFT (NLM Custom)')
show_dft(denoised_image_sota, 'Denoised Image DFT (State of the Art)')

cv2.waitKey(0)
cv2.destroyAllWindows()
