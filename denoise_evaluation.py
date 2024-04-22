import cv2

from denoise_cv2 import denoise_by_cv2
from evaluation import psnr
from psd_noise_filtering import remove_noise, wiener_filter_dft, estimate_noise_power, estimate_signal_power
from soft_thresholding import soft_thresholding_dft

denois_methods = [denoise_by_cv2, remove_noise, wiener_filter_dft, soft_thresholding_dft]
estimate_signal_power, estimate_noise_power

original_image = cv2.imread('images/t1_original.png', cv2.IMREAD_GRAYSCALE)
noisy_image = cv2.imread('images/t1_with_noise_0.75.png', cv2.IMREAD_GRAYSCALE)

soft_thresholding_threshold = 1927
psnr_noisy_value = psnr(original_image=original_image, denoised_image=noisy_image)

denoise_function = lambda image: remove_noise(image)

denoised_image = denoise_function(noisy_image)
psnr_denoised_value = psnr(original_image=original_image, denoised_image=denoised_image)

print(f'psnr noisy value: {psnr_noisy_value}')
print(f'psnr denoised value: {psnr_denoised_value}')
cv2.imshow('Original Image', original_image)
cv2.imshow('Noisy Image', noisy_image)
cv2.imshow('Denoised Image', denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
