import numpy as np

# evaluate image using quality measure peak signal-to-noise ratio(psnr)
def psnr(original_image, denoised_image):
    original_image = original_image.astype(np.float64)
    denoised_image = denoised_image.astype(np.float64)

    mse = np.mean((original_image - denoised_image) ** 2)

    max_pixel_value = 255.0

    if mse == 0:
        return float('inf')  # PSNR is infinity if images are identical

    psnr_value = 10 * np.log10(max_pixel_value ** 2 / mse)

    return psnr_value
