import cv2
import numpy as np


def denoise_by_cv2(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

    dft_shift = np.fft.fftshift(dft)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    r = 50  # adjust the radius of the circular mask
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), r, 1, -1)

    dft_shift[:, :, 0] *= mask
    dft_shift[:, :, 1] *= mask

    dft_ishift = np.fft.ifftshift(dft_shift)
    denoised_image = cv2.idft(dft_ishift)
    magnitude_spectrum = cv2.magnitude(denoised_image[:, :, 0], denoised_image[:, :, 1])
    denoised_image = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return denoised_image
