import cv2
import numpy as np


def denoise_by_cv2(image, radius=20):
    """
       Denoises the input image using frequency domain filtering with OpenCV using circular mask.

       Parameters:
           image (numpy.ndarray): The input image.
           radius: Radius of circular mask

       Returns:
           numpy.ndarray: The denoised image.
       """
    # Compute the DFT for image
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

    # Shift the zero frequency component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)

    rows, cols = image.shape

    # Calculate the center coordinates
    crow, ccol = rows // 2, cols // 2

    # Create a circular mask
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)

    # Apply the mask to the frequency domain representation
    dft_shift[:, :, 0] *= mask
    dft_shift[:, :, 1] *= mask

    # Shift the fft back to the original
    dft_ishift = np.fft.ifftshift(dft_shift)

    # Compute the inverse DFT to get the denoised image
    denoised_image = cv2.idft(dft_ishift)

    # Compute the magnitude spectrum (magnitude of the complex numbers)
    magnitude_spectrum = cv2.magnitude(denoised_image[:, :, 0], denoised_image[:, :, 1])

    # Normalize the magnitude spectrum to the range [0, 255]
    denoised_image = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return denoised_image
