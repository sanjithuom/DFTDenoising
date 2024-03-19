import numpy as np


def post_process_image(image, log_compression=False):
    stretched_image = image
    if log_compression:
        rows, cols = image.shape
        log_compressed_image = np.zeros((rows, cols), dtype=np.uint8)
        compression_factor = 10

        for i in range(rows):
            for j in range(cols):
                log_compressed_image[i][j] = compression_factor * np.log(1 + image[i][j])

        stretched_image = log_compressed_image

    min_value = np.min(stretched_image)
    max_value = np.max(stretched_image)
    stretched_image = ((stretched_image - min_value) / (max_value - min_value)) * 255
    stretched_image = np.clip(stretched_image, 0, 255)

    return stretched_image.astype(np.uint8)


def denoise(image):
    fft = np.fft.fft2(image)
    shifted_fft = np.fft.fftshift(fft)

    # rows, cols = image.shape
    # user_def_mask = get_mask((rows, cols))
    # filter_shift_fft = shifted_fft * user_def_mask
    shifted_inverse_fft = np.fft.ifftshift(shifted_fft)
    inverse_fft = np.fft.ifft2(shifted_inverse_fft)
    filtered_image = post_process_image(np.abs(inverse_fft))

    return filtered_image
