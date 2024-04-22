import cv2

from denoise_cv2 import denoise_by_cv2


def main():
    image_name = "fighter-jet"
    input_image = cv2.imread('fighter-jet.png', cv2.IMREAD_GRAYSCALE)

    output = denoise_by_cv2(input_image)

    # Write output file
    output_dir = 'output/'

    output_image_name = output_dir + image_name + "_filtered_image.jpg"
    cv2.imwrite(output_image_name, output)


if __name__ == "__main__":
    main()
