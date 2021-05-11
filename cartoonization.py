import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


def convert_cartoon(image):
    """
    Convert the given image by converting it into grayscale,
    apply grayscale image to median blur, then adaptiveThreshold with
    bilateralFilter for the cartoonization of the image.
    """
    # edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 7, 3)

    # cartoon
    color = cv2.bilateralFilter(image, 9, 250, 250)
    cartoon_image = cv2.bitwise_and(color, color, mask=edges)

    return cartoon_image


if __name__ == '__main__':
    print("\n An image to convert to a cartoonization format.\n")

    path = input("\n Provide an image path: \n")
    img = cv2.imread(path)
    img_cartoon = convert_cartoon(img)

    plt.figure(figsize=[10, 7])
    plt.subplot(121)
    plt.title("Original Image")
    plt.imshow(img[:, :,::-1])
    plt.subplot(122)
    plt.title("Cartoon Image")
    plt.imshow(img_cartoon[:, :,::-1])
    plt.show()
        
