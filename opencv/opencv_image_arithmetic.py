import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                default="./images/jedi.jpg", help="path to input image")
args = ap.parse_args()

max = cv2.add(np.uint8([200]), np.uint8([100]))
min = cv2.subtract(np.uint8([50]), np.uint8([100]))
print("opencv max: {}".format(max))
print("opencv min: {}".format(min))

max = np.add(np.uint8([200]), np.uint8([100]))
min = np.subtract(np.uint8([50]), np.uint8([100]))
print("numpy max: {}".format(max))
print("numpy min: {}".format(min))

image = cv2.imread(args.image)
cv2.imshow("Original", image)

M = np.ones(image.shape, dtype="uint8") * 50

lighter = cv2.add(image, M)
cv2.imshow("Lighter", lighter)

darker = cv2.subtract(image, M)
cv2.imshow("Darker", darker)

cv2.waitKey(0)