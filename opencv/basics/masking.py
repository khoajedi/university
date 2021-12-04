import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                default="./images/jedi.jpg", help="path to input image")
args = ap.parse_args()

image = cv2.imread(args.image)
cv2.imshow("Original", image)

mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (20, 400), (250, 860), 255, -1)
masked = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow("Masked", masked)
cv2.waitKey(0)
