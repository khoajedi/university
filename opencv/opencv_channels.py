import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                default="./images/jedi.jpg", help="path to input image")
args = ap.parse_args()

image = cv2.imread(args.image)
(B, G, R) = cv2.split(image)

merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)
cv2.waitKey(0)