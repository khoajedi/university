import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                default="./images/jedi.jpg", help="path to input image")
args = ap.parse_args()

image = cv2.imread(args.image)
cv2.imshow("Original", image)
(h, w) = image.shape[:2]

r = 150.0 / w
dim = (150, int(h * r))

resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resize width", resized)

r = 50.0 / h
dim = (int(w * r), 50.0)

resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resize height", resized)

resized = imutils.resize(image, width=100)
cv2.imshow("Resized via imutils", resized)

cv2.waitKey(0)
