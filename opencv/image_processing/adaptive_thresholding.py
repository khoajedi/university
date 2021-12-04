import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                default="./images/coins.png", help="path to input image")
args = ap.parse_args()

image = cv2.imread(args.image)
cv2.imshow("Input", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

thresh = cv2.adaptiveThreshold(blurred, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
cv2.imshow("Mean Adaptive Thresholding", thresh)

masked = cv2.bitwise_and(image, image, mask=thresh)
cv2.imshow("Mean Ouptut", masked)

cv2.waitKey(0)

thresh = cv2.adaptiveThreshold(blurred, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

cv2.imshow("Gaussian Adaptive Thresholding", thresh)

masked = cv2.bitwise_and(image, image, mask=thresh)
cv2.imshow("Gaussian Ouptut", masked)

cv2.waitKey(0)
