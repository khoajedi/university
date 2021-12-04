import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                default="./images/jedi.jpg", help="path to input image")
args = ap.parse_args()

image = cv2.imread(args.image)
cv2.imshow("Original", image)

kSizes = [(3, 3), (9, 9), (15, 15)]

for (kX, kY) in kSizes:
    blurred = cv2.blur(image, (kX, kY))
    cv2.imshow("Average ({}, {})".format(kX, kY), blurred)

cv2.waitKey(0)

cv2.destroyAllWindows()
cv2.imshow("Original", image)

for (kX, kY) in kSizes:
    blurred = cv2.GaussianBlur(image, (kX, kY), 0)
    cv2.imshow("Gaussian ({}, {})".format(kX, kY), blurred)

cv2.waitKey(0)

cv2.destroyAllWindows()
cv2.imshow("Original", image)

for k in (3, 9, 15):
    blurred = cv2.medianBlur(image, k)
    cv2.imshow("Median {}".format(k), blurred)

cv2.waitKey(0)