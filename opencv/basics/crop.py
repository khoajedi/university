import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                default="./images/jedi.jpg", help="path to input image")
args = ap.parse_args()

image = cv2.imread(args.image)
cv2.imshow("Original", image)

face = image[20:160, 300:420]
cv2.imshow("Face", face)

lightsaber = image[400:860, 20:250]
cv2.imshow("Lightsaber", lightsaber)

cv2.waitKey(0)