from skimage.exposure import is_low_contrast
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                default="./images/jedi.jpg", help="path to input image")
ap.add_argument("-t", "--thresh", type=float, default=0.35,
                help="threshold for low contrast")
args = ap.parse_args()

image = cv2.imread(args.image)
image = imutils.resize(image, width=450)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (7, 7), 0)

text = "Low contrast: No"
color = (0, 255, 0)

if is_low_contrast(gray, fraction_threshold=args.thresh):
    text = "Low contrast: Yes"
    color = (0, 0, 255)

cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

cv2.imshow("Image", image)
cv2.waitKey(0)