import numpy as np
import argparse
import cv2

def auto_canny(image, sigma=0.33):
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    edged = cv2.Canny(image, lower, upper)

    return edged

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                default="./images/jedi.jpg", help="path to input image")
args = ap.parse_args()

image = cv2.imread(args.image)
cv2.imshow("Original", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

wide = cv2.Canny(blurred, 10, 200)
tight = cv2.Canny(blurred, 255, 250)
auto = auto_canny(blurred)

cv2.imshow("Edges", np.hstack([wide, tight, auto]))
cv2.waitKey(0)