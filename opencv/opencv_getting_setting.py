import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
                default="./images/jedi.jpg", help="path to input image")
args = ap.parse_args()

image = cv2.imread(args.input)
(h, w) = image.shape[:2]
cv2.imshow("Original", image)

(b, g, r) = image[0, 0]
print("Pixel at (0, 0) - R: {}, G: {}, B: {}".format(r, g, b))

(b, g, r) = image[20, 50]
print("Pixel at (50, 20) - R: {}, G: {}, B: {}".format(r, g, b))

image[20, 50] = (0, 0, 255)
(b, g, r) = image[20, 50]
print("Pixel at (50, 20) - R: {}, G: {}, B: {}".format(r, g, b))

(cX, cY) = (w // 2, h // 2)

tl = image[:cY, :cX]
cv2.imshow("Top-Left Image", tl)

tr = image[:cY, cX:]
cv2.imshow("Top-Right Image", tr)

bl = image[cY:, :cX]
cv2.imshow("Bottom-Left Image", bl)

br = image[cY:, cX:]
cv2.imshow("Bottom-Right Image", br)

image[:cY, :cX] = (0, 255, 0)
cv2.imshow("Updated", image)

cv2.waitKey(0)