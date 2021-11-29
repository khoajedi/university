import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input image")
ap.add_argument("-o", "--output", required=True, help="path to output image")
args = ap.parse_args()

image = cv2.imread(args.input)
(h, w, c) = image.shape[:3]

print("width: {} pixels".format(w))
print("height: {} pixels".format(h))
print("channel: {} channels".format(c))

cv2.imshow("Image", image)
cv2.waitKey(0)

cv2.imwrite(args.output, image)