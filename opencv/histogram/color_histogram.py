from matplotlib import pyplot as plt
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                default="./images/jedi.jpg", help="path to input image")
args = ap.parse_args()

image = cv2.imread(args.image)

chans = cv2.split(image)
colors = ("b", "g", "r")

plt.figure()
plt.title("Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.figure()
plt.axis("off")
plt.imshow(imutils.opencv2matplotlib(image))

plt.show()