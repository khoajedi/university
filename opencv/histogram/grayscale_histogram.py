from matplotlib import pyplot as plt
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                default="./images/jedi.jpg", help="path to input image")
args = ap.parse_args()

image = cv2.imread(args.image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

plt.figure()
plt.axis("off")
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])

hist /= hist.sum()

plt.figure()
plt.title("Grayscale Histogram (Normalized)")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])

plt.show()