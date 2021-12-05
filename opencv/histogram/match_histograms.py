from skimage import exposure
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True,
                help="path to the input source image")
ap.add_argument("-r", "--reference", required=True,
                help="path to the input reference image")
args = vars(ap.parse_args())

src = cv2.imread(args["source"])
ref = cv2.imread(args["reference"])

multi = True if src.shape[-1] > 1 else False
matched = exposure.match_histograms(src, ref, channel_axis=-1)

cv2.imshow("Source", src)
cv2.imshow("Reference", ref)
cv2.imshow("Matched", matched)
cv2.waitKey(0)
