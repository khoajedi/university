from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import jedi.houses as hs
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="./datasets/houses",
                help="path to input dataset of house images")
args = ap.parse_args()

print("[INFO] loading house attributes...")
inputPath = os.path.sep.join([args.dataset, "info.txt"])
df = hs.load_attributes(inputPath)

print("[INFO] loading house images...")
images = hs.load_images(df, args.dataset)
images = images / 255.0

(trainAttrX, testAttrX, trainImgX, testImgX) = train_test_split(
    df, images, test_size=0.25)

maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice

model = hs.create_cnn(64, 64, 3, regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

print("[INFO] training model...")
model.fit(x=trainImgX, y=trainY,
          validation_data=(testImgX, testY),
          epochs=200, batch_size=8)

print("[INFO] predicting house prices...")
predY = model.predict(testImgX)
diff = predY.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

print("[INFO] avg. house price: {}, std house price: {}".format(
    df["price"].mean(), df["price"].std()))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
