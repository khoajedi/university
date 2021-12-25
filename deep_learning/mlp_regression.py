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

print("[INFO] constructing training/testing split...")
(train, test) = train_test_split(df, test_size=0.25)

maxPrice = train["price"].max()
trainY = train["price"] / maxPrice
testY = test["price"] / maxPrice

print("[INFO] processing data...")
(trainX, testX) = hs.process_attributes(df, train, test)

model = hs.create_mlp(trainX.shape[1], regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

print("[INFO] training model...")
model.fit(x=trainX, y=trainY,
          validation_data=(testX, testY),
          epochs=200, batch_size=8)

print("[INFO] predicting house prices...")
predY = model.predict(testX)

diff = predY.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

print("[INFO] avg. house price: {}, std house price: {}".format(
    df["price"].mean(), df["price"].std()))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))