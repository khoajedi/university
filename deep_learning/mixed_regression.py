from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
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

print("[INFO] processing data...")
split = train_test_split(df, images, test_size=0.25, random_state=42)
(trainAttrX, testAttrX, trainImgX, testImgX) = split

maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice

(trainAttrX, testAttrX) = hs.process_attributes(df, trainAttrX, testAttrX)

mlp = hs.create_mlp(trainAttrX.shape[1], regress=False)
cnn = hs.create_cnn(64, 64, 3, regress=False)

combinedInput = concatenate([mlp.output, cnn.output])

x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)

model = Model(inputs=[mlp.input, cnn.input], outputs=x)

opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

print("[INFO] training model...")
model.fit(
    x=[trainAttrX, trainImgX], y=trainY,
    validation_data=([testAttrX, testImgX], testY),
    epochs=200, batch_size=8)

print("[INFO] predicting house prices...")
predY = model.predict([testAttrX, testImgX])

diff = predY.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

print("[INFO] avg. house price: {}, std house price: {}".format(
    df["price"].mean(), df["price"].std()))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))