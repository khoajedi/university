from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from jedi.preprocessing import ResizePreprocessor
from jedi.datasets import SimpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
                help="# of nearest neighbors for classification")
args = ap.parse_args()

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args.dataset))

rsp = ResizePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[rsp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

print("[INFO] features matrix: {:.1f} MB".format(
    data.nbytes / (1024 * 1024.0)))

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25)

print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args.neighbors)
model.fit(trainX, trainY)
predY = model.predict(testX)
print(classification_report(testY, predY, target_names=le.classes_))
