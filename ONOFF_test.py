


import time
import librosa
from dtw import dtw
import librosa.display
from scipy.spatial.distance import cdist
import os
from sklearn.neighbors import KNeighborsClassifier
from pylab import *


dirname = "TRAIN"
files = [f for f in os.listdir(dirname) if not f.startswith('.')]

classifier = KNeighborsClassifier(n_neighbors=3,metric='euclidean')
classifier.fit(distances, y)


y, sr = librosa.load('test/ON17.wav')
mfcc = librosa.feature.mfcc(y, sr)
distanceTest = []
for i in range(len(files)):
    y1, sr1 = librosa.load(dirname+"/"+files[i])
    mfcc1 = librosa.feature.mfcc(y1, sr1)
    dist, _, _, _ = dtw(mfcc.T, mfcc1.T, dist=lambda x, y: norm(x - y, ord=1))
    distanceTest.append(dist)

# pre = classifier.predict(distanceTest)[0] # False
pre = classifier.predict([distanceTest])[0]
print(pre)
label[int(pre)]

print("Predict audio is: '{}'".format(label[int(pre)]))