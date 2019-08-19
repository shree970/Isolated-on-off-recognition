import time
import librosa
from dtw import dtw
import librosa.display
from scipy.spatial.distance import cdist
from pylab import *




y1, sr1 = librosa.load('TRAIN/fcmc0-a1-t.wav')
y2, sr2 = librosa.load('TRAIN/fcmc0-b1-t.wav')

#pylab inline
subplot(1, 2, 1)
mfcc1 = librosa.feature.mfcc(y1, sr1)
librosa.display.specshow(mfcc1)

subplot(1, 2, 2)
mfcc2 = librosa.feature.mfcc(y2, sr2)
librosa.display.specshow(mfcc2)

# Calculate the DTW between the 2 sample audios 'a' and 'b'
dist, cost, path, _ = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
print('Normalized distance between the two sounds:', dist)   



#MAIN Program
import os
dirname = "TRAIN"
files = [f for f in os.listdir(dirname) if not f.startswith('.')]

start = time.clock()
minval = 200
distances = ones((len(files), len(files)))
y = ones(len(files))

for i in range(len(files)):
    y1, sr1 = librosa.load(dirname+"/"+files[i])
    mfcc1 = librosa.feature.mfcc(y1, sr1)
    for j in range(len(files)):
        y2, sr2 = librosa.load(dirname+"/"+files[j])
        mfcc2 = librosa.feature.mfcc(y2, sr2)
        dist, _, _, _ = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
#         print files[i],mfcc1.T[0][0],mfcc2.T[0][0],files[j],dist
#         if dist<minval:
#             minval = dist
        distances[i,j] = dist
    if i%2==0:
        y[i] = 0  #'ON'
    else:
        y[i] = 1  #'OFF'
print("Time used: {}s".format(time.clock()-start))

distances[0] # A dictionary


label = ['ON','OFF']


#Train a kNN classifier to determine if the audio is 'ON' or 'OFF'

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3,metric='euclidean')
classifier.fit(distances, y)


#testing

y, sr = librosa.load('test/OFF17.wav')
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