"""
Must be placed in same directory as files needing to be analyzed. Files must be in .wav format.
All predictions are written to the file (without quotes) "predictions.txt".
Requires GPUs to run.
"""
import sys
import os
import keras
from keras.models import load_model
import numpy as np
from scipy.io import wavfile
from scipy.signal import decimate
from sklearn.preprocessing import scale
import keras.backend as K
import h5py
def fbeta(y_true, y_pred, threshold_shift=0):
	beta = 2

	y_pred = K.clip(y_pred, 0, 1)
	y_pred_bin = K.round(y_pred + threshold_shift)

	tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
	fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
	fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

	precision = tp / (tp + fp)
	recall = tp / (tp + fn)

	beta_squared = beta ** 2
	return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()))


def preprocess(data):

	#standardize length
	MAXLEN = 396900*3
	if len(data) < MAXLEN:
		data = np.tile(data, int(np.ceil(MAXLEN/len(data))))
	data = data[:MAXLEN]

	#low-pass filter
	data = decimate(data, 8, zero_phase=True)

	#scale data
	data = scale(data)
	#reshape data
	data = data.reshape(1, *data.shape, 1)

	return data
def predict(path):
	model = load_model("model.h5", custom_objects={'fbeta':fbeta})
	X = preprocess(wavfile.read(path)[1])
	probs = model.predict(X)
	prob = 1
	if probs[0] < 0.5:
		pred = 1.0 - probs[0]
	else:
		pred = probs[0]
	if probs[0] < 0.7:
		prob = 0
	print(probs[0])
	return ["murmur", "normal"][prob], pred
def getdigits(name):
	digits = []
	for i in name:
		if i.isdigit():
			digits.append(int(i))
	return digits
def main():
	y_pred = []
	names = []
	confidences = []
	count = 0
	try:
		for filename in os.listdir('.'):
			count += 1
			if filename.endswith(".wav"):
				p = predict(filename)
				y_pred.append(p[0])
				names.append(filename)
				confidences.append(str(p[1]))
	except KeyboardInterrupt:
		print("Interrupted by user.")
		pass
	f = open("predictions.csv", "w")
	map1 = {'murmur':'class I', 'normal':'class III'}
	map2 = {'murmur':'p', 'normal':'n'}
	f.write("wavfile name,AHA class,Murmur,Systolic Grade,Diastolic Grade,Timing,Heart Rate,Confidence level\n")
	for pred, name, confidence in zip(y_pred, names, confidences):
		s = ''.join([str(x) for x in getdigits(name)[:6]])
		s = s[:4]+'_'+s[4:]
		f.write(s+'.wav'+','+map1[pred]+','+map2[pred]+', , , , ,'+confidence[1:-1]+'\n')
	f.close()
if __name__ == "__main__":
	main()


