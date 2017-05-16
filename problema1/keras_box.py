import cv2
import numpy as np
from keras.models import load_model, Sequential
from keras.applications.vgg16 import VGG16, preprocess_input

SHAPE = (600, 800, 3)

def load_boxmodel(path):
	SHAPE = (600, 800, 3)

	m1 = VGG16(include_top=False, input_shape=SHAPE)
	m1.layers.pop()
	m2 = load_model(path)

	final_model = Sequential()
	final_model.add(m1)
	final_model.add(m2)

	return final_model

def get_boxes(model, img):
	h, w = img.shape[:2]
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, SHAPE[:2][::-1])
	img = np.expand_dims(img, 0)
	img = preprocess_input(img.astype(np.float32))
	boxes = model.predict(img)[0]
	boxes[:,0] *= w
	boxes[:,1] *= h
	boxes = boxes.astype(np.int32)
	
	return boxes.tolist()