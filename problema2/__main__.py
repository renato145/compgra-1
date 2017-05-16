import pdb
import os
import cv2
import numpy as np
from time import time
from skimage import measure

DIALOG_EXIT = 'Really want to exit (y)?'
INSTRUCTIONS ='''---------------------------
Next image         : c

Exit               : escape x2
---------------------------
'''

def mark(img, mask, color=(0,255,0)):
	out = img.copy()
	out[mask == 255] = color
	
	return out

def count_leucocitos(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(gray,140,255,cv2.THRESH_BINARY_INV)
	mask = np.zeros(thresh.shape, dtype="uint8")
	labels = measure.label(thresh, neighbors=4, background=255)
	cells = []

	for label in np.unique(labels):
		if label == 0:
			continue

		labelMask = np.zeros(thresh.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)

		if numPixels < 100:
			cells.append(labelMask)
			mask = cv2.add(mask, labelMask)

	return thresh, mask, cells

def count_all(img):
	out = img.copy()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)
	mask = np.zeros(thresh.shape, dtype="uint8")
	labels = measure.label(thresh, neighbors=4, background=0)
	cells = []

	for label in np.unique(labels):
		if label == 0:
			continue

		labelMask = np.zeros(thresh.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)

		if 10 < numPixels < 250:
			cells.append(labelMask)
			mask = cv2.add(mask, labelMask)

	# for cell in cells:
	# 	xs, ys = np.where(cell == 255)
	# 	x = int(np.round(np.mean([np.max(xs)])))
	# 	y = int(np.round(np.mean([np.max(ys)])))
	# 	cv2.drawMarker(out, (x,y), (0,255,0), 1, 0, 5)

	return thresh, mask, cells
	# return out, cells

class App(object):
	def __init__(self, title='main'):
		# main window
		self.i = 0
		self.files = ['data/Leucemia/pathology_cll20x01.jpg', 'data/Leucemia/pathology_cll40x03.jpg']
		self.box = []
		self.main_window = title
		self.extra_windows = []
		self.ready = False
		self.load_img()

	def run(self):
		print(INSTRUCTIONS)
		cv2.namedWindow(self.main_window, cv2.WINDOW_NORMAL)
		cv2.resizeWindow(self.main_window, 800, 600)
		cv2.setMouseCallback(self.main_window, self.click_event)

		while True:
			cv2.imshow(self.main_window, self.img)
			key = cv2.waitKey(1) & 255

			if key == 255:
				continue

			# next image
			elif key == ord('c'):
				self.close_extra_windows()
				self.i = 1 if self.i == 0 else 0
				self.ready = False
				self.load_img()

			# close
			elif key == 27:
				if self.exit():
					break

	def show_result(self):
		zoom = 3
		x1,y1,x2,y2 = self.get_points()
		img = self.img_bk[y1:y2,x1:x2].copy()
		cv2.namedWindow('img', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('img', img.shape[1]*zoom, img.shape[0]*zoom)
		cv2.imshow('img', img)
		self.extra_windows.append('img')
		
		thresh, mask, cells = count_leucocitos(img)
		leucocitos = mark(img, mask)
		cv2.namedWindow('leucocitos', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('leucocitos', leucocitos.shape[1]*zoom, leucocitos.shape[0]*zoom)
		cv2.imshow('leucocitos', leucocitos)
		self.extra_windows.append('leucocitos')
		n_leucocitos = len(cells)

		thresh, mask, cells = count_all(img)
		kernel = np.ones((3,3),np.uint8)
		mask = cv2.dilate(mask, kernel, iterations = 1)
		all_cells = mark(img, mask)
		# all_cells, cells = count_all(img)
		cv2.namedWindow('all_cells', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('all_cells', all_cells.shape[1]*zoom, all_cells.shape[0]*zoom)
		cv2.imshow('all_cells', all_cells)
		self.extra_windows.append('all_cells')
		n_all = len(cells)
		# n_globulos = n_all - n_leucocitos

		print('-'*30)
		print(f'Leucocitos = {n_leucocitos}')
		print(f'All cells  = {n_all}')
		print(f'Proporcion de leucocitos: %.2f' % (n_leucocitos / n_all))
		print('-'*30)

	def close_extra_windows(self):
		for w in set(self.extra_windows):
			cv2.destroyWindow(w)

		self.extra_windows.clear()

	def load_img(self):
		self.box.clear()
		file = self.files[self.i]
		self.img = cv2.imread(file)
		if file == 'data/Leucemia/pathology_cll40x03.jpg':
			n = 2
			h, w, _ = self.img.shape
			self.img = cv2.resize(self.img, (w//n,h//n))

		self.img_bk = self.img.copy()

	def reload_img(self):
		self.img = self.img_bk.copy()

	def click_event(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			if len(self.box) == 2:
				self.box.clear()
				self.reload_img()

			self.draw_point(x, y)
			self.box.append([x,y])

			if len(self.box) == 2:
				self.draw_box()
				self.show_result()

	def get_points(self):
		x1,y1 = self.box[0]; x2,y2 = self.box[1]
		return x1,y1,x2,y2

	def draw_box(self):
		x1,y1,x2,y2 = self.get_points()
		cv2.rectangle(self.img, tuple(self.box[0]), tuple(self.box[1]), (0,255,0), 2)

	def draw_point(self, x, y, c=(0,255,0)):
		cv2.drawMarker(self.img, (x,y), c, 1, 15, 2)

	def exit(self):
		print('Confirm exit: Esc')
		key = cv2.waitKey(2500) & 255
		if key == 27:
			print(':D!')
			return True
		else:
			print('-'*30)


if __name__ == '__main__':
	app = App()
	app.run()
