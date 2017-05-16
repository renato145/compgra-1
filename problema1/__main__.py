import pdb
import os
import cv2
import keras_box
import numpy as np
from time import time
from process_image import process_image, four_point_transform

PATH = 'data/placas'
SAVE_PATH = 'problema1/data/save.npy'
DIALOG_EXIT = 'Really want to exit (y)?'
INSTRUCTIONS ='''---------------------------
Process image      : Espace

Predict markers    : t
Previous marker    : q
Next marker        : e
Move marker        : awds

Previous image     : z
Next image         : c

Save marker        : f
Switch lvl mark    : x
(easy = green, hard = red)

Save results       : g

Exit               : escape
---------------------------
'''

class App(object):
	def __init__(self, title='main'):
		# main window
		self.i = 0
		self.box = []
		self.hard = False
		self.edit = False
		self.marker = 0
		self.main_window = title
		self.load()
		self.sres = None
		# Box model
		self.box_model = keras_box.load_boxmodel('problema1/model.h5')
		self.extra_windows = []
		self.methods = ['Adaptive + OTSU', 'Adaptive', 'OTSU']
		self.method = 0
		self.blur = 7
		self.invert = 0
		self.do_inv = True
		self.processed = None

	def run(self):
		print(INSTRUCTIONS)
		cv2.namedWindow(self.main_window)
		cv2.setMouseCallback(self.main_window, self.click_event)
		self.setup_trackbars()

		while True:
			if self.edit:
				self.show_warped()

			self.draw_lvl_mark()
			cv2.imshow(self.main_window, self.img)
			key = cv2.waitKey(1) & 255

			if key == 255:
				continue

			# save results
			elif key == ord('g'):
				self.save_results()

			# predict markers
			elif key == ord('t'):
				self.get_boxes()
				
			# save marker
			elif key == ord('f'):
				if self.edit:
					self.data[self.files[self.i]] = {'box': np.asarray(self.box, dtype=np.int32),
													 'hard': self.hard}
					self.save()

			# switch lvl mark
			elif key == ord('x'):
				self.hard = not(self.hard)

			# previous image
			elif key == ord('z'):
				self.close_extra_windows()
				self.load_cache()
				self.i = self.n_img - 1 if self.i == 0 else self.i - 1
				self.load_img()
				self.do_inv = True
				self.processed = None

			# next image
			elif key == ord('c'):
				self.close_extra_windows()
				self.load_cache()
				self.i = 0 if self.i == (self.n_img - 1) else self.i + 1
				self.load_img()
				self.do_inv = True
				self.processed = None

			# close
			elif key == 27:
				if self.exit():
					break

			# Interest zone functions
			elif self.edit:
				# prev marker
				if key == ord('q'):
					self.marker = 3 if self.marker == 0 else self.marker - 1
				# next marker
				elif key == ord('e'):
					self.marker = 0 if self.marker == 3 else self.marker + 1
				# left
				elif key == ord('a'):
					self.box[self.marker][0] -= 1
				# right
				elif key == ord('d'):
					self.box[self.marker][0] += 1
				# up
				elif key == ord('w'):
					self.box[self.marker][1] -= 1
				# down
				elif key == ord('s'):
					self.box[self.marker][1] += 1
				# process image
				elif key == 32:
					self.show_processed_image()

	def get_boxes(self):
		t0 = time()
		print('Getting boxes...', end='\r')
		self.box = keras_box.get_boxes(self.box_model, self.img_bk)
		self.edit = True
		# self.show_processed_image()
		print('Done (%.2fs).     ' % (time() - t0))

	def setup_trackbars(self):
		cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Trackbars', 400, 50)
		cv2.createTrackbar('Method', 'Trackbars', 0, len(self.methods)-1, self.callback_method)
		cv2.createTrackbar('Blur', 'Trackbars', self.blur, 30, self.callback_blur)
		cv2.createTrackbar('Invert', 'Trackbars', self.invert, 1, self.callback_invert)

	def callback_method(self, value):
		self.method = value
		print(f'METHOD CHANGED TO: {self.methods[self.method]}')
		if self.edit:
			self.show_processed_image()

	def callback_blur(self, value):
		if value > 0:
			self.blur = value
			if self.edit:
				self.show_processed_image()

	def callback_invert(self, value):
		self.invert = value
		if self.edit:
			self.show_processed_image()

	def show_processed_image(self):
		if self.do_inv:
			b, g, r = np.mean(self.warped, (0,1))
			if b > r+10 and g > r+10:
			    self.invert = 1
			else:
				self.invert = 0
			cv2.setTrackbarPos('Invert', 'Trackbars', self.invert)

		processed = process_image(self.warped, self.blur, self.method, self.invert)
		cv2.namedWindow('processed', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('processed', processed.shape[1]*3, processed.shape[0]*3)
		self.extra_windows.append('processed')
		cv2.imshow('processed', processed)

		if 'sres' in self.extra_windows:
			processed_sres = process_image(self.sres, self.blur, self.method, self.invert)
			cv2.namedWindow('processed-sres', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('processed-sres', processed_sres.shape[1]*3, processed_sres.shape[0]*3)
			self.extra_windows.append('processed-sres')
			cv2.imshow('processed-sres', processed_sres)

		self.do_inv = False
		self.processed = processed

	def save_results(self):
		if self.processed is not None:
			path_src = 'problema1/results'
			x = len([f for f in os.listdir(path_src) if f[-3:] == 'jpg']) // 2
			h, w = self.warped.shape[:2]
			h = 200 * h // w
			img1 = cv2.resize(self.warped.copy(), (200, h))
			img2 = np.repeat(np.expand_dims(self.processed, -1), 3, 2)
			save_img = np.hstack([img1, img2])
			path = os.path.join(path_src, f'plate_{x}.jpg')
			path_car = os.path.join(path_src, f'plate_car_{x}.jpg')
			cv2.imwrite(path_car, self.img_bk)
			cv2.imwrite(path, save_img)
			print('Saved: ' + path)

	def close_extra_windows(self):
		for w in self.extra_windows:
			cv2.destroyWindow(w)

		self.extra_windows.clear()
		self.sres = None

	def load(self):
		if os.path.exists(SAVE_PATH):
			self.data = np.load(SAVE_PATH).item()
		else:
			self.create_save_file()

		self.i = 0
		self.files = list(self.data)
		self.n_img = len(self.files)
		self.load_img()

	def create_save_file(self):
		folders = [os.path.join(PATH, f) for f in os.listdir(PATH)]
		files = []

		for folder in folders:
			for f in os.listdir(folder):
				if f[-3:] == 'jpg':
					files.append(os.path.join(folder, f))

		self.data = {f: {'box': None, 'hard': False} for f in files}
		self.save(cache=False)

	def load_img(self):
		self.box.clear()
		file = self.files[self.i]
		
		if self.data[file]['box'] is None:
			self.edit = False 
			self.hard = False
		else:
			self.edit = True
			self.box = self.data[file]['box'].tolist()
			self.hard = self.data[file]['hard']

		self.img = cv2.imread(file)
		self.img_bk = self.img.copy()
		self.save_cache()

	def save_cache(self):
		self.cache = self.data[self.files[self.i]]

	def load_cache(self):
		self.data[self.files[self.i]] = self.cache

	def draw_lvl_mark(self):
		c = (0,255,0) if self.hard == False else (0,0,255)
		cv2.rectangle(self.img, (8,8), (50,50), c, -1)

	def reload_img(self):
		self.img = self.img_bk.copy()

	def click_event(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			if len(self.box) == 4:
				self.box[self.marker] = [x,y]
				return

			self.draw_point(x, y)
			self.box.append([x,y])
			if len(self.box) == 4:
				self.edit = True
				self.do_inv = True

		elif event == cv2.EVENT_RBUTTONDOWN:
			self.do_inv = True
			self.edit = False
			self.box.clear()
			self.reload_img()

	def draw_text(self, img, text, size=0.5, width=2):
		cv2.putText(img, text, (200, 200), cv2.FONT_HERSHEY_SIMPLEX,
					size, (0, 0, 0), width * 2)
		cv2.putText(img, text, (200, 200), cv2.FONT_HERSHEY_SIMPLEX,
					size, (255, 255, 255), width)

	def draw_box(self):
		points = np.asarray(self.box, dtype=np.int32)
		cv2.polylines(self.img, [points], 1, (0,255,0), 1)

	def draw_point(self, x, y, c=(0,255,0)):
		cv2.drawMarker(self.img, (x,y), c, 1, 15, 2)

	def show_warped(self):
		self.reload_img()
		for i, (x,y) in enumerate(self.box):
			c = (0,0,255) if i == self.marker else (0,255,0)
			self.draw_point(x, y, c)

		self.draw_box()
		points = np.asarray(self.box, dtype=np.int32)
		warped = four_point_transform(self.img_bk, points)
		cv2.namedWindow('warped', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('warped', warped.shape[1]*2, warped.shape[0]*2)
		cv2.imshow('warped', warped)
		self.extra_windows.append('warped')
		self.warped = warped

	def save(self, cache=True):
		np.save(SAVE_PATH, self.data)
		if cache:
			self.save_cache()

		print(f'Saved on: {SAVE_PATH}')

	def exit(self):
		confirm_exit = self.img.copy()
		self.draw_text(confirm_exit, DIALOG_EXIT, 1, 4)
		cv2.imshow(self.main_window, confirm_exit)
		key = cv2.waitKey(2500) & 255

		if key == ord('y'):
			marked_files = 0
			hard_files = 0
			for file in self.data:
				if self.data[file]['box'] is not None:
					marked_files += 1
				if self.data[file]['hard']:
					hard_files += 1
			print('-' * 23)
			print(f' Number of files : {self.n_img:03}')
			print(f' Marked files    : {marked_files:03}')
			print(f' Hard files      : {hard_files:03}')
			print('-' * 23)

			return True

if __name__ == '__main__':
	app = App()
	app.run()
