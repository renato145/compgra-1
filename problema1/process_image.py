import cv2
import numpy as np

IMG_WIDTH = 200

def process_image(img, blur, process, invert):
	# 'Adaptive + OTSU', 'Adaptive', 'OTSU'
	h, w = img.shape[:2]
	h = IMG_WIDTH * h // w
	out = cv2.resize(img.copy(), (IMG_WIDTH, h))
	if invert > 0:
		out *= 255

	out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
	out = cv2.bilateralFilter(out,blur,75,75)
	if process == 1:
		out = cv2.adaptiveThreshold(out,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
	elif process == 2:
		_, out = cv2.threshold(out,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	elif process == 0:
		th1 = cv2.adaptiveThreshold(out,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
		_, th2 = cv2.threshold(out,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		out = th1 + th2

	return out

def four_point_transform(image, pts):
	rect = np.array(pts, dtype="float32")
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype="float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped
