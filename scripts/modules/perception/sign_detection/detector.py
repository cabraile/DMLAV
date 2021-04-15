import time
import os
import numpy as np
from typing import Union
import pickle
import cv2
import skimage
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog, greycomatrix, greycoprops
import torch

from modules.perception.sign_detection.yolo.utils import *
from modules.perception.sign_detection.yolo.utils.utils import *
from modules.perception.sign_detection.yolo.utils.torch_utils import *
from modules.perception.sign_detection.yolo.models import *

#from yolo.utils import *
#from yolo.utils.utils import *
#from yolo.utils.torch_utils import *
#from yolo.models import *


class BoundingBox:

	def __init__(self, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, prob, class_name):
		self.p1 = (p1x, p1y)
		self.p2 = (p2x, p2y)
		self.p3 = (p3x, p3y)
		self.p4 = (p4x, p4y)
		self.prob = prob
		self.class_name = class_name
		return

class Detector:

	def __init__(self, threshold : float, flag_use_cpu : bool = True, reset_cache : bool = False):

		"""
		Parameters
		==========
		threshold: float.
			The threshold value for accepting a detection as valid.
		flag_use_cpu: bool.
			Whether to use the CPU (True) or the GPU (False).
		reset_cache : bool.
			The recognition model uses KNN on features extracted from a dataset, which take a while.
			The first time executing this detector a file containing the recognition model will be cached.
			In case you find necessary, you can ignore the cache and rewrite it setting this parameter to True.
		"""

		# Init parameters
		self.cache_dir = os.path.dirname(os.path.realpath(__file__)) + "/cache"
		self.general_params = {
			"root_dir": os.path.dirname(os.path.realpath(__file__))
		}
		self.config_recognition = {
			"image_size" : 416,
			"nclasses" : 5,
			"feature_type" : "hog",
			"knn_k" : 5
		}
		self.config_detection = {
			"image_size" : 416,
			"nms_thres" : 0.5, # iou threshold for non-maximum suppression'
			"threshold" : threshold
		} 
		
		# Recognition settings
		self.class_to_ids = {"30":0, "40":1, "50":2, "60":3, "not_park":4}
		self.ids_to_class = {0:"30", 1:"40", 2:"50", 3:"60", 4:"not_park"}
		self.colors = {
			'30':(0, 0, 255), '40': (0,255,0),
			'50':(255,255,0), '60': (122,0,0), 
			'not_park':(255,122,100)
		}

		
		# Load and init detector
		cfg_path 		= self.general_params["root_dir"] + "/yolo/model/yolov3-tiny.cfg"
		weights_path 	= self.general_params["root_dir"] + "/yolo/model/yolov3-tiny.weights"
		self.device 	= select_device(force_cpu = flag_use_cpu, apex=False)
		torch.backends.cudnn.benchmark = False  # set False for reproducible results
		self.detection_model = Darknet(cfg_path, self.config_detection["image_size"])
		if weights_path.endswith('.pt'):  # pytorch format
			self.detection_model.load_state_dict(torch.load(weights_path, map_location=self.device)['model'])
		else:
			_ = load_darknet_weights(self.detection_model, weights_path)
		self.detection_model.to(self.device).eval() # eval mode

		if not self.from_disk(self.cache_dir) or reset_cache:
			print("[Detector.py] Computing recognition model. Might take a while.")
			self.load_recognition_model()
			self.to_disk(self.cache_dir)
			print("[Detector.py] Done computing the recognition model!")
		return

	# IMAGE PROCESSING METHODS
	# ============================

	def equalize_rgb(self, image_rgb : np.array) -> np.array:
		img_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
		img_hsv[:,:,2] = cv2.equalizeHist(image_rgb[:,:,2])
		img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
		return img

	def augment(self, image_rgb : np.array, n_times : int = 10) -> list :
		l_images = [image_rgb]
		for i in range(n_times):
			# pick a random degree of rotation between 25% on the left and 25% on the right
			im_rotated = skimage.transform.rotate(image_rgb,  random.uniform(-25, 25)).astype("uint8")
			# add random noise to the image
			im_noise = skimage.util.random_noise(image_rgb).astype("uint8")
			# blurred image
			sigma = np.random.rand() * 2
			im_blur = cv2.GaussianBlur(image_rgb, ksize=(0,0), sigmaX=sigma).astype("uint8")
			# Append to list
			l_images = l_images + [im_rotated,im_noise, im_blur]
		return l_images

	def preprocess_recognition(self, img_rgb : np.array, equalize : bool = False) -> np.array:
		img_tmp = img_rgb
		if(equalize):
			img_tmp = self.equalize_rgb(img_tmp)
		targ_size = self.config_recognition["image_size"]
		img_out = cv2.resize(img_tmp,(targ_size, targ_size))
		return img_out

	def letterbox(self, img_l : np.array, new_shape : int = 416, color : tuple = (128, 128, 128), mode : str ='auto') -> np.array :
		# Resize a rectangular image to a 32 pixel multiple rectangle
		# https://github.com/ultralytics/yolov3/issues/232
		shape = img_l.shape[:2]  # current shape [height, width]

		if isinstance(new_shape, int):
			ratio = float(new_shape) / max(shape)
		else:
			ratio = max(new_shape) / max(shape)  # ratio  = new / old
		ratiow, ratioh = ratio, ratio
		new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

		# Compute padding https://github.com/ultralytics/yolov3/issues/232
		if mode == 'auto':  # minimum rectangle
			dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
			dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
		elif mode == 'square':  # square
			dw = (new_shape - new_unpad[0]) / 2  # width padding
			dh = (new_shape - new_unpad[1]) / 2  # height padding
		elif mode == 'rect':  # square
			dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
			dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
		elif mode == 'scaleFill':
			dw, dh = 0.0, 0.0
			new_unpad = (new_shape, new_shape)
			ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

		if shape[::-1] != new_unpad:  # resize
			img_l = cv2.resize(img_l, new_unpad, interpolation=cv2.INTER_AREA)  # INTER_AREA is better, INTER_LINEAR is faster
		top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
		left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
		img_l = cv2.copyMakeBorder(img_l, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
		return img_l

	# ============================

	# RECOGNITION TASKS
	# ============================

	def compute_descriptor(self, image_rgb: np.array) -> np.array:
		"""
		Compute the features from an image using the global settings (HOG or GLCM).

		Parameters
		=========
		image_rbg: numpy.array.
			The input image (nrow, ncol, 3).
		
		Returns
		=========
		descriptor: numpy.array.
			An 1D array of the computed features.
		"""
		descriptor = None
		if(self.config_recognition["feature_type"] == "hog"):
			descriptor = hog(
				image_rgb, 
				orientations=8, 
				pixels_per_cell=(32,32),
				cells_per_block=(2, 2), 
				visualize=False, 
				feature_vector=True, 
				multichannel=True
			)
		elif(self.config_recognition["feature_type"] == "glcm"):
			glcms = greycomatrix(
				cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY), 
				distances=[1], 
				angles=[0., np.pi/4, np.pi/2, (3 * np.pi)/4], symmetric=True,
				normed=True
			)
			dsc_list = []
			for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
				dsc_arr =  greycoprops(glcms,prop).flatten()
				dsc_list.append(dsc_arr)
			descriptor = np.vstack(dsc_list).flatten()
		return descriptor

	def load_recognition_model(self):
		# Init
		labels=[]
		height = self.config_recognition["image_size"]
		width = self.config_recognition["image_size"]
		classes = self.config_recognition["nclasses"]

		root_path = self.general_params["root_dir"] + "/street_signs_dataset"
		Class_names = os.listdir(root_path)
		class_ids = {"30":0, "40":1, "50":2, "60":3, "not_park":4}
		data = []

		for name in Class_names:
			path = root_path + "/" + name
			Class=os.listdir(path)
			for a in Class:
				try:
					# Check if is image
					ext = a.split(".")[-1]
					if ext != "ppm" and ext != "jpg" and ext != "png" and ext != "jpeg":
						continue
					# Load image
					image_rgb = cv2.imread(path+"/"+a)[:,:,::-1]
					# Process image
					image_rgb = self.preprocess_recognition(image_rgb)
					# Augment - includes the input image
					l_images = self.augment(image_rgb)
					l_images.append( self.equalize_rgb(image_rgb) )

					for image in l_images:
						# Compute descriptor array
						descriptor = self.compute_descriptor(image)
						# Store the descriptor and its respective class id
						data.append(descriptor)
						labels.append(class_ids[name])
				except AttributeError:
					print("[load_recognition_model] Error during image processing!")

		# Convert the descriptors to an array
		X_train = np.vstack(data)
		y_train = np.array(labels)

		self.recognition_model = KNeighborsClassifier(n_neighbors = self.config_recognition["knn_k"])
		self.recognition_model.fit(X_train, y_train)
		return

	def classify_bounding_boxes(self, img_rgb : np.array, bboxes : BoundingBox, visualize : bool = False) -> Union[list,np.array]:
		labels = []
		img_viz = None
		if visualize:
			img_viz = np.copy(img_rgb)
		for bbox in bboxes:
			p1 = bbox.p1; p3 = bbox.p3
			img_sign_rgb = img_rgb[p1[1]:p3[1], p1[0]:p3[0],:]
			img_sign_rgb = self.preprocess_recognition(img_sign_rgb, equalize=True)
			descriptor	 = self.compute_descriptor(img_sign_rgb)
			class_id 	= self.recognition_model.predict([descriptor])
			label 		= self.ids_to_class[class_id[0]]
			bbox.class_name = label
			labels.append(label)
			if visualize:
				img_viz = cv2.rectangle(img_viz, (p1[0], p1[1]), (p3[0],p3[1]), self.colors[label], 2)
				img_viz = cv2.putText(img_viz, label, (p1[0] - 2, p1[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 1.8, self.colors[label], 2, cv2.LINE_AA)
		return labels, img_viz

	# DETECTION + RECOGNITION
	# ============================

	def detect_and_recognize(
		self, 
		img_rgb : np.array, 
		conf_thres : float = None, 
		nms_thres : float = 0.5, 
		visualize : bool = False
	) -> Union[list, list, np.array]:
		"""
		Detects signs and recognizes which in which class they belong.

		Parameters
		==============
		img_rgb: numpy.array.
			The input image (nrow,ncol,3).
		conf_thresh: float (optional).
			The threshold for accepting a detection (defaults to the threshold 
				passed as argument for the constructor).
		nms_thres: float (optional).
			The threshold for the IOU of the bounding boxes. Defaults to 0.5.
		visualize : bool (optional).
			Whether to provide the detection image (with bounding boxes drawn) or not.
			Defaults to False.

		Returns
		==============
		bboxes: list of BoundingBox.
			The list of the detections as bounding boxes.
		labels: list of string.
			The class names for each bounding box (redundant, since a 
			BoundingBox already carry its own class label).
		img_viz: np.array.
			The (nrow,ncol,3) array of the RGB image with illustrated bounding boxes.
		"""
		if(conf_thres is None):
			conf_thres = self.config_detection["threshold"]
		img = self.letterbox(img_rgb, new_shape=self.config_detection["image_size"], mode="auto")
		img = img.transpose(2, 0, 1) # Swap the channels with the batch dims

		# Normalize RGB
		img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to fp16/fp32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		
		# Get detections
		img = torch.from_numpy(img).unsqueeze(0).to(self.device)
		pred, _ = self.detection_model(img)

		# NMS for overlapping bboxes
		det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0]

		# Recognition
		bboxes = None
		labels = None
		img_viz = None
		if det is not None and len(det) > 0:
			det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_rgb.shape).round()
			bboxes = self.detection_results_to_bounding_boxes(det, img_rgb)
			labels, img_viz = self.classify_bounding_boxes(img_rgb, bboxes, visualize)
		return bboxes, labels, img_viz

	def detection_results_to_bounding_boxes(self, objects, img : np.array) -> list:

		bounding_array = []
		for x0, x1, x2, x3, conf, cls_conf, _cls  in objects:

			prob_c = conf
			
			p1x= int(x0)
			p1y= int(x1)
		
			p2x= int(x0)
			p2y= int(x3)
		
			p3x= int(x2)
			p3y= int(x3)
		
			p4x= int(x2)
			p4y= int(x1)

			b = BoundingBox(p1x, p1y,p2x, p2y, p3x, p3y, p4x, p4y, float(prob_c), None)
			bounding_array.append(b)

		return bounding_array
			
	# ============================

	# CACHE MANAGEMENT
	# ============================

	def from_disk(self,load_dir : str) -> bool:
		flag_loaded_model = False
		flag_loaded_params = False
		if os.path.isfile(load_dir+"/recognition_model.data"):
			with open(load_dir+"/recognition_model.data", "rb") as f_rec_model:
				self.recognition_model = pickle.load(f_rec_model)
				flag_loaded_model = True
		if os.path.isfile(load_dir+"/recognition_model_params.data"):
			with open(load_dir+"/recognition_model_params.data", "rb") as f_rec_params:
				self.config_recognition = pickle.load(f_rec_params)
				flag_loaded_params = True
		return (flag_loaded_model and flag_loaded_params)

	def to_disk(self, save_dir : str) -> bool:
		flag_saved_model = False
		flag_saved_params = False
		with open(save_dir+"/recognition_model.data", "wb") as f_rec_model:
			pickle.dump(self.recognition_model, f_rec_model)
			flag_saved_model = True
		with open(save_dir+"/recognition_model_params.data", "wb") as f_rec_params:
			pickle.dump(self.config_recognition, f_rec_params)
			flag_saved_params = True
		return (flag_saved_model and flag_saved_params)

	# ============================

if __name__=='__main__':
	import glob
	import matplotlib.pyplot as plt
	import time
	
	# If the recognition module was not saved to the disk yet, save.
	# Load from disk otherwise
	detector = Detector(flag_use_cpu=False, threshold=0.85, reset_cache=False)

	test_dir = "C:/Users/carlo/Local_Workspace/Datasets/Dataset_DMBL/images"
	list_img = glob.glob(test_dir + "/*")
	for test_image_path in list_img:
		img = cv2.imread(test_image_path)[:,:,::-1]
		start_time = time.time()
		bboxes, labels, img_viz = detector.detect_and_recognize(img, visualize=True)
		duration = time.time() - start_time
		print(f"\r{test_image_path}: {duration* 1000}ms", end="")
		if(img_viz is not None):
			for bbox in bboxes:
				print(f"\n> Detected {bbox.class_name} with probability {bbox.prob * 100}%")
			plt.imshow(img_viz)
			plt.show()
	exit(0)
	