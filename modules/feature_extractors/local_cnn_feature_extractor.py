from tensorflow import keras
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cv2
from modules.feature_extractors.feature_extractor import FeatureExtractor

from joblib import dump, load

class LocalCNNFeatureExtractor(FeatureExtractor):

    def __init__(self, input_shape_dims : tuple, n_keypoints : int):
        self.flag_fit = False
        self.input_shape = input_shape_dims
        shape = (input_shape_dims[0], input_shape_dims[1], 3)
        self.model = keras.applications.vgg16.VGG16(include_top=False,input_shape=shape)
        self.model_preprocess = keras.applications.vgg16.preprocess_input
        self.output_size = 1
        self.pca = None
        self.scaler = None
        for dim in self.model.output_shape:
            if(dim is None):
                continue
            self.output_size *= dim
        return

    def prepare(self, image : np.array) -> np.array:
        """
        Crop, resize and add extra dimension to the input image.

        Parameters
        =======
        img: numpy.array.
            The RGB image to be processed.
            
        Returns
        =======
        img_data: numpy.array.
            The prepared image.
        """
        # Square image
        if(image.shape[0] != image.shape[1]):
            min_dim_size    = min(image.shape[0], image.shape[1])
            center_row      = int(image.shape[0]/2)
            center_col      = int(image.shape[1]/2)
            half            = int(min_dim_size/2)
            img_square      = image[center_row-half: center_row+half, center_col-half : center_col + half]
        else:
            img_square      = image[:,:,:]
        
        # Resize to the default size
        targ_row, targ_col = self.input_shape
        shape = (targ_col, targ_row)
        image_resized = cv2.resize(img_square, shape, None)
        return image_resized

    def extract(self, image : np.array) -> np.array :
        """
        Holistic feature extraction. Returns the output of the feature extraction model.

        Parameters
        =======
        img: numpy.array
            The RGB image from which the features are going to be computed.

        Returns 
        =======
        features: numpy.array
            The flattened feature array
        """
        image_prep              = self.prepare(image)
        img_data_expanded       = np.expand_dims(image_prep, axis=0)
        img_data                = self.model_preprocess(img_data_expanded)
        features                = self.model.predict(img_data).flatten()
        return features

    def extract_batch(self, images_list : list) -> np.array:
        """
        Parameters
        ===========
        images_array: numpy.array.
            (n_images, n_row, n_col, 3) array.

        Returns
        ===========
        descriptors_array : numpy.array.
            (n_images, n_features) array.
        """
        n_images = len(images_list)
        n_features = self.output_size
        targ_row, targ_col = self.input_shape

        # Preprocess
        images_array = np.empty((n_images, targ_row, targ_col, 3))
        for i in range(n_images):
            images_array[i,:,:,:] = self.prepare(images_list[i])
        images_array = self.model_preprocess(images_array)
        
        # Compute
        descriptors_array = self.model.predict(images_array).reshape(n_images, -1) # WARNING: CHECK IF OK!
        return descriptors_array

    def fit(self, descriptors_array : np.array, n_components : int):
        """
        Fits a Starndard Scaler and a PCA models to the extracted descriptors.

        Parameters
        ==============
        descriptors_array: numpy.array.
            The 2D array of descriptors (n_descriptors, descriptor_size).
        n_components: int.
            The number of components to be kept after PCA.
        """
        self.scaler =  StandardScaler().fit(descriptors_array)
        scaled_data = self.scaler.transform(descriptors_array)
        self.pca = PCA(n_components=n_components).fit(scaled_data)
        self.flag_fit = True
        return
    
    def transform(self, descriptors_array : np.array) -> np.array:
        """
        Scales the features and reduces the number of dimensions 
        for each of the the descriptors using PCA.

        Parameters
        ==============
        descriptors_array: numpy.array.
            The 2D array of descriptors (n_descriptors, descriptor_size).

        Returns
        ==============
        array_proc: numpy.array.
            The descriptors' array after PCA.
        """
        assert self.flag_fit, "Error: attempted to transform descriptors before fitting the training data!"
        array_proc = self.scaler.transform(descriptors_array)
        #array_proc = self.pca.transform(array_proc)
        return array_proc

    def save(self, save_dir : str) :
        with open(save_dir + f"/pca.joblib","wb") as f:
            dump(self.pca,f)
        with open(save_dir + f"/scaler.joblib","wb") as f:
            dump(self.scaler,f)
        return

    def load(self, load_dir : str) :
        with open(load_dir + f"/pca.joblib", 'rb') as f:
            self.pca = load(f)
        with open(load_dir + f"/scaler.joblib", 'rb') as f:
            self.scaler = load(f)
        self.flag_fit = True
        return

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import os

    cache_dir = "fex_4096"
    
    # Load dataset
    test_dir = "C:/Users/carlo/Local_Workspace/VPR_Dataset"
    n_components = 700
    input_size = 224
    batch_size = 32
    file_names = os.listdir(test_dir)

    # Init model
    extractor = CNNFeatureExtractor((input_size,input_size))

    print("----------------------")
    print("Loading batches")
    print("----------------------")
    N_files = len(file_names)
    batch_descriptors_list = []
    images_list = []
    for f_idx in range(N_files):
        f_name = file_names[f_idx]
        f_path = test_dir + "/" + f_name
        img_rgb = cv2.imread(f_path)[:,:,::-1]
        images_list.append(img_rgb)
        if(len(images_list) == batch_size):
            print(f"> Prediction for batch {len(batch_descriptors_list)}")
            batch_descriptors = extractor.extract_batch(images_list)
            batch_descriptors_list.append(batch_descriptors)
            images_list = []
    descriptors_array = np.vstack(batch_descriptors_list)

    print("----------------------")
    print("Fit Scaler and PCA")
    print("----------------------")
    load_model = os.path.isfile(f"{cache_dir}/pca.joblib") and os.path.isfile(f"{cache_dir}/scaler.joblib")
    if(not load_model):
        extractor.fit(descriptors_array, n_components)
        extractor.save(cache_dir)
    else:
        extractor.load(cache_dir)

    print("----------------------")
    print("Apply Scaler and PCA")
    print("----------------------")
    descriptors_array_reduced = extractor.transform(descriptors_array)
    
    print("----------------------")
    print("Plot")
    print("----------------------")
    for i in range(descriptors_array_reduced.shape[0]):
        plt.plot(descriptors_array_reduced[i,:])
    plt.show()
    exit(0)