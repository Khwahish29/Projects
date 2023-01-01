from PIL import Image
import numpy as np
import pylab
import sys
import glob
import os
from . import pca


class EigenFaces(object):
    def train(self, training_images):
        self.projected_classes = []
        self.count_captures = 0
        self.count_timer=0
        self.list_array_images, self.list_label, \
            fclass_samples_list = \
                read_images(training_images)

        image_matrix = np.array([np.array(Image.fromarray(img)).flatten()
              for img in self.list_array_images],'f')

        self.eigen_matrix, variance, self.mean_image = pca.pca(image_matrix)

        for class_sample in fclass_samples_list:
            class_weights = self.project_image(class_sample)
            self.projected_classes.append(class_weights.mean(0))

    def project_image(self, X):
        X = X - self.mean_image
        return np.dot(X, self.eigen_matrix.T)

    def predict_face(self, X):
        min_class = -1
        min_distance = np.finfo('float').max
        projected_target = self.project_image(X)
        projected_target = np.delete(projected_target, -1)
        for i in range(len(self.projected_classes)):
            distance = np.linalg.norm(projected_target - np.delete(self.projected_classes[i], -1))
            if distance < min_distance:
                min_distance = distance
                min_class = self.list_label[i]
        return min_class

    def __repr__(self):
        return "PCA (num_components=%d)" % (self.numerical_comp)


def read_images(path, sz=None):

    class_samples_list = []
    class_matrix = []
    images, image_labels = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdir in dirnames:
            path = os.path.join(dirname, subdir)
            class_samples_list = []
            for filename in os.listdir(path):
                if filename != ".DS_Store":
                    try:
                        im = Image.open(os.path.join(path, filename))
                        if (sz is not None):
                            im = im.resize(sz, Image.ANTIALIAS)
                        images.append(np.asarray(im, dtype = np.uint8))
                    except IOError as e:
                        errno, strerror = e.args
                        print("I/O error({0}): {1}".format(errno, strerror))
                    except:
                        print("Unexpected error:", sys.exc_info()[0])
                        raise

                    class_samples_list.append(np.asarray(im, dtype = np.uint8))

            class_samples_matrix = np.array([img.flatten()
                for img in class_samples_list],'f')

            class_matrix.append(class_samples_matrix)

            image_labels.append(subdir)

    return images, image_labels, class_matrix