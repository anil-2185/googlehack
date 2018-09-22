from sklearn.decomposition import PCA
# organize imports
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py
import os
import json
import pickle
# import seaborn as sns
# import matplotlib.pyplot as plt

# load the user configs
with open('conf.json') as f:
    config = json.load(f)

# config variables
test_size = config["test_size"]
seed = config["seed"]
features_path = config["features_path"]
labels_path = config["labels_path"]
results = config["results"]
classifier_pat = config["classifier_path"]
test_features_path = config["test_features"]
train_path = config["train_path"]
num_classes = config["num_classes"]
test_index = config["test_index"]
reduce_features = config["reduce_features"]
reduce_test = config["reduce_test"]
classifier_path = config["classifier_path"]

# import features and labels
h5f_data = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')
h5f_test = h5py.File(test_features_path, 'r')

features_string = h5f_data['dataset_1']
labels_string = h5f_label['dataset_1']
test_string = h5f_test['dataset_1']

features = np.array(features_string)
labels = np.array(labels_string)
test = np.array(test_string)

h5f_test.close()
h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print ("[INFO] features shape: {}".format(features.shape))
print ("[INFO] labels shape: {}".format(labels.shape))


print ("[INFO] splitted train and test data...")
print ("[INFO] train data  : {}".format(features.shape))
print ("[INFO] test data   : {}".format(test.shape))
print ("[INFO] train labels: {}".format(labels.shape))

# use logistic regression as the model
print ("[INFO] creating model...")

model = PCA(n_components=2000)
data = model.fit_transform(features, labels)
test_data = model.transform(test)

print ("[INFO] writing...")
h5f_data = h5py.File(reduce_features, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(data))

h5f_data = h5py.File(reduce_test, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(test_data))
