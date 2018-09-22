import numpy as np
import h5py

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import json


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

h5f_data.close()
h5f_label.close()
h5f_test.close()

# verify the shape of features and labels
print ("[INFO] features shape: {}".format(features.shape))
print ("[INFO] labels shape: {}".format(labels.shape))

print ("[INFO] training started...")
# split the training and testing data
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
                                                                  np.array(labels),
                                                                  test_size=test_size,
                                                                  random_state=seed)

print ("[INFO] splitted train and test data...")
print ("[INFO] train data  : {}".format(trainData.shape))
print ("[INFO] test data   : {}".format(testData.shape))
print ("[INFO] train labels: {}".format(trainLabels.shape))
print ("[INFO] test labels : {}".format(testLabels.shape))
