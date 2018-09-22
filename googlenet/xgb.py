# organize imports
from __future__ import print_function

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

model = LogisticRegression(random_state=seed)
model.fit(features, labels)

# use rank-1 and rank-5 predictions
print ("[INFO] evaluating model...")

filenames = []
f = open(test_index, "r")
for r in f:
    filenames.append(r.strip())
f.close()

# evaluate the model of test data
preds = model.predict(test)
labels = sorted(list(os.listdir(train_path)))

f = open(results, "w")
f.write("Filename,Category\n")
for i, name in enumerate(filenames):
    # write the classification report to file
    f.write("{},{}\n".format(name, labels[preds[i]]))
f.close()

# dump classifier to file
print ("[INFO] saving model...")
pickle.dump(model, open(classifier_path, 'wb'))

# display the confusion matrix
print ("[INFO] confusion matrix")
