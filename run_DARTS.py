__author__ = 'taras-sereda'
import numpy as np
import os
from scipy.io import loadmat
from DARTS_supplementary import read_gt, DARTS_bisection, DARTS_eval
import cPickle as pickle


files_location = "/Users/taras-sereda/imageNet/anytime_recognition/ext/hedging-1.0/code/"
features_dir = os.path.join(files_location, '../features')
models_dir = os.path.join(files_location, '../models')

model_location = os.path.join(models_dir,"ilsvrc65.subset0.C100.model.mat")
meta_location = os.path.join(files_location, "ilsvrc65_meta.mat")

train_mat_location = os.path.join(files_location, "train_data_7.0.mat")
val_mat_location = os.path.join(files_location, "val_data_7.0.mat")
test_mat_location = os.path.join(files_location, "test_data_7.0.mat")


val_leaf_probs_loc = os.path.join(files_location, "val_leaf_probs.mat")
test_leaf_probs_loc = os.path.join(files_location, "test_leaf_probs.mat")

accuracy_guarantees = np.append(np.arange(0,0.9,0.1), np.array([0.85,0.9,0.95,0.99]))
num_bs_iters = 25
confidence = .95

class_structure = loadmat(meta_location)
tree = class_structure['synsets']

print("loading gt files...")

train_file = os.path.join(files_location, "ilsvrc65.train.gt")
val_file = os.path.join(files_location, "ilsvrc65.val.gt")
test_file = os.path.join(files_location, "ilsvrc65.test.gt")

train_gt_map, train_gt_ids, train_gt_weights = read_gt(train_file)
val_gt_map, val_gt_ids, val_gt_weights = read_gt(val_file)
test_gt_map, test_gt_ids, test_gt_weights = read_gt(test_file)

print("Loading train")
train_data = loadmat(train_mat_location, variable_names=['ids'])
print("Loading val")
val_data = loadmat(val_mat_location)
print("Loading test")
test_data = loadmat(test_mat_location)


# Make labels
train_labels = np.zeros(train_data['ids'].size, dtype=np.int)
val_labels = np.zeros(val_data['ids'].size, dtype=np.int)
test_labels = np.zeros(test_data['ids'].size, dtype=np.int)
with open("class2id_map.pkl", 'rb') as f:
    class2id_map = pickle.load(f)


for i in range(len(train_labels)):
    train_labels[i] = class2id_map[train_gt_map[train_data['ids'][i][0][0]]]

for i in range(len(val_labels)):
    val_labels[i] = class2id_map[val_gt_map[val_data['ids'][i][0][0]]]

for i in range(len(test_labels)):
    test_labels[i] = class2id_map[test_gt_map[test_data['ids'][i][0][0]]]

num_leaves = len(set(train_labels))

print("Loading model...")
loadmat(model_location)


print("Loading val leaf probs values")
val_leaf_probs = loadmat(val_leaf_probs_loc)['leaf_probs']
print("Loading test leaf porbs values")
test_leaf_probs = loadmat(test_leaf_probs_loc)['test_leaf_probs']


methods = {}
# Run each pf the methods
print("DARTS")

lambdas = DARTS_bisection(accuracy_guarantees, val_leaf_probs, val_labels,
                          tree, num_bs_iters, confidence)

test_results = DARTS_eval(test_leaf_probs, test_labels, tree, lambdas)

methods['DARTS'] = test_results
print methods['DARTS']

