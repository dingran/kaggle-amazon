import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from tqdm import tqdm
import os
import glob
from tag_translation import train_label, tags_to_vec, map_predictions

dir_path = os.path.dirname(os.path.realpath(__file__))
code_dir = dir_path
data_dir = os.path.join(dir_path, '../input')


model_paths = glob.glob(os.path.join(code_dir, 'model*.hdf5'))
if model_paths:
    model_path = min(model_paths)
    print('loading ', model_path)
    model_name = os.path.basename(model_path).replace('.hdf5', '')
else:
    print('no model available, abort')
    assert 0

model_name = 'model-0.14135'
raw_prediction_on_train_set = 'raw_pred_{}.pkl'.format(model_name)

special_str = ''
batch_method = 0
prediction_filename = os.path.join(data_dir, '../output/keras_pred_{}{}_BM{}.csv'.format(model_name, special_str, str(batch_method)))

with open(prediction_filename.replace('.csv', '.pkl'), 'rb') as f:
    test_filenames, ytest = pickle.load(f)
print(ytest.shape)

with open(raw_prediction_on_train_set, 'rb') as f:
    ypred_train, ypred_valid, ytrain, yvalid = pickle.load(f)

print(ypred_train.shape, ypred_valid.shape, ytrain.shape, yvalid.shape)
y_pred = np.concatenate((ypred_train, ypred_valid))
y_true = np.concatenate((ytrain, yvalid))


thresholds = np.random.rand(17)

def proba_to_int(yproba, thresh):
    y_pred_t = yproba.copy()
    for i in range(y_pred_t.shape[1]):
        y_pred_t[:,i] = yproba[:,i] > thresh[i]
        #print(y_pred)
    
    y_pred_i = y_pred_t.astype(int)
    return y_pred_i

def fbeta_with_thresholds(thresh, y_true=None, y_pred=None):
    if isinstance(thresh, float):
        thresh = np.ones(17)*thresh
    
    y_pred_i = proba_to_int(y_pred, thresh)
    
    score = fbeta_score(y_true, y_pred_i, beta=2, average='samples')
    #print(score)
    return score*-1

opt_thresh = list(np.ones(17)*.2)
opt_score = fbeta_with_thresholds(opt_thresh, y_true, y_pred)
for i in tqdm(range(len(opt_thresh))):
    for t in np.arange(0.01, .9, .011):
        tmp_thresh = opt_thresh.copy()
        tmp_thresh[i]  = t
        new_score = fbeta_with_thresholds(tmp_thresh, y_true, y_pred)
        if new_score < opt_score :
            opt_thresh[i] = t
            opt_score = new_score

print(opt_score)
print(opt_thresh)

# run again in reverse to check convergence
ilist = list(range(len(opt_thresh)))
for i in tqdm(ilist[::-1]):
    # print(i)
    for t in np.arange(0.01, .9, .011):
        tmp_thresh = opt_thresh.copy()
        tmp_thresh[i]  = t
        new_score = fbeta_with_thresholds(tmp_thresh, y_true, y_pred)
        if new_score < opt_score :
            opt_thresh[i] = t
            opt_score = new_score

print(opt_score)
print(opt_thresh)

ytest_i = proba_to_int(ytest, opt_thresh)


predicted_labels = map_predictions(ytest_i)
predicted_labels_str = [' '.join(x) for x in predicted_labels]
df = pd.DataFrame({'image_name': test_filenames, 'tags': predicted_labels_str})

prediction_filename = os.path.join(data_dir, '../output/keras_pred_{}_opt.csv'.format(model_name))
df.to_csv(prediction_filename, index=False)




