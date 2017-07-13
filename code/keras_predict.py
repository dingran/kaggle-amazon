import numpy as np
import pandas as pd
import glob
import os
import gc
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from tqdm import tqdm
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import pickle

import glob

dir_path = os.path.dirname(os.path.realpath(__file__))
code_dir = dir_path
data_dir = os.path.join(dir_path, '../input')
print(data_dir)

# prepare lable decoder
train_label = pd.read_csv(os.path.join(data_dir, 'train_v2.csv'))
labels_str = 'agriculture, artisinal_mine, bare_ground, blooming, blow_down, clear, cloudy, conventional_mine, cultivation, habitation, haze, partly_cloudy, primary, road, selective_logging, slash_burn, water'
labels = labels_str.split(', ')
label_map = {x: labels.index(x) for x in labels}


def map_predictions(predictions, labels_map, thresholds=np.ones(17) * 0.2):
    predictions_labels = []
    for prediction in predictions:
        labels = [labels_map[i] for i, value in enumerate(prediction) if value > thresholds[i]]
        predictions_labels.append(labels)
    return predictions_labels


# load model
model_paths = glob.glob(os.path.join(code_dir, 'model*.hdf5'))
if model_paths:
    model_path = min(model_paths)
    print('loading ', model_path)
    model_name = os.path.basename(model_path).replace('.hdf5', '')
else:
    print('no model available, abort')
    assert 0
model = load_model(model_path)


# all test files

def assemble_batch(sub_list):
    print('sublist length ', len(sub_list))
    # X_test = np.empty([len(test_file_list), 299, 299, 3])
    X_test = np.empty([len(sub_list), 299, 299, 3], dtype='float32')
    test_filenames = []
    i = 0
    for t in tqdm(sub_list):
        filename = os.path.basename(t).replace('.jpg', '')
        test_filenames.append(filename)
        image = io.imread(t)
        image = resize(image, (299, 299), mode='constant')  # for InceptionV3
        X_test[i, :, :, :] = image

    # X_test = np.stack(X_test, axis=0)
    print(X_test.shape)
    print(X_test.dtype)

    return X_test, test_filenames


test_file_list = glob.glob(os.path.join(data_dir, 'test-jpg/*.jpg'))

batch_method = 0
if batch_method:
    N_test = len(test_file_list)
    endpoints = list(range(N_test))[::1000] + [N_test]
    print(endpoints)

    ytest_record = []
    test_filename_record = []
    df_list = []
    for i in tqdm(range(len(endpoints) - 1)):
        print('batch{}'.format(i), endpoints[i], endpoints[i + 1])
        sub_list = test_file_list[endpoints[i]:endpoints[i + 1]]
        X_test_batch, test_filenames_batch = assemble_batch(sub_list)
        X_test_batch /= 255
        ytest_batch = model.predict(X_test_batch, verbose=1)
        predicted_labels = map_predictions(ytest_batch, labels)
        predicted_labels_str = [' '.join(x) for x in predicted_labels]
        df_batch = pd.DataFrame({'image_name': test_filenames_batch, 'tags': predicted_labels_str})

        df_list.append(df_batch)
        ytest_record.append(ytest_batch)
        test_filename_record.append(test_filenames_batch)

    df = pd.concat(df_list, axis=0)
    print(df.shape)

    ytest = np.concatenate(ytest_record, axis=0)
    test_filenames = np.concatenate(test_filename_record)
    print(ytest.shape)
    print(test_filenames.shape)
else:

    X_test, test_filenames = assemble_batch(test_file_list)
    X_test /= 255
    ytest = model.predict(X_test, verbose=1)
    predicted_labels = map_predictions(ytest, labels)
    predicted_labels_str = [' '.join(x) for x in predicted_labels]
    df= pd.DataFrame({'image_name': test_filenames, 'tags': predicted_labels_str})

prediction_filename = os.path.join(data_dir, '../output/keras_pred_{}_BM{}.csv'.format(model_name, str(batch_method)))
df.to_csv(prediction_filename, index=False)
with open(prediction_filename.replace('.csv', '.pkl'), 'wb') as f:
    pickle.dump((test_filenames, ytest), f)
