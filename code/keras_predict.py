import numpy as np
import pandas as pd
import os
from skimage import io
from skimage.transform import resize
from tqdm import tqdm
from keras.models import load_model
from tag_translation import train_label, tags_to_vec, map_predictions
import pickle

import glob

dir_path = os.path.dirname(os.path.realpath(__file__))
code_dir = dir_path
data_dir = os.path.join(dir_path, '../input')
print(data_dir)

model_type = 'inception'
model_type_file = os.path.join(code_dir, 'model_type.txt')

if os.path.exists(model_type_file):
    with open(model_type_file, 'r') as f:
        model_type = f.readline()
        print(model_type)

if model_type.startswith('i'):
    print('using InceptionV3')
    image_shape = (299, 299)
else:
    print('using ResNet50')
    image_shape = (224, 224)

def assemble_batch(sub_list):
    print('sublist length ', len(sub_list))
    # X_test = np.empty([len(test_file_list), 299, 299, 3])
    X_test = np.empty([len(sub_list), image_shape[0], image_shape[1], 3], dtype='float32')
    test_filenames = []
    i = 0
    for t in tqdm(sub_list):
        filename = os.path.basename(t).replace('.jpg', '')
        test_filenames.append(filename)
        image = io.imread(t)
        image = resize(image, image_shape, mode='constant')  # for InceptionV3
        X_test[i, :, :, :] = image
        i += 1

    # X_test = np.stack(X_test, axis=0)
    print(X_test.shape)
    print(X_test.dtype)

    return X_test, test_filenames


test_file_list = glob.glob(os.path.join(data_dir, 'test-jpg/*.jpg'))
predict_on_train = False
if predict_on_train:
    print('!!!!!!!!!!!!!!!!!!! predicting on training set')
    test_file_list = glob.glob(os.path.join(data_dir, 'train-jpg/*.jpg'))

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
        predicted_labels = map_predictions(ytest_batch)
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
    predicted_labels = map_predictions(ytest)
    predicted_labels_str = [' '.join(x) for x in predicted_labels]
    df = pd.DataFrame({'image_name': test_filenames, 'tags': predicted_labels_str})

special_str = ''
if predict_on_train:
    special_str = '_on_train'

prediction_filename = os.path.join(data_dir, '../output/keras_pred_{}{}_BM{}.csv'.format(model_name, special_str,
                                                                                         str(batch_method)))
df.to_csv(prediction_filename, index=False)
with open(prediction_filename.replace('.csv', '.pkl'), 'wb') as f:
    pickle.dump((test_filenames, ytest), f)
