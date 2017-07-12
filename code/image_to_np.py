import os
import pandas as pd
import numpy as np
import glob
# import cv2
from skimage import io
from skimage.transform import resize

from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, '../input')
print(data_dir)

train_label = pd.read_csv(os.path.join(data_dir, 'train_v2.csv'))
labels_str = 'agriculture, artisinal_mine, bare_ground, blooming, blow_down, clear, cloudy, conventional_mine, cultivation, habitation, haze, partly_cloudy, primary, road, selective_logging, slash_burn, water'
labels = labels_str.split(', ')
label_map = {x: labels.index(x) for x in labels}


def tags_to_vec(tags):
    tags_list = tags.split(' ')
    vec = np.zeros(17)
    for t in tags_list:
        vec[label_map[t]] = 1
    return vec


def map_predictions(predictions, label_map, thresholds=0.2):
    assert isinstance(thresholds, float) or isinstance(thresholds, np.ndarray)

    if isinstance(thresholds, float):
        thresholds = np.ones(17) * thresholds

    predictions_labels = []
    for prediction in predictions:
        labels = [label_map[i] for i, value in enumerate(prediction) if value > thresholds[i]]
        predictions_labels.append(labels)

    return predictions_labels


def main():
    train_label['y'] = train_label.tags.apply(tags_to_vec)

    file_type = 'jpg'
    print('file type: ', file_type)

    do_resize = True
    process_train = True
    process_test = True

    if process_train:
        chunck_id = 0
        count = 0
        X_chunk = []
        y_chunk = []
        chucksize = 1000
        for idx, row in tqdm(train_label.iterrows(), total=train_label.shape[0]):
            image = io.imread(
                os.path.join(data_dir, 'train-{}'.format(file_type),
                             '{}.{}'.format(row['image_name'], file_type)))
            # image = image[:,:,::-1]
            # plt.imshow(image)
            if do_resize:
                image = resize(image, (299, 299))  # for InceptionV3

            X_chunk.append(image)
            y_chunk.append(row['y'])
            count += 1
            if count % chucksize == 0 or count == train_label.shape[0]:
                chunck_id += 1
                print('chunk', chunck_id)
                X_chunk = np.stack(X_chunk, axis=0)
                y_chunk = np.stack(y_chunk, axis=0)
                print(X_chunk.shape, y_chunk.shape)
                np.save(os.path.join(data_dir, 'xtrain-{}-chunk{}'.format(file_type, chunck_id)), X_chunk)
                np.save(os.path.join(data_dir, 'ytrain-{}-chunk{}'.format(file_type, chunck_id)), y_chunk)
                X_chunk = []
                y_chunk = []

    if process_test:
        test_file_list = glob.glob(os.path.join(data_dir, 'test-jpg/*.jpg'))
        print('n test files: {}'.format(test_file_list))
        chunck_id = 0
        count = 0
        X_chunk = []
        filename_chunk = []
        chucksize = 1000

        for t in tqdm(test_file_list):
            image = io.imread(t)
            if do_resize:
                image = resize(image, (299, 299))  # for InceptionV3
            X_chunk.append(image)

            filename = os.path.basename(t).replace('.jpg', '')
            filename_chunk.append(filename)
            count += 1
            if count % chucksize == 0 or count == len(test_file_list):
                chunck_id += 1
                print('chunk', chunck_id)
                X_chunk = np.stack(X_chunk, axis=0)
                filename_chunk = np.array(filename_chunk)
                print(X_chunk.shape, filename_chunk.shape)
                np.save(os.path.join(data_dir, 'xtest-{}-chunk{}'.format(file_type, chunck_id)), X_chunk)
                np.save(os.path.join(data_dir, 'testfilename-{}-chunk{}'.format(file_type, chunck_id)),
                        filename_chunk)
                X_chunk = []
                filename_chunk = []


if __name__ == '__main__':
    main()
