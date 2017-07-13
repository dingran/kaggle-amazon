import pandas as pd
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
code_dir = dir_path
data_dir = os.path.join(dir_path, '../input')

train_label = pd.read_csv(os.path.join(data_dir, 'train_v2.csv'))
labels_str = 'agriculture, artisinal_mine, bare_ground, blooming, blow_down, clear, cloudy, conventional_mine, cultivation, habitation, haze, partly_cloudy, primary, road, selective_logging, slash_burn, water'
labels_list = labels_str.split(', ')
label_to_idx = {x: labels_list.index(x) for x in labels_list}


def tags_to_vec(tags):
    tags_list = tags.split(' ')
    vec = np.zeros(17)
    for t in tags_list:
        vec[label_to_idx[t]] = 1
    return vec


def map_predictions(predictions, labels_list=labels_list, thresholds=np.ones(17) * 0.2):
    predictions_labels = []
    for prediction in predictions:
        labels = [labels_list[i] for i, value in enumerate(prediction) if value > thresholds[i]]
        predictions_labels.append(labels)
    return predictions_labels


train_label['y'] = train_label.tags.apply(tags_to_vec)
