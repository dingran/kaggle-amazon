import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from tqdm import tqdm

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

dir_path = os.path.dirname(os.path.realpath(__file__))
code_dir = dir_path
data_dir = os.path.join(dir_path, '../input')
print(data_dir)

file_type = 'jpg'
print('file type: ', file_type)

xtrain_files = glob.glob(os.path.join(data_dir, 'xtrain-{}-chunk{}.npy'.format(file_type, '*')))
ytrain_files = glob.glob(os.path.join(data_dir, 'ytrain-{}-chunk{}.npy'.format(file_type, '*')))

print(len(xtrain_files))
print(len(ytrain_files))

xtrain_npy = None
ytrain_npy = None

N_chunck = len(xtrain_files)
print('N_chunck', N_chunck)

for i in range(1, N_chunck+1):
    xtrain_fname = 'xtrain-{}-chunk{}.npy'.format(file_type, str(i))
    ytrain_fname = 'ytrain-{}-chunk{}.npy'.format(file_type, str(i))
    print(xtrain_fname)
    print(ytrain_fname)

    if xtrain_npy is None:
        xtrain_npy = np.load(xtrain_fname)
        ytrain_npy = np.load(ytrain_files)
    else:
        xtrain_npy = np.concatenate((xtrain_npy, np.load(xtrain_fname)), axis=0)
        ytrain_npy = np.concatenate((ytrain_npy, np.load(ytrain_fname)), axis=0)
        print(xtrain_npy.shape)
        print(ytrain_npy.shape)

print(xtrain_npy.shape)
print(ytrain_npy.shape)


def image_normalization(xdata):
    pass


xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain_npy, ytrain_npy, test_size=0.2)
print(xtrain.shape, xvalid.shape, ytrain.shape, yvalid.shape)

xtrain = xtrain.astype('float32')
xvalid = xvalid.astype('float32')
xtrain /= 255
xvalid /= 255

batch_size = 32
num_classes = 17
epochs = 200
data_augmentation = True

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(17, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='binary_crossentropy')

import scipy.optimize as so

# defining a set of callbacks
class f2beta(Callback):
    def __init__(self, xval, yval):
        self.xval = xval
        self.yval = yval
        self.maps = []

    def eval_map(self):
        x_val = self.xval
        y_true = self.yval
        y_pred = self.model.predict(x_val)

        y_pred = (y_pred > 0.2).astpe(int)
        score = fbeta_score(y_true, y_pred, beta=2, average='samples')

        return score

    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_map()
        print("f2beta for epoch %d is %f" % (epoch, score))
        self.maps.append(score)


beta_score = f2beta(xvalid, yvalid)

checkpoint = ModelCheckpoint(os.path.join(code_dir, "model-{val_loss:.5f}.hdf5"),
                             monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(xtrain, ytrain,
              batch_size=batch_size,
              epochs=epochs, callbacks=[checkpoint, beta_score, earlystop],
              validation_data=(xvalid, yvalid),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=359,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.3,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.3,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(xtrain)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(xtrain, ytrain, batch_size=batch_size),
                        steps_per_epoch=xtrain.shape[0] // batch_size,
                        epochs=epochs, callbacks=[checkpoint, beta_score, earlystop],
                        validation_data=(xvalid, yvalid))