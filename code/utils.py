import gc
import numpy as np
from tqdm import tqdm
import cv2
import tensorflow as tf
import time
import os
from sklearn.utils import shuffle
import inspect
import pickle
import matplotlib.pyplot as plt
import hashlib

DEFAULT_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

from sklearn.metrics import fbeta_score

N_CLASSES = 17


def generate_data(X_train, y_train, X_valid, y_valid, X_test, y_test,
                  flip=True, augment=True, aug_factor=1.0,
                  distort=True, resize=True, rotate=True, shift=True, blursharpen=True,
                  boost=False, target_y=[26, 21],
                  use_grayscale=True, keep_original=False):
    if flip:
        X_train, y_train = flip_extend(X_train, y_train)
        X_valid, y_valid = flip_extend(X_valid, y_valid)

    if augment:
        setting = dict(distort=distort,
                       resize=resize,
                       rotate=rotate,
                       shift=shift,
                       blursharpen=blursharpen)
        boost = boost

        if boost:
            target_y = target_y
            N_copy = 10
            X_train_boost, y_train_boost = augment_data(X_train, y_train, N_copy=N_copy, target_y=target_y, **setting)
            X_valid_boost, y_valid_boost = augment_data(X_valid, y_valid, N_copy=N_copy, target_y=target_y, **setting)

        X_train, y_train = augment_data(X_train, y_train, **setting, factor=aug_factor)
        X_valid, y_valid = augment_data(X_valid, y_valid, **setting, factor=aug_factor)

        if boost:
            X_train = np.concatenate([X_train, X_train_boost], axis=0)
            y_train = np.concatenate([y_train, y_train_boost], axis=0)
            X_valid = np.concatenate([X_valid, X_valid_boost], axis=0)
            y_valid = np.concatenate([y_valid, y_valid_boost], axis=0)

    print(X_train.shape, y_train.shape)
    print(X_valid.shape, y_valid.shape)

    # color conversion, hist equalization and normalization
    X_train = preprocess_data(X_train, use_grayscale=use_grayscale, keep_original=keep_original)
    X_valid = preprocess_data(X_valid, use_grayscale=use_grayscale, keep_original=keep_original)
    X_test = preprocess_data(X_test, use_grayscale=use_grayscale, keep_original=keep_original)

    return (X_train, y_train, X_valid, y_valid, X_test, y_test)


def preprocess_single_image(xx, use_grayscale=True, keep_original=False, normed=True):
    if use_grayscale:
        color_method = cv2.COLOR_BGR2GRAY
    else:
        color_method = cv2.COLOR_BGR2YUV

    res = cv2.cvtColor(xx, color_method)
    if len(res.shape) < 3:
        res = np.expand_dims(res, axis=2)

    if keep_original:
        res = np.concatenate([res, xx], axis=2)

    # if equalize_all:
    #     for i in range(res.shape[-1]):
    #         res[:, :, i] = clahe.apply(res[:, :, i])
    # else:  # only sharpening channel 0, assuming this is a grayscale channel!!!!!
    #     res[:, :, 0] = clahe.apply(res[:, :, 0])

    if normed:
        res = res.astype(float)
        for i in range(res.shape[-1]):  # normalize to 0 mean, and 1 stdev
            res[:, :, i] = (res[:, :, i] - res[:, :, i].mean()) / res[:, :, i].std()

    return res


def preprocess_data(X, use_grayscale=True, keep_original=False, normed=True):
    # assert X.shape[1] == X.shape[2] == 32

    print('Input shapes: ', X.shape)
    X_out = []
    clahe = DEFAULT_CLAHE

    if use_grayscale:
        color_method = cv2.COLOR_BGR2GRAY
        print('Using gray')
    else:
        color_method = cv2.COLOR_BGR2YUV
        print('Using YUV')

    for xx in tqdm(X):
        res = preprocess_single_image(xx, use_grayscale, keep_original, normed)
        # res = cv2.cvtColor(xx, color_method)
        # if len(res.shape) < 3:
        #     res = np.expand_dims(res, axis=2)
        #
        # if keep_original:
        #     res = np.concatenate([res, xx], axis=2)
        #
        # # if equalize_all:
        # #     for i in range(res.shape[-1]):
        # #         res[:, :, i] = clahe.apply(res[:, :, i])
        # # else:  # only sharpening channel 0, assuming this is a grayscale channel!!!!!
        # #     res[:, :, 0] = clahe.apply(res[:, :, 0])
        #
        # if normed:
        #     res = res.astype(float)
        #     for i in range(res.shape[-1]):  # normalize to 0 mean, and 1 stdev
        #         res[:, :, i] = (res[:, :, i] - res[:, :, i].mean()) / res[:, :, i].std()

        X_out.append(res)

    X_out = np.asarray(X_out)
    print('Output shapes: ', X_out.shape)
    return X_out


def augment_data(X, y, distort=True, resize=True, rotate=True, shift=True, blursharpen=True,
                 N_copy=1, target_y=None, factor=1.0):
    print('========= augment_data() arguments: =========')
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    for i in args[2:]:
        print("{} = {}".format(i, values[i]))
    print('=============================================')

    print('Input shapes: ', X.shape, y.shape)
    X_out = []
    Y_out = []

    for xx, yy in zip(tqdm(X), y):

        if target_y is not None:
            if yy not in target_y:
                continue

        if target_y is None:
            # the original
            X_out.append(xx)
            Y_out.append(yy)

        for i in range(N_copy):
            if distort:
                for d in [3, 5]:
                    X_out.append(distort_img(xx, d_limit=d * factor))
                    Y_out.append(yy)

            if resize:
                for s in np.concatenate([[0.9, 1.1], np.random.uniform(0.8, 1.2, 2)]):
                    X_out.append(resize_img(xx, scale=s * factor))
                    Y_out.append(yy)
            if rotate:
                for r in np.concatenate([[-15, 15], np.random.uniform(-20, 20, 2)]):
                    X_out.append(rotate_img(xx, angle=r * factor))
                    Y_out.append(yy)

            if shift:
                for dxdy in np.random.uniform(-4, 4, (4, 2)):
                    X_out.append(shift_img(xx, dx=dxdy[0] * factor, dy=dxdy[1] * factor))
                    Y_out.append(yy)

            if blursharpen:
                b, s = blur_and_sharpen_img(xx, factor=factor)
                X_out.append(b)
                Y_out.append(yy)
                X_out.append(s)
                Y_out.append(yy)

    X_out = np.asarray(X_out)
    Y_out = np.asarray(Y_out)
    print('Output shapes: ', X_out.shape, Y_out.shape)
    return X_out, Y_out


params_orig_lenet = dict(conv1_k=5, conv1_d=6, conv1_p=0.95,
                         conv2_k=5, conv2_d=16, conv2_p=0.95,
                         fc3_size=120, fc3_p=0.5,
                         fc4_size=84, fc4_p=0.5,
                         num_classes=N_CLASSES, model_name='lenet', name='orig_lenet')

params_big_lenet = dict(conv1_k=5, conv1_d=6 * 4, conv1_p=0.8,
                        conv2_k=5, conv2_d=16 * 4, conv2_p=0.8,
                        fc3_size=120 * 4, fc3_p=0.5,
                        fc4_size=84 * 3, fc4_p=0.5,
                        num_classes=N_CLASSES, model_name='lenet', name='big_lenet')

params_huge_lenet = dict(conv1_k=5, conv1_d=6 * 8, conv1_p=0.8,
                         conv2_k=5, conv2_d=16 * 8, conv2_p=0.8,
                         fc3_size=120 * 8, fc3_p=0.5,
                         fc4_size=84 * 6, fc4_p=0.5,
                         num_classes=N_CLASSES, model_name='lenet', name='huge_lenet')


def lenet(x, params, is_training):
    print(params)
    do_batch_norm = False
    if 'batch_norm' in params.keys():
        if params['batch_norm']:
            do_batch_norm = True

    with tf.variable_scope('conv1'):
        conv1 = conv_relu(x, kernel_size=params['conv1_k'], depth=params['conv1_d'], is_training=is_training,
                          BN=do_batch_norm)
    with tf.variable_scope('pool1'):
        pool1 = pool(conv1, size=2)
        pool1 = tf.cond(is_training, lambda: tf.nn.dropout(pool1, keep_prob=params['conv1_p']), lambda: pool1)
    with tf.variable_scope('conv2'):
        conv2 = conv_relu(pool1, kernel_size=params['conv2_k'], depth=params['conv2_d'], is_training=is_training,
                          BN=do_batch_norm)
    with tf.variable_scope('pool2'):
        pool2 = pool(conv2, size=2)
        pool2 = tf.cond(is_training, lambda: tf.nn.dropout(pool2, keep_prob=params['conv2_p']), lambda: pool2)

    shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])
    print('lenet pool2 reshaped size: ', pool2.get_shape().as_list())

    with tf.variable_scope('fc3'):
        fc3 = fully_connected_relu(pool2, size=params['fc3_size'], is_training=is_training, BN=do_batch_norm)
        fc3 = tf.cond(is_training, lambda: tf.nn.dropout(fc3, keep_prob=params['fc3_p']), lambda: fc3)

    with tf.variable_scope('fc4'):
        fc4 = fully_connected_relu(fc3, size=params['fc4_size'], is_training=is_training, BN=do_batch_norm)
        fc4 = tf.cond(is_training, lambda: tf.nn.dropout(fc4, keep_prob=params['fc4_p']), lambda: fc4)

    with tf.variable_scope('out'):
        logits = fully_connected(fc4, size=params['num_classes'], is_training=is_training)

    return logits


params_sermanet_v2 = dict(conv1_k=5, conv1_d=32, conv1_p=0.9,
                          conv2_k=5, conv2_d=64, conv2_p=0.8,
                          conv3_k=5, conv3_d=128, conv3_p=0.7,
                          fc4_size=1024, fc4_p=0.5,
                          num_classes=N_CLASSES, model_name='sermanet_v2', name='standard')

params_sermanet_v2_big = dict(conv1_k=5, conv1_d=32 * 2, conv1_p=0.9,
                              conv2_k=5, conv2_d=64 * 2, conv2_p=0.8,
                              conv3_k=5, conv3_d=128 * 2, conv3_p=0.7,
                              fc4_size=1024 * 2, fc4_p=0.5,
                              num_classes=N_CLASSES, model_name='sermanet_v2', name='big')


def sermanet_v2(x, params, is_training):
    print(params)
    do_batch_norm = False
    if 'batch_norm' in params.keys():
        if params['batch_norm']:
            do_batch_norm = True

    with tf.variable_scope('conv1'):
        conv1 = conv_relu(x, kernel_size=params['conv1_k'], depth=params['conv1_d'], is_training=is_training,
                          BN=do_batch_norm)
    with tf.variable_scope('pool1'):
        pool1 = pool(conv1, size=2)
        pool1 = tf.cond(is_training, lambda: tf.nn.dropout(pool1, keep_prob=params['conv1_p']), lambda: pool1)
    with tf.variable_scope('conv2'):
        conv2 = conv_relu(pool1, kernel_size=params['conv2_k'], depth=params['conv2_d'], is_training=is_training,
                          BN=do_batch_norm)
    with tf.variable_scope('pool2'):
        pool2 = pool(conv2, size=2)
        pool2 = tf.cond(is_training, lambda: tf.nn.dropout(pool2, keep_prob=params['conv2_p']), lambda: pool2)
    with tf.variable_scope('conv3'):
        conv3 = conv_relu(pool2, kernel_size=params['conv3_k'], depth=params['conv3_d'], is_training=is_training,
                          BN=do_batch_norm)
    with tf.variable_scope('pool3'):
        pool3 = pool(conv3, size=2)
        pool3 = tf.cond(is_training, lambda: tf.nn.dropout(pool3, keep_prob=params['conv3_p']), lambda: pool3)

    # Fully connected

    # 1st stage output
    pool1 = pool(pool1, size=4)
    shape = pool1.get_shape().as_list()
    pool1 = tf.reshape(pool1, [-1, shape[1] * shape[2] * shape[3]])

    # 2nd stage output
    pool2 = pool(pool2, size=2)
    shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])

    # 3rd stage output
    shape = pool3.get_shape().as_list()
    pool3 = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]])

    flattened = tf.concat([pool1, pool2, pool3], 1)

    with tf.variable_scope('fc4'):
        fc4 = fully_connected_relu(flattened, size=params['fc4_size'], is_training=is_training, BN=do_batch_norm)
        fc4 = tf.cond(is_training, lambda: tf.nn.dropout(fc4, keep_prob=params['fc4_p']), lambda: fc4)
    with tf.variable_scope('out'):
        logits = fully_connected(fc4, size=params['num_classes'], is_training=is_training)
    return logits


params_sermanet = dict(conv1_k=5, conv1_d=108, conv1_p=0.9,
                       conv2_k=5, conv2_d=108, conv2_p=0.8,
                       fc4_size=100, fc4_p=0.5,
                       num_classes=N_CLASSES, model_name='sermanet', name='standard')

params_sermanet_big = dict(conv1_k=5, conv1_d=100, conv1_p=0.9,
                           conv2_k=5, conv2_d=200, conv2_p=0.8,
                           fc4_size=200, fc4_p=0.5,
                           num_classes=N_CLASSES, model_name='sermanet', name='big')


def sermanet(x, params, is_training):
    print(params)
    do_batch_norm = False
    if 'batch_norm' in params.keys():
        if params['batch_norm']:
            do_batch_norm = True

    with tf.variable_scope('conv1'):
        conv1 = conv_relu(x, kernel_size=params['conv1_k'], depth=params['conv1_d'], is_training=is_training,
                          BN=do_batch_norm)
    with tf.variable_scope('pool1'):
        pool1 = pool(conv1, size=2)
        pool1 = tf.cond(is_training, lambda: tf.nn.dropout(pool1, keep_prob=params['conv1_p']), lambda: pool1)
    with tf.variable_scope('conv2'):
        conv2 = conv_relu(pool1, kernel_size=params['conv2_k'], depth=params['conv2_d'], is_training=is_training,
                          BN=do_batch_norm)
    with tf.variable_scope('pool2'):
        pool2 = pool(conv2, size=2)
        pool2 = tf.cond(is_training, lambda: tf.nn.dropout(pool2, keep_prob=params['conv2_p']), lambda: pool2)

    # Fully connected

    # 1st stage output
    pool1 = pool(pool1, size=2)
    shape = pool1.get_shape().as_list()
    pool1 = tf.reshape(pool1, [-1, shape[1] * shape[2] * shape[3]])

    # 2nd stage output
    shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])

    flattened = tf.concat([pool1, pool2], 1)

    with tf.variable_scope('fc4'):
        fc4 = fully_connected_relu(flattened, size=params['fc4_size'], is_training=is_training, BN=do_batch_norm)
        fc4 = tf.cond(is_training, lambda: tf.nn.dropout(fc4, keep_prob=params['fc4_p']), lambda: fc4)
    with tf.variable_scope('out'):
        logits = fully_connected(fc4, size=params['num_classes'], is_training=is_training)
    return logits


def train_model(X_train, y_train, X_valid, y_valid, X_test, y_test,
                resuming=False,
                model=lenet, model_params=params_orig_lenet,
                learning_rate=0.001, max_epochs=1001, batch_size=256,
                early_stopping_enabled=True, early_stopping_patience=10,
                log_epoch=1, print_epoch=1):
    print('========= train_model() arguments: ==========')
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    for i in args[6:]:
        print("{} = {}".format(i, values[i]))
    print('=============================================')

    fn = ''
    model_name = model_params.pop('name', None)
    print(model_name)
    for k in sorted(model_params.keys()):
        if k != 'num_classes' and k != 'model_name' and k != 'batch_norm':
            fn += k + '_' + str(model_params[k]) + '_'
    data_str = ''
    if X_test.shape[-1] > 1:
        data_str = str(X_test.shape[-1])
    model_id = model_params['model_name'] + data_str + '__' + fn[:-1]

    if 'batch_norm' in model_params.keys():
        if model_params['batch_norm']:
            model_id = 'BN_' + model_id

    model_id_hash = str(hashlib.sha1(model_id.encode('utf-8')).hexdigest()[-16:])
    print(model_id)
    print(model_id_hash)
    model_dir = os.path.join(os.getcwd(), 'models', model_id_hash)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, model_id), exist_ok=True)
    print('model dir: {}'.format(model_dir))
    model_fname = os.path.join(model_dir, 'model_cpkt')
    model_fname_best_epoch = os.path.join(model_dir, 'best_epoch')
    model_train_history = os.path.join(model_dir, 'training_history.npz')

    start = time.time()

    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
        x = tf.placeholder(tf.float32, (None, X_test.shape[1], X_test.shape[2], X_test.shape[-1]))
        y = tf.placeholder(tf.float32, (None, 17))
        # one_hot_y = tf.one_hot(y, model_params['num_classes'])
        is_training = tf.placeholder(tf.bool)

        logits = model(x, params=model_params, is_training=is_training)

        predictions = tf.nn.sigmoid(logits)
        # top_k_predictions = tf.nn.top_k(predictions, top_k)
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
        loss_operation = tf.reduce_mean(cross_entropy, name='loss')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_operation = optimizer.minimize(loss_operation)

        # pred_y = tf.argmax(logits, 1, name='prediction')
        # actual_y = tf.argmax(one_hot_y, 1)
        # correct_prediction = tf.equal(pred_y, actual_y)
        # accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print(variable)
            print(shape)
            # print(len(shape))
            variable_parametes = 1
            for dim in shape:
                # print(dim)
                variable_parametes *= dim.value
            # print(variable_parametes)
            total_parameters += variable_parametes
        print('total # of parameters: ', total_parameters)

        def predict(X_data):
            n_data = len(X_data)
            logits_pred = []
            for offset in tqdm(range(0, n_data, batch_size)):
                batch_x = X_data[offset:offset + batch_size]

                batch_logits = sess.run(predictions, feed_dict={x: batch_x, is_training: False})

                batch_logits = np.array(batch_logits)
                # print(batch_logits.shape)
                logits_pred.append(batch_logits)

            logits_pred = np.concatenate(logits_pred, axis=0)
            y_pred = np.array(logits_pred) > 0.2
            y_pred = y_pred.astype(int)

            return y_pred, logits_pred

        def evaluate(X_data, y_data):
            n_data = len(X_data)
            logits_pred = []
            y_actual = []
            loss_batch = np.array([])
            batch_sizes = np.array([])
            for offset in range(0, n_data, batch_size):
                batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
                batch_sizes = np.append(batch_sizes, batch_y.shape[0])

                batch_logits, loss, ya_ = sess.run([predictions, loss_operation, y],
                                                   feed_dict={x: batch_x, y: batch_y, is_training: False})

                y_actual.append(ya_)
                batch_logits = np.array(batch_logits)
                # print(batch_logits.shape)
                logits_pred.append(batch_logits)
                loss_batch = np.append(loss_batch, loss)

            y_actual = np.concatenate(y_actual, axis=0)
            logits_pred = np.concatenate(logits_pred, axis=0)
            y_pred = np.array(logits_pred) > 0.2
            y_pred = y_pred.astype(int)

            final_loss = np.average(loss_batch, weights=batch_sizes)

            final_f2beta = fbeta_score(y_actual, y_pred, beta=2, average='samples')

            return final_f2beta, final_loss

        # If we chose to keep training previously trained model, restore session.
        if resuming:
            try:
                tf.train.Saver().restore(sess, model_fname)
                print('Restored session from {}'.format(model_fname))
            except Exception as e:
                print("Failed restoring previously trained model: file does not exist.")
                print("Trying to restore from best epoch from previously training session.")
                try:
                    tf.train.Saver().restore(sess, model_fname_best_epoch)
                    print('Restored session from {}'.format(model_fname_best_epoch))
                except Exception as e:
                    print("Failed to restore, will train from scratch now.")

                    # print([v.op.name for v in tf.all_variables()])
                    # print([n.name for n in tf.get_default_graph().as_graph_def().node])

        saver = tf.train.Saver()
        early_stopping = EarlyStopping(tf.train.Saver(), sess, patience=early_stopping_patience, minimize=True,
                                       restore_path=model_fname_best_epoch)

        train_loss_history = np.empty([0], dtype=np.float32)
        train_accuracy_history = np.empty([0], dtype=np.float32)
        valid_loss_history = np.empty([0], dtype=np.float32)
        valid_accuracy_history = np.empty([0], dtype=np.float32)
        if max_epochs > 0:
            print("================= TRAINING ==================")
        else:
            print("================== TESTING ==================")
        print(" Timestamp: " + get_time_hhmmss())

        for epoch in range(max_epochs):
            gc.collect() 
            X_train, y_train = shuffle(X_train, y_train)

            for offset in tqdm(range(0, X_train.shape[0], batch_size)):
                end = offset + batch_size
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, is_training: True})

            # If another significant epoch ended, we log our losses.
            if epoch % log_epoch == 0:
                train_accuracy, train_loss = evaluate(X_train, y_train)
                valid_accuracy, valid_loss = evaluate(X_valid, y_valid)

                if epoch % print_epoch == 0:
                    print("-------------- EPOCH %4d/%d --------------" % (epoch, max_epochs))
                    print("     Train loss: %.8f, f2beta: %.2f%%" % (train_loss, 100 * train_accuracy))
                    print("Validation loss: %.8f, f2beta: %.2f%%" % (valid_loss, 100 * valid_accuracy))
                    print("      Best loss: %.8f at epoch %d" % (
                        early_stopping.best_monitored_value, early_stopping.best_monitored_epoch))
                    print("   Elapsed time: " + get_time_hhmmss(start))
                    print("      Timestamp: " + get_time_hhmmss())
            else:
                valid_loss = 0.
                valid_accuracy = 0.
                train_loss = 0.
                train_accuracy = 0.

            valid_loss_history = np.append(valid_loss_history, [valid_loss])
            valid_accuracy_history = np.append(valid_accuracy_history, [valid_accuracy])
            train_loss_history = np.append(train_loss_history, [train_loss])
            train_accuracy_history = np.append(train_accuracy_history, [train_accuracy])

            if early_stopping_enabled:
                # Get validation data predictions and log validation loss:
                if valid_loss == 0:
                    _, valid_loss = evaluate(X_valid, y_valid)
                if early_stopping(valid_loss, epoch):
                    print("Early stopping.\nBest monitored loss was {:.8f} at epoch {}.".format(
                        early_stopping.best_monitored_value, early_stopping.best_monitored_epoch
                    ))
                    break

        if max_epochs == -2:  # make prediction only
            return predict(X_test)

        # Evaluate on test dataset.
        valid_accuracy, valid_loss = evaluate(X_valid, y_valid)
        test_accuracy, test_loss = evaluate(X_test, y_test)
        print("=============================================")
        print(" Valid loss: %.8f, f2beta= %.2f%%)" % (valid_loss, 100 * valid_accuracy))
        print(" Test loss: %.8f, f2beta= %.2f%%)" % (test_loss, 100 * test_accuracy))
        print(" Total time: " + get_time_hhmmss(start))
        print("  Timestamp: " + get_time_hhmmss())

        # Save model weights for future use.
        saved_model_path = saver.save(sess, model_fname)
        print("Model file: " + saved_model_path)
        np.savez(model_train_history, train_loss_history=train_loss_history,
                 train_accuracy_history=train_accuracy_history, valid_loss_history=valid_loss_history,
                 valid_accuracy_history=valid_accuracy_history)
        print("Train history file: " + model_train_history)

    result_dict = dict(test_accuracy=test_accuracy, test_loss=test_loss,
                       valid_accuracy=valid_accuracy, valid_loss=valid_loss)
    return result_dict


class EarlyStopping(object):
    """
    Provides early stopping functionality. Keeps track of model accuracy,
    and if it doesn't improve over time restores last best performing
    parameters.
    """

    def __init__(self, saver, session, patience=100, minimize=True, restore_path=None):
        """
        Initialises a `EarlyStopping` isntance.

        Parameters
        ----------
        saver     :
                    TensorFlow Saver object to be used for saving and restoring model.
        session   :
                    TensorFlow Session object containing graph where model is restored.
        patience  :
                    Early stopping patience. This is the number of epochs we wait for
                    accuracy to start improving again before stopping and restoring
                    previous best performing parameters.

        Returns
        -------
        New instance.
        """
        self.minimize = minimize
        self.patience = patience
        self.saver = saver
        self.session = session
        self.best_monitored_value = np.inf if minimize else 0.
        self.best_monitored_epoch = 0
        self.restore_path = restore_path

    def __call__(self, value, epoch):
        """
        Checks if we need to stop and restores the last well performing values if we do.

        Parameters
        ----------
        value     :
                    Last epoch monitored value.
        epoch     :
                    Last epoch number.

        Returns
        -------
        `True` if we waited enough and it's time to stop and we restored the
        best performing weights, or `False` otherwise.
        """
        if (self.minimize and value < self.best_monitored_value) or (
                    not self.minimize and value > self.best_monitored_value):
            self.best_monitored_value = value
            self.best_monitored_epoch = epoch
            self.saver.save(self.session, self.restore_path)
        elif self.best_monitored_epoch + self.patience < epoch:
            if self.restore_path is not None:
                self.saver.restore(self.session, self.restore_path)
            else:
                print("ERROR: Failed to restore session")
            return True

        return False


def fully_connected(input_x, size, is_training, BN=False):
    """
    Performs a single fully connected layer pass, e.g. returns `input * weights + bias`.
    """
    weights = tf.get_variable('weights',
                              shape=[input_x.get_shape()[1], size],
                              initializer=tf.contrib.layers.xavier_initializer()
                              )
    biases = tf.get_variable('biases',
                             shape=[size],
                             initializer=tf.constant_initializer(0.0)
                             )
    out = tf.matmul(input_x, weights) + biases
    if BN:
        out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=is_training, scope='bn')
    return out


def fully_connected_relu(input_x, size, is_training, BN=False):
    return tf.nn.relu(fully_connected(input_x, size, is_training, BN=BN))


def conv_relu(input_x, kernel_size, depth, is_training, BN=False):
    """
    Performs a single convolution layer pass.
    """
    weights = tf.get_variable('weights',
                              shape=[kernel_size, kernel_size, input_x.get_shape()[3], depth],
                              initializer=tf.contrib.layers.xavier_initializer()
                              )
    biases = tf.get_variable('biases',
                             shape=[depth],
                             initializer=tf.constant_initializer(0.0)
                             )
    conv = tf.nn.conv2d(input_x, weights, strides=[1, 1, 1, 1], padding='SAME')
    if BN:
        conv = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=is_training, scope='bn')
    return tf.nn.relu(conv + biases)


def pool(input_x, size):
    """
    Performs a max pooling layer pass.
    """
    return tf.nn.max_pool(input_x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')


def distort_img(input_img, d_limit=4):
    """
    Apply warpPerspective transformation on image, with 4 key points, randomly generated around the corners
    with uniform distribution with a range of [-d_limit, d_limit]
    :param input_img:
    :param d_limit:
    :return:
    """
    if d_limit == 0:
        return input_img
    rows, cols, ch = input_img.shape
    pts2 = np.float32([[0, 0], [rows - 1, 0], [0, cols - 1], [rows - 1, cols - 1]])
    pts1 = np.float32(pts2 + np.random.uniform(-d_limit, d_limit, pts2.shape))
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(input_img, M, (cols, rows), borderMode=1)
    return dst


def resize_img(input_img, scale=1.1):
    """
    Function to scale image content while keeping the overall image size, padding is done with border replication
    Scale > 1 means making content bigger
    :param input_img: X * Y * ch
    :param scale: positive real number
    :return: scaled image
    """
    if scale == 1.0:
        return input_img
    rows, cols, ch = input_img.shape
    d = rows * (scale - 1)  # overall image size change from rows, cols, to rows - 2d, cols - 2d
    pts1 = np.float32([[d, d], [rows - 1 - d, d], [d, cols - 1 - d], [rows - 1 - d, cols - 1 - d]])
    pts2 = np.float32([[0, 0], [rows - 1, 0], [0, cols - 1], [rows - 1, cols - 1]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(input_img, M, (cols, rows), borderMode=1)
    return dst


def rotate_img(input_img, angle=15):
    if angle == 0:
        return input_img
    rows, cols, ch = input_img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(input_img, M, (cols, rows), borderMode=1)
    return dst


def shift_img(input_img, dx=2, dy=2):
    if dx == 0 and dy == 0:
        return input_img
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    dst = cv2.warpAffine(input_img, M, (input_img.shape[0], input_img.shape[1]), borderMode=1)
    return dst


def blur_and_sharpen_img(input_img, kernel=(3, 3), ratio=0.7, factor=1.0):
    blur = cv2.GaussianBlur(input_img, kernel, 0)
    sharp = cv2.addWeighted(input_img, 1.0 + ratio * factor, blur, -ratio * factor, 0)
    return blur, sharp


def get_time_hhmmss(start=None):
    """
    Calculates time since `start` and formats as a string.
    """
    if start is None:
        return time.strftime("%Y/%m/%d %H:%M:%S")
    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)
    return time_str
