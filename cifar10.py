import pickle
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load(path):
    # label
    label_name = unpickle(path+'/batches.meta')[b'label_names']
    # print(label_name)

    # data train
    train_data = []
    train_label = []
    for i in range(1, 6):
        file = path + '/data_batch_' + str(i)
        data_batch = unpickle(file)
        train_data.append(data_batch[b'data'].astype(float))
        train_label.append(np.array(data_batch[b'labels']))
    train_data = np.concatenate(train_data)
    train_label = np.concatenate(train_label)
    # print(train_data.shape)
    # print(train_label.shape)

    # data test
    test_batch = unpickle(path+'/test_batch')
    test_data = test_batch[b'data'].astype(float)
    test_label = np.array(test_batch[b'labels'])
    # print(test_data.shape)
    # print(test_label.shape)

    # normalized
    tmp = np.vstack((train_data, test_data))
    # print(tmp.shape)
    max_v = np.max(tmp, axis=0)
    min_v = np.min(tmp, axis=0)
    dif = max_v - min_v
    train_data /= dif
    test_data /= dif
    # print(dif.shape)
    # print(train_data.shape)
    # print(test_data.shape)
    # print(train_data[0:])

    return label_name, train_data, train_label, test_data, test_label