import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from collections import Counter

def get_model_list(dirname, key, mode='max'):
    if os.path.exists(dirname) is False:
        return None
    # print(dirname)
    models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if models is None:
        return None
    if mode == 'max':
        models = sorted(models, key=lambda x : int(x[-19:-16]) + int(x[-15:-12]))
    else:
        models = sorted(models, key=lambda x : int(x[-11:-3]))
    last_model_name = models[-1]
    return last_model_name, int(last_model_name[-11:-3])

def get_stat_bar(key_train, key_val, value_train, value_val, name):
    plt.figure()
    # key = np.array(key_train)
    # value = np.array(value_train)
    # plt.bar(key, value, width=0.5, color='b')

    key = np.array(key_val)
    value = np.array(value_val)
    plt.title(name)
    plt.bar(key, value, width=0.5, color='r')

    plt.savefig('./fig/bar_label_{}.png'.format(name))

def get_stat_pred(pred, label, name):
    # self.v_train = Counter(self.v_labels_train)
    # self.a_train = Counter(self.a_labels_train)
    # self.v_val = Counter(self.v_labels_val)
    # self.a_val = Counter(self.a_labels_val)
    pred_dic = Counter([int(p*10)//10 for p in pred])
    label_dic = Counter([int(p*10)//10 + 0.5 for p in label])
    print(pred_dic)
    print(label_dic)
    pred_key = list(pred_dic.keys())
    pred_val = list(pred_dic.values())
    label_key = list(label_dic.keys())
    label_val = list(label_dic.values())
    plt.figure()
    key = np.array(pred_key)
    value = np.array(pred_val)
    plt.bar(key, value, color='b')
    plt.title(name)
    key = np.array(label_key)
    value = np.array(label_val)
    plt.bar(key, value, width=0.5, color='r')

    plt.savefig('./fig/bar_pred_{}.png'.format(name))


def get_stat_scatter(point_set, values, name):
    plt.figure()
    v = []
    a = []
    col = []
    v_max = np.max(np.max(values, 0))
    v_min = np.min(np.min(values, 0))
    values = (values - v_min) / (v_max - v_min)
    # print(values)
    # exit()
    for point in point_set:
        v.append(point[0])
        a.append(point[1])
        col.append((1, 1, 1-values[point[0]+10][point[1]+10]))
    plt.scatter(v, a, color=col)

    plt.savefig('./scatter_test_{}.png'.format(name))

def get_stat_scatter_column(point_set, values, name):
    plt.figure()
    v = []
    a = []
    col = []
    v_max = np.max(values, 0)
    v_min = np.min(values, 0)
    values = (values - v_min) / (v_max - v_min)
    # print(values)
    # exit()
    for point in point_set:
        v.append(point[0])
        a.append(point[1])
        col.append((1, 1, 1-values[point[0]+10][point[1]+10]))
    plt.scatter(v, a, color=col)

    plt.savefig('./scatter_column_{}.png'.format(name))