import pickle
import my_callbacks
import numpy as np
from keras.utils import to_categorical
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

def calculate_amplitudes(set, str):
	N_MEAN_AMP = 2 #number of samples to measure amplitude
	magnitudes = set[str]
	sort_magnitudes = np.sort(magnitudes, axis=1)
	A_max = np.mean(sort_magnitudes[:, -N_MEAN_AMP:], axis=1)
	A_min = np.mean(sort_magnitudes[:, :N_MEAN_AMP], axis=1)
	amplitudes = np.abs(A_max - A_min) / 2
	set['amplitude'] = amplitudes

def change_classes(targets):
	#print(targets)
	target_keys = np.unique(targets)
	#print(target_keys)
	target_keys_idxs = np.argsort(np.unique(targets))
	targets = target_keys_idxs[np.searchsorted(target_keys, targets, sorter=target_keys_idxs)]

	return targets


def read_data_irr(file, str_mag, str_time):

    with open(file, 'rb') as f: data = pickle.load(f)

    #print(data[0].keys())
    calculate_amplitudes(data[0], str_mag)

    mgt = np.asarray(data[0][str_mag])
    t = np.asarray(data[0][str_time])
    X_train = np.stack((mgt, t), axis=-1)
    X_train =  X_train.reshape(X_train.shape[0], X_train.shape[1], 1, X_train.shape[2])

    y_train = np.asarray(data[0]['class'])

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    y_train = change_classes(y_train)
    y_train = to_categorical(y_train)
    A_train = data[0]['amplitude']

    calculate_amplitudes(data[1], str_mag)
    mgt = np.asarray(data[1][str_mag])
    t = np.asarray(data[1]['time'])
    X_val = np.stack((mgt, t), axis=-1)
    X_val =  X_val.reshape(X_val.shape[0], X_val.shape[1], 1, X_val.shape[2])
    y_val = np.asarray(data[1]['class'])
    y_val = change_classes(y_val)
    y_val = to_categorical(y_val)
    X_val, y_val = shuffle(X_val, y_val, random_state=42)
    A_val = data[1]['amplitude']

    calculate_amplitudes(data[2], str_mag)
    mgt = np.asarray(data[2][str_mag])
    t = np.asarray(data[2][str_time])
    X_test = np.stack((mgt, t), axis=-1)
    X_test =  X_test.reshape(X_test.shape[0], X_test.shape[1], 1, X_test.shape[2])
    y_test = np.asarray(data[2]['class'])
    y_test = change_classes(y_test)
    y_test = to_categorical(y_test)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)
    A_test = data[2]['amplitude']

    return X_train, y_train, A_train, X_val, y_val, A_val, X_test, y_test, A_test

def check_dir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

folder = 'starlight_amp_noisy_irregular_all'
date = '2003'

check_dir('TSTR_'+ date)
check_dir('TSTR_'+ date +'/train/')
check_dir('TSTR_'+ date +'/train/')
check_dir('TSTR_'+ date +'/train/'+ folder)
check_dir('TSTR_'+ date +'/test/')
check_dir('TSTR_'+ date +'/test/'+ folder)

dataset_syn = 'starlight_amp_noisy_irregular_all_generated'
dataset_real = 'starlight_noisy_irregular_all_classes'

X_train, y_train, A_train, X_val, y_val, A_val, X_test, y_test, A_test  = read_data_irr('TSTR_data/generated/'+ folder +'/' + dataset_syn + '.pkl', 'generated_magnitude', 'time')
X_trainR, y_trainR, A_trainR, X_valR, y_valR, A_valR, X_testR, y_testR, A_testR  = read_data_irr('TSTR_data/datasets_original/REAL/'+ dataset_real +'.pkl', 'original_magnitude', 'time')


if os.path.isfile('TSTR_'+ date +'/train/'+ folder +'/mse_train.npy'):
	mse = np.load('TSTR_'+ date +'/train/'+ folder +'/mse_train.npy')
else:
	mse = mean_squared_error(A_trainR, A_train[:len(A_trainR)])
	np.save('TSTR_'+ date +'/train/'+ folder +'/mse_train.npy', mse)

print('\nMSE train: ',mse)

if os.path.isfile('TSTR_'+ date +'/test/'+ folder +'/mse_test.npy'):
	mse = np.load('TSTR_'+ date +'/test/'+ folder +'/mse_test.npy')
else:
	mse = mean_squared_error(A_testR, A_test[:len(A_testR)])
	np.save('TSTR_'+ date +'/test/'+ folder +'/mse_test.npy', mse)

print('MSE test: ',mse)
