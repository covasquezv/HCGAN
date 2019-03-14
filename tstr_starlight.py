from model_keras import *
from keras.callbacks import History, ModelCheckpoint, EarlyStopping
import pickle
import my_callbacks
import numpy as np
from keras.utils import to_categorical
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import Model, load_model
from sklearn.metrics import roc_auc_score, accuracy_score


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def read_data(file):

	with open(file, 'rb') as f: data = pickle.load(f)

	X_train = np.asarray(data[0]['generated_magnitude'])
	#print(X_train.shape)
	X_train =  X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
	#print(X_train.shape)
	y_train = np.asarray(data[0]['class'])
	X_train, y_train = shuffle(X_train, y_train, random_state=42)
	y_train = change_classes(y_train)
	y_train = to_categorical(y_train)

	X_val = np.asarray(data[1]['generated_magnitude'])
	X_val =  X_val.reshape(X_val.shape[0], X_val.shape[1], 1, 1)
	y_val = np.asarray(data[1]['class'])
	y_val = change_classes(y_val)
	y_val = to_categorical(y_val)
	X_val, y_val = shuffle(X_val, y_val, random_state=42)

	X_test = np.asarray(data[2]['generated_magnitude'])
	X_test =  X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)
	y_test = np.asarray(data[2]['class'])
	y_test = change_classes(y_test)
	y_test = to_categorical(y_test)
	X_test, y_test = shuffle(X_test, y_test, random_state=42)

	return X_train, y_train, X_val, y_val, X_test, y_test

def read_data_original_irr(file):

	with open(file, 'rb') as f: data = pickle.load(f)

	print(data[0].keys())

	mgt = np.asarray(data[0]['original_magnitude'])
	t = np.asarray(data[0]['time'])
	X_train = np.stack((mgt, t), axis=-1)
	X_train =  X_train.reshape(X_train.shape[0], X_train.shape[1], 1, X_train.shape[2])

	y_train = np.asarray(data[0]['class'])

	X_train, y_train = shuffle(X_train, y_train, random_state=42)
	y_train = change_classes(y_train)
	y_train = to_categorical(y_train)

	mgt = np.asarray(data[1]['original_magnitude'])
	t = np.asarray(data[1]['time'])
	X_val = np.stack((mgt, t), axis=-1)
	X_val =  X_val.reshape(X_val.shape[0], X_val.shape[1], 1, X_val.shape[2])
	y_val = np.asarray(data[1]['class'])
	y_val = change_classes(y_val)
	y_val = to_categorical(y_val)
	X_val, y_val = shuffle(X_val, y_val, random_state=42)

	mgt = np.asarray(data[2]['original_magnitude'])
	t = np.asarray(data[2]['time'])
	X_test = np.stack((mgt, t), axis=-1)
	X_test =  X_test.reshape(X_test.shape[0], X_test.shape[1], 1, X_test.shape[2])
	y_test = np.asarray(data[2]['class'])
	y_test = change_classes(y_test)
	y_test = to_categorical(y_test)
	X_test, y_test = shuffle(X_test, y_test, random_state=42)

	return X_train, y_train, X_val, y_val, X_test, y_test

def read_data_generated_irr(file):

	with open(file, 'rb') as f: data = pickle.load(f)

	print(data[0].keys())

	mgt = np.asarray(data[0]['generated_magnitude'])
	t = np.asarray(data[0]['time'])
	X_train = np.stack((mgt, t), axis=-1)
	X_train =  X_train.reshape(X_train.shape[0], X_train.shape[1], 1, X_train.shape[2])
	#print(X_train.shape)
	y_train = np.asarray(data[0]['class'])
	print(np.unique(y_train))
	X_train, y_train = shuffle(X_train, y_train, random_state=42)
#	for i in y_train:
#		if i != None:
#			print(i)
	y_train = change_classes(y_train)
	y_train = to_categorical(y_train)

	mgt = np.asarray(data[1]['generated_magnitude'])
	t = np.asarray(data[1]['time'])
	X_val = np.stack((mgt, t), axis=-1)
	X_val =  X_val.reshape(X_val.shape[0], X_val.shape[1], 1, X_val.shape[2])
	y_val = np.asarray(data[1]['class'])
	y_val = change_classes(y_val)
	y_val = to_categorical(y_val)
	X_val, y_val = shuffle(X_val, y_val, random_state=42)

	mgt = np.asarray(data[2]['generated_magnitude'])
	t = np.asarray(data[2]['time'])
	X_test = np.stack((mgt, t), axis=-1)
	X_test =  X_test.reshape(X_test.shape[0], X_test.shape[1], 1, X_test.shape[2])
	y_test = np.asarray(data[2]['class'])
	y_test = change_classes(y_test)
	y_test = to_categorical(y_test)
	X_test, y_test = shuffle(X_test, y_test, random_state=42)

	return X_train, y_train, X_val, y_val, X_test, y_test


def change_classes(targets):
	#print(targets)
	target_keys = np.unique(targets)
	#print(target_keys)
	target_keys_idxs = np.argsort(np.unique(targets))
	targets = target_keys_idxs[np.searchsorted(target_keys, targets, sorter=target_keys_idxs)]

	return targets


def open_data(file):

	with open(file, 'rb') as f: data = pickle.load(f)

	print(len(data['generated_magnitude']))

	X = np.asarray(data['generated_magnitude'])
	X =  X.reshape(X.shape[0], X.shape[1], 1, 1)
	y = np.asarray(data['class'])
	X, y = shuffle(X, y, random_state=42)
	y = change_classes(y)
	y = to_categorical(y)


	return X, y


def evaluation(X_test, y_test, n_classes):
	y_pred_prob = model.predict_proba(X_test)

	n = 10
	probs = np.array_split(y_pred_prob, n)

	score = []
	mean = []
	std = []

	Y = []
	for prob in probs:
		ys = np.zeros(n_classes)#[0, 0
		for class_i in range(n_classes):
			for j in prob:
				ys[class_i] = ys[class_i] + j[class_i]

		ys[:] = [x/len(prob) for x in ys]


		Y.append(np.asarray(ys))

	ep = 1e-12
	tmp = []
	for s in range(n):
		kl = (probs[s] * np.log((probs[s] + ep)/Y[s])).sum(axis=1)
		E = np.mean(kl)
		IS = np.exp(E)
		#pdb.set_trace()
		tmp.append(IS)

	score.append(tmp)
	mean.append(np.mean(tmp))
	std.append(np.std(tmp))

	print('Inception Score:\nMean score : ', mean[-1])
	print('Std : ', std[-1])

	return score, mean, std

def check_dir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)



date = '1403'
folder = 'starlight_amp_noisy_irregular_all'

check_dir('TSTR_'+ date)
check_dir('TSTR_'+ date +'/train/')
check_dir('TSTR_'+ date +'/train/')
check_dir('TSTR_'+ date +'/train/'+ folder)
check_dir('TSTR_'+ date +'/test/')
check_dir('TSTR_'+ date +'/test/'+ folder)

if os.path.isfile('TSTR_'+ date +'/train/'+ folder +'/trainonsynthetic_model.h5'):

	print('\nTrain metrics:')

	mean = np.load('TSTR_'+ date +'/train/'+ folder +'/trainonsynthetic_is_mean.npy')

	std = np.load('TSTR_'+ date +'/train/'+ folder +'/trainonsynthetic_is_std.npy')

	print('Training metrics:')
	print('Inception Score:\nMean score : ', mean[-1])
	print('Std : ', std[-1])

	acc = np.load('TSTR_'+ date +'/train/'+ folder +'/trainonsynthetic_history_acc.npy')
	val_acc = np.load('TSTR_'+ date +'/train/'+ folder +'/trainonsynthetic_history_val_acc.npy')
	loss = np.load('TSTR_'+ date +'/train/'+ folder +'/trainonsynthetic_history_loss.npy')
	val_loss = np.load('TSTR_'+ date +'/train/'+ folder +'/trainonsynthetic_history_val_loss.npy')

	print('ACC : ', np.mean(acc))
	print('VAL_ACC : ', np.mean(val_acc))
	print('LOSS : ', np.mean(loss))
	print('VAL_LOSS : ', np.mean(val_loss))

	print('\nTest metrics:')

	score = np.load('TSTR_'+ date +'/train/'+ folder +'/testonreal_score.npy')
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	roc = np.load('TSTR_'+ date +'/train/'+ folder +'/testonreal_rocauc.npy')
	print('auc roc', roc)


else:

	irr = True
	dataset_syn = 'starlight_amp_noisy_irregular_all_generated'
	one_d = False

## Train on synthetic

	if irr == True:
		X_train, y_train, X_val, y_val, X_test, y_test  = read_data_generated_irr('TSTR_data/generated/'+ folder +'/' + dataset_syn + '.pkl')
	else:
		X_train, y_train, X_val, y_val, X_test, y_test  = read_data('/TSTR_data/generated/'+ folder + '/' + dataset_syn + '.pkl')

	print('')
	print ('Training new model')
	print('')

	batch_size = 512
	epochs = 200

	num_classes = 3

	m = Model_(batch_size, 100, num_classes)

	if one_d == True:
		model = m.cnn()
	else:
		model = m.cnn2()

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

## callbacks
	history = my_callbacks.Histories()
	rocauc = my_callbacks.ROC_AUC(X_train, y_train, X_test, y_test)
	inception = my_callbacks.Inception(X_test, num_classes)

	checkpoint = ModelCheckpoint('TSTR_'+ date +'/train/'+ folder +'/weights.best.trainonsynthetic.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	earlyStopping = EarlyStopping(monitor='val_loss',min_delta = 0.00000001  , patience=10, verbose=1, mode='min') #0.00000001   patience 0

	model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data = (X_val, y_val),
		callbacks = [history,
					checkpoint,
					earlyStopping,
					rocauc,
					inception
					])

	model.save('TSTR_'+ date +'/train/'+ folder +'/trainonsynthetic_model.h5')

	#Create dictionary, then save into two different documments.
## Loss
	history_dictionary_loss = history.loss
	np.save('TSTR_'+ date +'/train/'+ folder +'/trainonsynthetic_history_loss.npy', history_dictionary_loss)
## Val Loss
	history_dictionary_val_loss = history.val_loss
	np.save('TSTR_'+ date +'/train/'+ folder +'/trainonsynthetic_history_val_loss.npy', history_dictionary_val_loss)
## Acc
	history_dictionary_acc = history.acc
	np.save('TSTR_'+ date +'/train/'+ folder +'/trainonsynthetic_history_acc.npy', history_dictionary_acc)
## Val Acc
	history_dictionary_val_acc = history.val_acc
	np.save('TSTR_'+ date +'/train/'+ folder +'/trainonsynthetic_history_val_acc.npy', history_dictionary_val_acc)
## AUC ROC
	roc_auc_dictionary = rocauc.roc_auc
	np.save('TSTR_'+ date +'/train/'+ folder +'/trainonsynthetic_rocauc_dict.npy', roc_auc_dictionary)
## IS
	scores_dict = inception.score
	np.save('TSTR_'+ date +'/train/'+ folder +'/trainonsynthetic_is.npy', scores_dict)
	mean_scores_dict = inception.mean
	np.save('TSTR_'+ date +'/train/'+ folder +'/trainonsynthetic_is_mean.npy', mean_scores_dict)
	std_scores_dict = inception.std
	np.save('TSTR_'+ date +'/train/'+ folder +'/trainonsynthetic_is_std.npy', std_scores_dict)


### plot loss and validation_loss v/s epochs
	plt.figure(1)
	plt.yscale("log")
	plt.plot(history.loss)
	plt.plot(history.val_loss)
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper right')
	plt.savefig('TSTR_'+ date +'/train/'+ folder +'/trainonsynthetic_loss.png')
### plot acc and validation acc v/s epochs
	plt.figure(2)
	plt.yscale("log")
	plt.plot(history.acc)
	plt.plot(history.val_acc)
	plt.title('model acc')
	plt.ylabel('Acc')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper right')
	plt.savefig('TSTR_'+ date +'/train/'+ folder +'/trainonsynthetic_acc.png')



	print('Training metrics:')
	print('Inception Score:\nMean score : ', mean_scores_dict[-1])
	print('Std : ', std_scores_dict[-1])

	print('ACC : ', np.mean(history_dictionary_acc))
	print('VAL_ACC : ', np.mean(history_dictionary_val_acc))
	print('LOSS : ', np.mean(history_dictionary_loss))
	print('VAL_LOSS : ', np.mean(history_dictionary_val_loss))


	## Test on real

	print('\nTest metrics:')

	# Load dataset
	dataset_real = 'starlight_noisy_irregular_all_classes'
	if irr == True:
		X_train, y_train, X_val, y_val, X_test, y_test  = read_data_original_irr('TSTR_data/datasets_original/REAL/'+ dataset_real +'.pkl')
	else:
		X_train, y_train, X_val, y_val, X_test, y_test  = read_data('TSTR_data/datasets_original/REAL/'+ dataset_real +'.pkl')

	sc, me, st = evaluation(X_test, y_test, num_classes)
	np.save('TSTR_'+ date +'/test/'+ folder +'/testonreal_is.npy', sc)
	np.save('TSTR_'+ date +'/test/'+ folder +'/testonreal_is_mean.npy', me)
	np.save('TSTR_'+ date +'/test/'+ folder +'/testonreal_is_std.npy', st)

	score = model.evaluate(X_test, y_test, verbose=1)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	np.save('TSTR_'+ date +'/test/'+ folder +'/testonreal_score.npy', score)

	y_pred = model.predict(X_test)
	roc = roc_auc_score(y_test, y_pred)
	print('auc roc', roc)
	np.save('TSTR_'+ date +'/test/'+ folder +'/testonreal_rocauc.npy', roc)
