import keras
from sklearn.metrics import roc_auc_score
import numpy as np
import pdb
import pickle

class Histories(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.loss = []
		self.val_loss = []
		self.acc = []
		self.val_acc = []

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
		self.loss.append(logs.get('loss'))
		self.val_loss.append(logs.get('val_loss'))
		self.acc.append(logs.get('acc'))
		self.val_acc.append(logs.get('val_acc'))
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return

class ROC_AUC(keras.callbacks.Callback):
	def __init__(self, X_train, y_train, X_val, y_val):
		self.X_train = X_train
		self.y_train = y_train
		self.X_val = X_val
		self.y_val = y_val

	def on_train_begin(self, logs={}):
		self.roc_auc = []
		return

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
		y_pred = self.model.predict(self.X_train)
		roc = roc_auc_score(self.y_train, y_pred)
		y_pred_val = self.model.predict(self.X_val)
		roc_val = roc_auc_score(self.y_val, y_pred_val)
		roc_list = self.roc_auc.append(roc_val)
		#print('roc_auc', roc_val)
		#print('\rroc-auc: %s - roc-auc_val: %s' % (str(roc),str(roc_val)),end=100*' '+'\n')

		#with open('roc_auc_list.pkl', 'wb') as f: pickle.dump(self.roc_list, f)
		#with open('roc_auc_mean.pkl', 'wb') as f: pickle.dump(self.roc_list, f)

		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return

class Inception(keras.callbacks.Callback):
	def __init__(self, X_test, n_classes):
		self.X_test = X_test
		self.n_classes = n_classes

	def on_train_begin(self, logs={}):
		self.score = []
		self.mean = []
		self.std = []
		return

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):

		#pdb.set_trace()

		y_pred_prob = self.model.predict_proba(self.X_test)

		n = 10
		probs = np.array_split(y_pred_prob, n)

		Y = []
		#for prob in probs:
		#	ys = [0, 0, 0]
		#	for j in prob:
		for prob in probs:
			ys = np.zeros(self.n_classes)#[0, 0, 0]
			for class_i in range(self.n_classes):
				for j in prob:
					ys[class_i] = ys[class_i] + j[class_i]

				#ys[0] = ys[0] + j[0]
				#ys[1] = ys[1] + j[1]
				#ys[2] = ys[2] + j[2]

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

		self.score.append(tmp)
		self.mean.append(np.mean(tmp))
		self.std.append(np.std(tmp))

		#print(self.mean[-1])

		#with open('score_noise.pkl', 'wb') as f: pickle.dump(self.score, f)
		#with open('mean_score_noise.pkl', 'wb') as f: pickle.dump(self.mean, f)
		#with open('std_score_noise.pkl', 'wb') as f: pickle.dump(self.std, f)

		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return
