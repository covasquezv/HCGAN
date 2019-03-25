from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, LSTM, Bidirectional, TimeDistributed
from keras.models import Model
from keras.models import Sequential

class Model_():

	def __init__(self, batch_size, length, num_classes):
		self.batch_size=batch_size
		self.length = length
		self.num_classes = num_classes

	def cnn(self):

		model = Sequential()

		model.add(Conv2D(32, kernel_size=(3,1), padding="same", activation="relu", input_shape=(self.length, 1, 1)))  #32
		model.add(MaxPooling2D(pool_size=(2,1)))

		model.add(Conv2D(64, kernel_size=(3,1), padding="same", activation="relu"))   #64
		model.add(MaxPooling2D(pool_size=(2,1)))

		model.add(Conv2D(128, kernel_size=(3,1), padding="same", activation="relu"))  #128
		model.add(MaxPooling2D(pool_size=(2,1)))

		model.add(Conv2D(64, kernel_size=(3,1), padding="same", activation="relu"))   #64
		model.add(MaxPooling2D(pool_size=(2,1)))

		model.add(Conv2D(64, kernel_size=(1,1), padding="same", activation="relu"))    #64
		model.add(MaxPooling2D(pool_size=(2,1)))

		model.add(Flatten())

		model.add(Dense(100, activation='relu'))  #1024
		model.add(Dropout(0.8))
		model.add(Dense(output_dim=self.num_classes, activation='softmax'))

		print(model.summary())

		return model

	def cnn2(self):
		model = Sequential()

		model.add(Conv2D(32, kernel_size=(3,1), padding="same", activation="relu", input_shape=(self.length, 1, 2)))  #32
		model.add(MaxPooling2D(pool_size=(2,1)))

		model.add(Conv2D(64, kernel_size=(3,1), padding="same", activation="relu"))   #64
		model.add(MaxPooling2D(pool_size=(2,1)))

		model.add(Conv2D(128, kernel_size=(3,1), padding="same", activation="relu"))  #128
		model.add(MaxPooling2D(pool_size=(2,1)))

		model.add(Conv2D(64, kernel_size=(3,1), padding="same", activation="relu"))   #64
		model.add(MaxPooling2D(pool_size=(2,1)))

		model.add(Conv2D(64, kernel_size=(1,1), padding="same", activation="relu"))    #64
		model.add(MaxPooling2D(pool_size=(2,1)))

		model.add(Flatten())

		model.add(Dense(100, activation='relu'))  #1024
		model.add(Dropout(0.8))
		model.add(Dense(output_dim=self.num_classes, activation='softmax'))

		print(model.summary())

		return model

	def fc(self):
		model = Sequential()

		model.add(Dense(100, activation='relu', input_dim=100))
		model.add(Dense(50, activation='relu'))
		model.add(Dense(10, activation='relu'))
		model.add(Dense(output_dim=self.num_classes, activation='softmax'))

		print(model.summary())

		return model

	def bi_lstm(self):
		model = Sequential()
		model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(self.length, 2)))
		model.add(Dropout(0.2))
		model.add(Bidirectional(LSTM(100, return_sequences=True)))
		model.add(Dropout(0.2))
		model.add(Flatten())
		model.add(Dense(output_dim=self.num_classes, activation='softmax'))

		return model

	def lstm(self):
		model = Sequential()
		model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, input_shape=(self.length, 2)))
		model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
		#model.add(Dropout(0.2))
		model.add(Flatten())
		model.add(Dense(output_dim=self.num_classes, activation='softmax'))
		return model
