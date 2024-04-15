from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.callbacks import EarlyStopping


def modelCNN(x_train, y_train, k_size):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
	model = keras.Sequential(
		[
			keras.Input(shape=(n_timesteps,n_features)),
			layers.Conv1D(filters=100, kernel_size=k_size, activation='relu'),
			layers.Conv1D(filters=100, kernel_size=k_size, activation='relu'),
			layers.Dropout(0.5),
			layers.MaxPooling1D(pool_size=2),
			layers.Flatten(),
			layers.Dense(100, activation='relu'),
			layers.Dense(n_outputs, activation='softmax')
   		]
	)
	return model

modelA_desc = "LSTM RNN, Credit: Brownlee"
def build_modelA(x_train, y_train, k_size):
    #k_size is not used for the lstm model
  #verbose, epochs, batch_size = 0, 15, 64 # original values, not set in train
  n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
  model = keras.Sequential(
    [
	    layers.LSTM(100, input_shape=(n_timesteps,n_features)),
        layers.Dropout(0.5),
	    layers.Dense(100, activation='relu'),
	    layers.Dense(n_outputs, activation='softmax')
        ])
  return model
  	#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
if (False): #set to True while developing new model
    x_train, y_train, x_valid, y_valid, x_test, y_test, k_size, EPOCHS, t_names = get_dataset('UCI_HAR')
    modelA = build_modelA(x_train, y_train, k_size)
    keras.utils.plot_model(modelA, 'modelA.png',show_shapes=True)
    plt.imshow(plt.imread('modelA.png'))


modelB_desc = "Two 1D-CNN and GlobalAvgPool"
def build_modelB(x_train, y_train, k_size):
  n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

  model = keras.Sequential(
    [
      layers.Conv1D(filters=100, kernel_size=k_size, activation='relu',input_shape=(n_timesteps,n_features)),
      layers.Conv1D(filters=100, kernel_size=k_size, activation='relu'),
      layers.GlobalAveragePooling1D(),
      layers.Dense(n_outputs, activation='softmax')
    ])
  return model
if (False): #set to True while developing new model
    modelB = build_modelB(x_train, y_train, k_size)
    keras.utils.plot_model(modelB, 'modelB.png',show_shapes=True)


     