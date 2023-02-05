import glob
import pickle 
import matplotlib.pyplot as plt

### To load the data ###

# path = 'data/*.pkl'
# files = glob.glob(path)

# result = []
# for file in files:
#     with open(file, 'rb') as f:
#         data = pickle.load(f)
#         result.extend(data)

# with open('result.pkl', 'wb') as f:
#     pickle.dump(result, f)


with open('result.pkl', 'rb') as f:
  data = pickle.load(f)

from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Dropout, BatchNormalization
from keras.models import Model, Sequential
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def model_deep(data):
    # Creating the model
    model = Sequential()

    # Adding convolutional layers
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(8,8,3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Flattening the output from the convolutional layers
    model.add(Flatten())

    # Adding a dense layer for classification
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())

    # Output layer with the number of possible moves as the number of nodes
    model.add(Dense(168, activation='softmax'))

    # Compiling the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model on the input data and output labels
    board = [sublist[0] for sublist in data]
    neg_board = [sublist[1] for sublist in data]
    p_plan = [sublist[2] for sublist in data]
    id_move = [sublist[3] for sublist in data]

    X = np.concatenate((board, neg_board, p_plan), axis=2)  # input data is the first three columns of the data array
    X = X.reshape(-1, 8, 8, 3)
    y = id_move

    # # one-hot encode the output data
    y = to_categorical(y)

    # # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # # fit the model on the train set
    # model.fit(X_train, y_train)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=1,batch_size=64)

    # Plot the training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('loss_plot.png')


    # Plot the training and validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('accuracy_plot.png')

    return model 

print(np.array(data).shape)
model = model_deep(data)

model.save('alexNet_model_2.h5')