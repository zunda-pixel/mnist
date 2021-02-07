from tensorflow.config.experimental import list_physical_devices, set_memory_growth
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

physical_devices = list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        set_memory_growth(physical_devices[k], True)
else:
    print("Not enough GPU hardware devices available")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test  = x_test.reshape(x_test.shape[0],   x_test.shape[1]  * x_test.shape[2])

x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

x_train /= 255
x_test  /= 255

num_classes = len(list(set(y_train)))

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

input_shape = x_train[0].shape

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=input_shape))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
model.summary()

batch_size = 16
epochs = 3

history  = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

model.save('model_widgets.h5')
