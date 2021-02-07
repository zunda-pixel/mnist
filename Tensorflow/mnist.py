from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
x_test  = x_test.astype('float32')
x_test  /= 255

num_classes = len(list(set(y_test)))

y_test = to_categorical(y_test, num_classes)

model = load_model('model_widgets.h5')

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
