import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from keras.preprocessing.image import ImageDataGenerator as imgen
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix
traingen = imgen(preprocessing_function=preprocess_input,
                zoom_range=0.2,
                 shear_range=0.2,
                 horizontal_flip=True,
                 validation_split=0.12
                )
testgen = imgen(preprocessing_function=preprocess_input)
trainds = traingen.flow_from_directory("C:/Users/Saranya/Desktop/python/animal detection system/animal detection/train",
                                      target_size=(128, 128),
                                       seed=123,
                                       batch_size=32,
                                       class_mode="categorical",
                                       subset="training"
                                      )
valds = traingen.flow_from_directory("C:/Users/Saranya/Desktop/python/animal detection system/animal detection/train",
                                     target_size=(128, 128),
                                      seed=123,
                                      batch_size=32,
                                      class_mode="categorical",
                                      subset="validation"
                                     )
testds = testgen.flow_from_directory("C:/Users/Saranya/Desktop/python/animal detection system/animal detection/val",
                                    target_size=(128, 128),
                                      seed=123,
                                      batch_size=32,
                                      class_mode="categorical",
                                     shuffle=False
                                    )
classes = trainds.class_indices
classes = list(classes.keys())
dist = trainds.classes
sns.countplot(x=dist)


def showImages(x, y):
    plt.figure(figsize=[15, 11])
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(x[i])
        plt.title(classes[np.argmax(y[i])])
        plt.axis("off")
    plt.show()


x, y = next(trainds)
showImages(x, y)
base_model = Xception(include_top=False, weights="imagenet", pooling="avg", input_shape=(128, 128, 3))

base_model.trainable = False
image_input = Input(shape=(128, 128, 3))
x = base_model(image_input, training=False)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
image_output = Dense(3, activation="softmax")(x)
model = Model(image_input, image_output)
# compile
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# callbacks
my_calls = [EarlyStopping(monitor="val_accuracy", patience=3),
            ModelCheckpoint("Model.h5", verbose=1, save_best_only=True)]
hist = model.fit(trainds, epochs=15, validation_data=valds, callbacks=my_calls)
model.evaluate(testds)
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(hist.epoch, hist.history['accuracy'], label='Training')
plt.plot(hist.epoch, hist.history['val_accuracy'], label='validation')

plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist.epoch, hist.history['loss'], label='Training')
plt.plot(hist.epoch, hist.history['val_loss'], label='validation')

plt.title("Loss")
plt.legend()
plt.show()
pred = model.predict(testds, verbose=1)
pred = [np.argmax(i) for i in pred]
y_test = testds.classes
print(classification_report(pred, y_test))
sns.heatmap(confusion_matrix(pred, y_test), annot=True, fmt="d", cmap="BuPu")