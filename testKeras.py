import numpy as np
from cv2 import cv2
from tensorflow import keras

img = cv2.imread("circuits/cnn/tests/4.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)[1]

thresh = 255 - thresh

thresh = cv2.resize(thresh, dsize=(150,150), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("th.jpg", thresh)
thresh = thresh.reshape(-1, 150, 150, 1)
model = keras.models.load_model("symbolsModel.h5")

prediction = np.argmax(model.predict(thresh)[0])
print()
print(model.predict(thresh))
print()
print(prediction)
