import cv2
import numpy as np
import time

data = []
name = input("Enter Name(Only First Name) - ")
no_of_pics = 100
no_pic = no_of_pics
capture = cv2.VideoCapture(0)



face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while no_of_pics > 0 :
    
    returned ,read_image = capture.read()
    
    if not  returned:
        continue
    
    faces = face_cascade.detectMultiScale(read_image, 1.3,5)
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    faces = faces[:1]

    for face in faces:
        x,y,w,h = face

        only_face = read_image[y:y+h,x:x+w]
        # cv2.imshow("face",only_face)
        # cv2.waitKey()
        print(only_face.shape)
        only_face = cv2.resize(only_face , (100,100))
        print(only_face.shape)
        data.append(only_face)
        no_of_pics -= 1 


print(len(data))
data = np.array(data)
print("data shape ",data.shape)
# print(data)
data = data.reshape((data.shape[0],-1))
print(data.shape)
np.save(("dataset/"+name),data)
capture.release()
cv2.destroyAllWindows()


import os
files = [f for f in os.listdir('dataset') if f.endswith('.npy')]
names = [f[:-4] for f in files]

face_data = []
present = []

for filename in files:
    data = np.load('dataset/'+filename)
    face_data.append(data)


face_data = np.array(face_data)
faces = []
for i in face_data:
    for j in i:
        faces.append(j)
faces = np.array(faces)
print(faces.shape)
faces = np.reshape(faces,(len(faces),100,100,3))
print(faces.shape)

X_train = faces
y_train = []
t = 0
for _ in files:
    y_train = y_train + [t]*no_pic
    t = t + 1



from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, len(files))

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()

classifier.add(Convolution2D(32 ,(3 ,3) ,input_shape=(100 ,100 ,3) ,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32 ,(3 ,3)  ,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu' ))
classifier.add(Dense(units = y_train[0].shape[0], activation = 'softmax' ))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])




datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(X_train)



classifier.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 32, epochs=25)


classifier.save('model.h5')


