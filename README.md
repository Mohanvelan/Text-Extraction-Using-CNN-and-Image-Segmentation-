# Text-Extraction-Using-CNN-and-Image-Segmentation

Extracting text from scanned images has been done using **Convolutional Neural Network (CNN)** and **OpenCV** was used image processing operations on our image data. 
The **Extended-MNIST dataset** (letters dataset) was used for training and testing purpose.
Implementation has been carried out by using python language along with modules such as tensorflow, keras, OpenCV, etc. 

## Training phase


#### One hot encoding

categorical values have been converted into binary vectors by using one hot encoding 

```
from keras.utils.np_utils import to_categorical

for i in range(y_train.shape[0]):
  y_train.iloc[i,] -=1
for i in range(y_test.shape[0]):
  y_test.iloc[i,] -=1

nclasses =y_train.nunique()

```

#### CNN architecture (Net definition)

```
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

net = Sequential()

net.add(Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu', input_shape = (28,28,1)))
net.add(Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu'))
net.add(MaxPooling2D(pool_size = (2,2)))

net.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
net.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
net.add(MaxPooling2D(pool_size = (2,2)))

net.add(Flatten())
net.add(Dense(256, activation = 'relu'))
net.add(Dense(nclasses, activation = 'softmax'))
net.summary()

```


## Testing phase

Testing phase is divided into three parts.
- Testing is done for a single character.
- Testing is done for a single word. (by character segmentation)
- Testing an image for group of words. (by word and character segmentation)

Image segmentation techniques that we used as follows,
- Word segmentation – it converts the given image into a list of words.
- Character segmentation – it segments the word into a list of characters prior to classification.
