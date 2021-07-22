#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import pickle
import numpy as np


import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array,ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import Callback


# In[2]:


get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[3]:


# 분류할 클래스

classes = ['Coffee', 'glasses', 'mouse', 'papercup', 'phone', 'SangHoon']


# In[4]:


img_shape = [224, 224, 3]
lr = 0.0001
batch_size = 4
epochs = 10


# In[5]:


# 모델 훈련에 사용할 데이터 내용

class_list = []
num_list = []

image_base_path = './facedata/'
train_path = image_base_path + 'train/'
for folder in os.listdir(train_path):
    folder_size = len(os.listdir(train_path + folder))
    class_list.append(folder)
    num_list.append(folder_size)

plotting = pd.Series(num_list, index=class_list)
plotting.sort_values().plot(kind='bar')
plt.show()

print(plotting.sort_values())


# In[6]:


# 모델 훈련에 사용할 데이터 내용

class_list = []
num_list = []

image_base_path = './facedata/'
valid_path = image_base_path + 'valid/'
for folder in os.listdir(valid_path):
    folder_size = len(os.listdir(valid_path + folder))
    class_list.append(folder)
    num_list.append(folder_size)

plotting = pd.Series(num_list, index=class_list)
plotting.sort_values().plot(kind='bar')
plt.show()

print(plotting.sort_values())


# In[7]:


# 모델 훈련에 사용할 데이터 내용

class_list = []
num_list = []

image_base_path = './facedata/'
test_path = image_base_path + 'test/'
for folder in os.listdir(test_path):
    folder_size = len(os.listdir(test_path + folder))
    class_list.append(folder)
    num_list.append(folder_size)

plotting = pd.Series(num_list, index=class_list)
plotting.sort_values().plot(kind='bar')
plt.show()

print(plotting.sort_values())


# ## 전이학습 시작하기

# In[8]:


base_model = MobileNetV2(input_shape=(224,224,3),
                         include_top=False,
                         weights='imagenet'
                        )
base_model.trainable = False


# In[9]:


base_model.summary()


# ## Base Model의 마지막 3개의 Block에서만 학습

# In[10]:


set_trainable = False
for layer in base_model.layers:
    if layer.name in ['block_14_expand', 'block_15_expand', 'block_16_expand']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


# In[11]:


model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(classes), activation='softmax')
])

model.summary()


# In[12]:


model.compile(loss=categorical_crossentropy,
             optimizer = Adam(learning_rate=lr),
             metrics=['accuracy'])


# In[13]:


train_datagen = ImageDataGenerator(rescale=1./255,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)
train_data = train_datagen.flow_from_directory(image_base_path + 'train/',
                                              target_size=(224,224),
                                              batch_size=batch_size,
                                              shuffle=True,
                                              )

valid_datagen = ImageDataGenerator(rescale=1./255)

valid_data = valid_datagen.flow_from_directory(image_base_path + 'valid/',
                                              target_size=(224,224),
                                              batch_size=batch_size,
                                              shuffle=True,
                                              )

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(image_base_path + 'test/',
                                            target_size=(224,224),
                                            batch_size=batch_size,
                                            shuffle=True,
                                            )


# In[14]:


train_data[0]


# In[15]:


history = model.fit(train_data,
                   steps_per_epoch=train_data.n // train_data.batch_size,
                   epochs=10,
                   validation_data=valid_data,
                   validation_steps=valid_data.n // valid_data.batch_size)


# In[16]:


model.save('./models/MobilenetV2_class6.h5')


# In[17]:


test_data.class_indices.items()


# In[18]:


class6 = dict()
for key, value in test_data.class_indices.items():
    class6[value] = key

    
with open('./models/class6.pickle', 'wb') as f:
    pickle.dump(class6, f)


# In[19]:


class6


# ### 모델 평가하기

# In[20]:


train_loss, train_acc = model.evaluate_generator(train_data)

print(train_loss)
print(train_acc)


# In[21]:


test_loss, test_acc = model.evaluate_generator(test_data)

print(test_loss)
print(test_acc)


# ### 테스트 해보기

# In[22]:


def predict_test_img(path):
    img = cv2.imread(path)
    
    model = load_model('./models/MobilenetV2_class6.h5')
    
    print('Original shape : ', img.shape)
    
    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    print('Resized shape : ', img.shape)
    plt.imshow(img)
    plt.show()
    
    result = model.predict_classes(np.expand_dims(img,axis=0))
    with open('./models/class6.pickle', 'rb') as f:
        class6 = pickle.load(f)
    print('Predict : {}'.format(class6[result[0]]))
    
    predicted_result = model.predict(np.expand_dims(img, axis=0))
    
    pd.DataFrame(predicted_result, columns=class6.values()).iloc[0].plot(kind='bar')
    plt.show()    


# In[23]:


predict_test_img('./facedata/dylan/')


# In[29]:


import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
frozen_out_path = './models/tf/'
frozen_graph_filename = 'MobileNetV2'

full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(shape= (224,224), dtype=tf.float32)
)

frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print('-'*60)
print('Frozen model layers:')
for layer in layers:
    print(layer)

print("-" * 60)
print("Fronzen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)


# In[ ]:




