#!/usr/bin/env python
# coding: utf-8

# ### Model => xml, bin
# 
# Openvino로 inference하기 위해서는 model을 xml, bin 파일로 바꿔주어야 한다.
# Openvino의 model_optimizer가 이 작업을 수행한다.

# ### 1. Keras h5 파일을 Tensorflow pb 파일로 변환

# In[1]:


from tensorflow.keras.models import load_model


# In[2]:


model = load_model('./models/MobilenetV2_class6.h5')


# In[3]:


model.summary()


# In[4]:


import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model

tf.keras.backend.clear_session()

save_pb_dir = './models/tf/'
model_filename = './models/MobilenetV2_class6.h5'
def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_mobilenet.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0) 

model = load_model('./models/MobilenetV2_class6.h5')

session = tf.keras.backend.get_session()

INPUT_NODE = [t.op.name for t in model.inputs]
OUTPUT_NODE = [t.op.name for t in model.outputs]
print(INPUT_NODE, OUTPUT_NODE)
frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)


# In[9]:


get_ipython().run_line_magic('run', 'mo.py --framework tf --input_model C:/dev-tf/models/tf/frozen_mobilenet.pb --batch 1 --data_type FP32 --output_dir C:/dev-tf/models/tf')

