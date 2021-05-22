#!/usr/bin/env python
# coding: utf-8

# ### TER Part begins

# In[2]:


print("[+] Importing Modules...")
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import re, sys, os, csv, keras, pickle
import matplotlib.pyplot as plt
import itertools, pickle
from PIL import ImageFile
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Flatten,Concatenate,BatchNormalization,Dense
from itertools import product
from numpy import savetxt
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Input, Flatten, Concatenate
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from keras.engine.topology import Layer, InputSpec
print("[+] Finished Importing Modules!")
print("[+] Using Keras version",keras.__version__)


# In[3]:
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, required=True)    
    args = parser.parse_args()
    return args

args = parse_args()

epochs = args.epoch

MAX_NB_WORDS = 40000         # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 30     # max length of text (words) including padding
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 200          # embedding dimensions for word vectors (word2vec/GloVe)
GLOVE_DIR = "data_files/glove.twitter.27B."+str(200)+"d.txt"
print("[+] Loaded Parameters:\n", MAX_NB_WORDS,MAX_SEQUENCE_LENGTH+5, VALIDATION_SPLIT,EMBEDDING_DIM,"\n", GLOVE_DIR)


# In[4]:


texts, labels = [], []
print("[+] Reading from csv file...", end="")
with open('data_files/text_labels.csv') as csvfile: #CSV file containing label, image (and optionally, text)
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        texts.append(row[1])
        labels.append(row[0])
print("Done!")


# In[5]:


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
with open('data_files/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("[+] Saved word tokenizer to file: tokenizer.pickle")


# In[6]:


with open('data_files/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# In[7]:


sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('[i] Found %s unique tokens.' % len(word_index))
data_int = pad_sequences(sequences, padding='pre', maxlen=(MAX_SEQUENCE_LENGTH-5))
data = pad_sequences(data_int, padding='post', maxlen=(MAX_SEQUENCE_LENGTH))


# In[8]:


labels = to_categorical(np.asarray(labels))       # convert to one-hot encoding vectors
print('[+] Shape of data tensor:', data.shape)
print('[+] Shape of label tensor:', labels.shape)


# In[9]:


indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])


# In[10]:


x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('[i] Number of entries in each category:')
print("[+] Training:\n",y_train.sum(axis=0))
print("[+] Validation:\n",y_val.sum(axis=0))


# In[11]:


embeddings_index = {}
f = open(GLOVE_DIR, encoding="utf8")
print("[+] Loading GloVe from:",GLOVE_DIR,"...",end="")
for line in f:
    values = line.split()
    word = values[0]
    embeddings_index[word] = np.asarray(values[1:], dtype='float32')
f.close()
print("Done.\n[+] Proceeding with Embedding Matrix...", end="")
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print("[+] Completed!")


# In[12]:


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


# In[13]:


def initial_boost(epoch):
    if epoch==0: return float(8.0)
    elif epoch==1: return float(4.0)
    elif epoch==2: return float(2.0)
    elif epoch==3: return float(1.5)
    else: return float(1.0)


# In[14]:


def step_cyclic(epoch):
    try:
        l_r, decay = 1.0, 0.0001
        if epoch%33==0:multiplier = 10
        else:multiplier = 1
        rate = float(multiplier * l_r * 1/(1 + decay * epoch))
        #print("Epoch",epoch+1,"- learning_rate",rate)
        return rate
    except Exception as e:
        print("Error in lr_schedule:",str(e))
        return float(1.0)


# In[15]:


# second embedding matrix for non-static channel
embedding_matrix_ns = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_ns[i] = embedding_vector


# In[16]:


sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

# static channel
embedding_layer_frozen = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
embedded_sequences_frozen = embedding_layer_frozen(sequence_input)

# non-static channel
embedding_layer_train = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix_ns],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
embedded_sequences_train = embedding_layer_train(sequence_input)


# In[17]:


l_lstm1f = Bidirectional(LSTM(6,return_sequences=True,dropout=0.3, recurrent_dropout=0.0))(embedded_sequences_frozen)
l_lstm1t = Bidirectional(LSTM(6,return_sequences=True,dropout=0.3, recurrent_dropout=0.0))(embedded_sequences_train)
l_lstm1 = Concatenate(axis=1)([l_lstm1f, l_lstm1t])


# In[18]:


l_conv_2 = Conv1D(filters=24,kernel_size=2,activation='relu')(l_lstm1)
l_conv_2 = Dropout(0.3)(l_conv_2)
l_conv_3 = Conv1D(filters=24,kernel_size=3,activation='relu')(l_lstm1)
l_conv_3 = Dropout(0.3)(l_conv_3)

l_conv_5 = Conv1D(filters=24,kernel_size=5,activation='relu',)(l_lstm1)
l_conv_5 = Dropout(0.3)(l_conv_5)
l_conv_6 = Conv1D(filters=24,kernel_size=6,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(l_lstm1)
l_conv_6 = Dropout(0.3)(l_conv_6)

l_conv_8 = Conv1D(filters=24,kernel_size=8,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(l_lstm1)
l_conv_8 = Dropout(0.3)(l_conv_8)

conv_1 = [l_conv_6,l_conv_5, l_conv_8,l_conv_2,l_conv_3]

l_lstm_c = Concatenate(axis=1)(conv_1)


# In[19]:


flat=Flatten()(l_lstm_c)


# In[20]:


print(l_conv_2)


# ### TER Part ends
# ### IER Part begins

# In[21]:


test=[]
test_images=pd.read_csv("data_files/image_labels.csv") #Test CSV file containing label, image (and optionally, text)
ImageFile.LOAD_TRUNCATED_IMAGES = True

for i in test_images['image']:
    img=load_img(i,target_size=(224,224))
    img=img_to_array(img)
    img=img/255
    test.append(img)
test_new=np.array(test)
#y_new=model.predict(test_new)
#y_pred_class = np.argmax(y_new,axis=1)
#print(y_pred_class)


# In[22]:


vgg16_model=tf.keras.applications.vgg16.VGG16(input_shape= [224,224,3], weights='imagenet', include_top=False)
vgg16_model.summary()


# In[23]:


vgg16_model.layers.pop()


# In[24]:


vgg16_model.summary()


# In[25]:


for layer in vgg16_model.layers:
        layer.trainable=False


# ### IER Part ends ... Intermediate Fusion model starts

# In[26]:


fl=Flatten()(vgg16_model.output)
df = Dense(3000, activation='relu')(fl)
dl = Dense(500, activation='relu')(df)
concat=Concatenate(axis=1)([dl,flat])

l_conv_4f = Conv1D(filters=12,kernel_size=4,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(embedded_sequences_frozen)
l_conv_4f = Dropout(0.3)(l_conv_4f)
l_conv_4t = Conv1D(filters=12,kernel_size=4,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(embedded_sequences_train)
l_conv_4t = Dropout(0.3)(l_conv_4t)

l_conv_3f = Conv1D(filters=12,kernel_size=3,activation='relu',)(embedded_sequences_frozen)
l_conv_3f = Dropout(0.3)(l_conv_3f)
l_conv_3t = Conv1D(filters=12,kernel_size=3,activation='relu',)(embedded_sequences_train)
l_conv_3t = Dropout(0.3)(l_conv_3t)

l_conv_2f = Conv1D(filters=12,kernel_size=2,activation='relu')(embedded_sequences_frozen)
l_conv_2f = Dropout(0.3)(l_conv_2f)
l_conv_2t = Conv1D(filters=12,kernel_size=2,activation='relu')(embedded_sequences_train)
l_conv_2t = Dropout(0.3)(l_conv_2t)

conv_2 = [l_conv_4f, l_conv_4t,l_conv_3f, l_conv_3t, l_conv_2f, l_conv_2t]

l_merge_2 = Concatenate(axis=1)(conv_2)
l_c_lstm = Bidirectional(LSTM(12,return_sequences=True,dropout=0.3, recurrent_dropout=0.0))(l_merge_2)
l_flat_l = Flatten()(l_c_lstm)
l_merge = Concatenate(axis=1)([concat, l_flat_l])
de = Dense(1000, activation='relu')(l_merge)
d = Dense(4, activation='softmax')(de)


# In[27]:


model = Model(inputs=[sequence_input, vgg16_model.input], outputs=d)
from tensorflow.keras.optimizers import Adam
adam = Adam(lr=0.0001, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[41]:


tensorboard = callbacks.TensorBoard(log_dir='tb_logs', histogram_freq=0, batch_size=16, write_grads=True , write_graph=True)
model_checkpoints = callbacks.ModelCheckpoint("model_checkpoints/checkpoint-{val_loss:.3f}.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=0)
lr_schedule = callbacks.LearningRateScheduler(initial_boost)


# In[29]:


model.summary()


# In[30]:


labels=[y for y in test_images['label']]
l=np.array(labels)


# In[31]:


model.fit([data,test_new],l,batch_size=32, 
          epochs=epochs,
          verbose=1, validation_split=0.3 ,shuffle=True,callbacks=[tensorboard, model_checkpoints])


# In[34]:


model_img=load_model('checkpoints/inter_fusion.h5')    #If pre-trained 'intermediate.h5' is available
model.save("checkpoints/inter_fusion.h5")


# In[ ]:


y_pred=model.predict([data,test_new])
y_pred_class=np.argmax(y_pred,axis=1)


# In[36]:


savetxt('data_files/inter_prob.csv', y_pred, delimiter=',')
savetxt('data_files/inter_pred.csv', y_pred_class, delimiter=',')

