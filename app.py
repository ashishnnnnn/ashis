import pandas as pd
from tqdm.notebook import tqdm
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.keras.layers import TimeDistributed
tf.keras.backend.clear_session()
from tensorflow.keras.layers import Input, Softmax, RNN, Dense, Embedding, LSTM
from tensorflow.keras.models import Model
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
import os
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import unicodedata

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd


app = Flask(__name__)

data = pd.read_csv('data.csv',names = ['eng','ita','--'])

eng = list(data['eng'])[:230162]
ita = list(data['ita'])[:230162]


# Converts the unicode file to ascii
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

def preprocess_sentence(lis):
  lis_1 = []
  for w in lis:
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    lis_1.append(w)
  return lis_1  

x_train, x_val, y_train, y_val = train_test_split(eng, ita, test_size=0.009,random_state = 98)

x_train = preprocess_sentence(x_train)
y_train = preprocess_sentence(y_train)

with open("english_tockenizer.pickle","rb") as h:
  english_tockenizer = pickle.load(h)

with open("italian_tockenizer.pickle","rb") as h:
  italian_tockenizer = pickle.load(h)  
 
input_data = english_tockenizer.texts_to_sequences(x_train)
input_data = tf.keras.preprocessing.sequence.pad_sequences(input_data,padding='post')

output_data = italian_tockenizer.texts_to_sequences(y_train)
output_data = tf.keras.preprocessing.sequence.pad_sequences(output_data,padding='post')


with open("index_2_eng.pickle","rb") as h:
  index_2_eng = pickle.load(h)

with open("index_2_ita.pickle","rb") as h:
  index_2_ita = pickle.load(h) 

with open("eng_2_index.pickle" ,"rb") as h:
  eng_2_index = pickle.load(h)

with open("ita_2_index.pickle","rb") as h:
  ita_2_index = pickle.load(h)



embedding_dimn = 100
encoder_unitt = 1024
batch_siz = 456
decoder_unitt = 1024



input_vocab_length = len(english_tockenizer.word_index)+1#Extra one for padding
output_vocab_length = len(italian_tockenizer.word_index)+1#Extra one for padding


data = tf.data.Dataset.from_tensor_slices((input_data, output_data)).shuffle(output_data.shape[0])
data = data.batch(batch_size=batch_siz, drop_remainder=True)


input_length = input_data.shape[1]
output_length = output_data.shape[1]



class Encoder(tf.keras.Model):
  def __init__(self, vocabulary_sz, embedding_dim, encoder_unit, batch_size):
    """
    Here Vocabulary Size is 30 as there are 3 more charcter other than alphabet and zero for padding
    embedding_dim = Dimension for the embedding vector
    encoder unit = Number of unit we use for lstm in encoder
    batch_size = The batch size we use during training
    """
    super(Encoder, self).__init__()
    self.batch_size = batch_size
    self.encoder_unit = encoder_unit
    self.embedding = tf.keras.layers.Embedding(vocabulary_sz, embedding_dim)
    self.encoder = tf.keras.layers.LSTM(self.encoder_unit,return_sequences=True,return_state=True)
  def call(self, inputt, hidden):
    """
    Here we will give inputt and hidden state to the lstm of encoder, and it will return output and updated hidden state
    """
    inputt = self.embedding(inputt)
    outputt, state_h,state_c = self.encoder(inputt, initial_state = hidden)
    """
    Here outputt is the output of every lstm at each time steps.
    state_h is the output of the last lstm cell
    """
    return outputt, state_h ,state_c
  def initialize_hidden_state(self,batch_size):
    """
    It is used to intilalize the first hidden state for lstm encoder.  
    Here We will intilalize by making them zero value
    """
    initial_states = [tf.zeros((batch_size, self.encoder_unit)),tf.zeros((batch_size, self.encoder_unit))]
    return initial_states

encoder = Encoder(input_vocab_length,embedding_dimn,encoder_unitt,batch_siz)

class Attention(tf.keras.layers.Layer):
  def __init__(self,dense_unit = 32):
    super(Attention, self).__init__()
    self.wa = tf.keras.layers.Dense(dense_unit)
    self.wb = tf.keras.layers.Dense(dense_unit)
    self.v = tf.keras.layers.Dense(1)

  def call(self, hid, enc_out):

    gh = tf.expand_dims(hid,1)
    
    sc = self.v(tf.nn.tanh(self.wa(enc_out)+self.wb(gh)))
    #print(self.wa(enc_out).shape)
    #print(self.wb(gh).shape)
    #print((self.wa(enc_out)+self.wb(gh)).shape)
    #print(sc.shape)
    at_we = tf.nn.softmax(sc, axis = 1)
    context_vector = at_we * enc_out
    #print(context_vector.shape,"pre")
    context_vector = tf.reduce_sum(context_vector, axis=1)
    #print(context_vector.shape,"cv1")
    #print(enc_out.shape)

    return context_vector, at_we

attention_layer = Attention()

class OneStepDecoder(tf.keras.Model):
  def __init__(self,tar_vocab_size, embedding_dim, dec_units):
    super(OneStepDecoder, self).__init__()
    self.tar_vocab_size = tar_vocab_size
    self.decoder_units = dec_units
    self.embedding = tf.keras.layers.Embedding(tar_vocab_size, embedding_dim)
    self.decoder = tf.keras.layers.LSTM(self.decoder_units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(tar_vocab_size)

    self.attention = Attention(dec_units)


  def call(self,input_to_decoder, encoder_output, state_h,state_c):
    input_dec = self.embedding(input_to_decoder)
    context_vec,att_weight = self.attention(state_h,encoder_output)
    input_dec = tf.concat([tf.expand_dims(context_vec, 1), input_dec], axis=-1)
    outputt,state_h,state_c = self.decoder(input_dec,[state_h,state_c])
    outputt = tf.reshape(outputt, (-1, outputt.shape[2]))
    outputt = self.fc(outputt)
    return outputt,state_h,state_c,att_weight,context_vec
    
onestepdecoder=OneStepDecoder(output_vocab_length, embedding_dimn, decoder_unitt)

optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
  """
  Here this mask ensure that is there any real output or just padding is there in this output.
  If it will be other than zero then mask will return True else it will return false.
  """
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  #Now loss will be calculated even it is padded("0") input.
  loss_ = loss(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)#This mask will be return 0 be it is padded otherwise it will return 1.
  loss_ *= mask#Now mask output will be multiplied with the loss calculated so if it padded input, then it will not contribute to loss.

  return tf.reduce_mean(loss_) # Here , we are taking mean along the batch to give one value of loss.

@tf.function
def train_step(inp, targ, enc_hidden):
  """
  inp shape = batchsize,max_length_input
  targ shape = batch_size, max_length_output
  """
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, state_h,state_c = encoder(inp, enc_hidden)#Initally the enc_hidden will be all zeros
    dec_input = tf.expand_dims([ita_2_index['<start>']] * batch_siz, 1)
    for t in range(1,targ.shape[1]):
      predictions, state_h,state_c,_,_= onestepdecoder(dec_input, enc_output,state_h,state_c)

      loss += loss_function(targ[:, t], predictions)#Now loss wii be calculated with the prediction given by decoder.

      dec_input = tf.expand_dims(targ[:, t], 1)
      break


  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + onestepdecoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

steps_per_epoch = input_data.shape[0]//batch_siz  

EPOCHS = 10


for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state(batch_siz)
  total_loss = 0
  batch = 0

  for (inp, targ) in data.take(steps_per_epoch):
    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss
    batch+=1
    if batch % 1 == 0:
      break
  # saving (checkpoint) the model every 2 epochs
  break

encoder.load_weights('encoder_eng_2_ita.h5')
onestepdecoder.load_weights('decoder_eng_2_ita.h5')

def preprocess_sentence_(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

def predict(sent):
  attn_plot = np.zeros((output_length,input_length))
  sent = preprocess_sentence_(sent)
  #enc_input = [x_tokenizer.word_index[i] for i in sent.split() ]
  #print(sent)
  enc_input = [english_tockenizer.word_index[i] if i in english_tockenizer.word_index.keys() else english_tockenizer.word_index['UNKE'] for i in sent.split()]
  #print(enc_input)
  enc_input = tf.keras.preprocessing.sequence.pad_sequences([enc_input],maxlen=input_length,padding='post')
  #print(enc_input)
  enc_input = tf.convert_to_tensor(enc_input)
  ##print(enc_input)
  translation = ''
  hidden = [tf.zeros((1, encoder_unitt)),tf.zeros((1, encoder_unitt))]
  enc_out, enc_state_h,enc_state_c = encoder(enc_input, hidden)
  dec_state_h = enc_state_h
  dec_state_c = enc_state_c
  dec_input = tf.expand_dims([italian_tockenizer.word_index['<start>']], 0)
  for t in range(output_length):
    predict,dec_state_h,dec_state_c,att_weight,context_vec = onestepdecoder(dec_input,enc_out,dec_state_h,dec_state_c)
    att_weight = tf.reshape(att_weight, (-1, ))
    attn_plot[t] = att_weight.numpy()
    predict_id = tf.argmax(predict[0]).numpy()
    #print(predict_id)
    translation += italian_tockenizer.index_word[predict_id] + ' '

    if index_2_ita[predict_id] == '<end>':
      return translation

    dec_input = tf.expand_dims([predict_id], 0)

  return translation

print("----------------------------")  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate',methods=['POST'])
def translate():
    '''
    For rendering results on HTML GUI
    '''
    sent1 = request.form.values()
    sent = ""
    for i in sent1:
        sent+=str(i)
        sent+=" "    
    translation = predict(sent)
    #print(translation)
    k = translation.split(" ")
    ret = ""
    for i in range(len(k)-2):
        ret+=k[i]
        ret+=" "   
    output = ret

    return render_template('index.html', prediction_text='Translation is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)    

