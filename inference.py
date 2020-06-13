import pandas as pd
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense
from keras.layers import Activation, dot, concatenate
import ast


PADDING_CHAR_CODE=0
START_CHAR_CODE=1
input_dict_size,output_dict_size=19,19
INPUT_LENGTH,OUTPUT_LENGTH=19,19

file = open("vocab/input_encoding.txt", "r")

contents = file.read()
input_encoding = ast.literal_eval(contents)

file.close()

file = open("vocab/input_decoding.txt", "r")

contents = file.read()
input_decoding = ast.literal_eval(contents)

file.close()

file = open("vocab/output_encoding.txt", "r")

contents = file.read()
output_encoding = ast.literal_eval(contents)

file.close()

file = open("vocab/output_decoding.txt", "r")

contents = file.read()
output_decoding = ast.literal_eval(contents)

file.close()
def get_model():
  encoder_input = Input(shape=(INPUT_LENGTH,))
  decoder_input = Input(shape=(OUTPUT_LENGTH,))
  encoder = Embedding(input_dict_size, 64, input_length=INPUT_LENGTH, mask_zero=True)(encoder_input)
  #encoder = LSTM(64, return_sequences=True, unroll=True)(encoder)
  encoder = LSTM(64, return_sequences=True, unroll=True)(encoder)
  encoder_last = encoder[:,-1,:]

  decoder = Embedding(output_dict_size, 64, input_length=OUTPUT_LENGTH, mask_zero=True)(decoder_input)
  decoder = LSTM(64, return_sequences=True, unroll=True)(decoder, initial_state=[encoder_last, encoder_last])

  #C[j] = sum( [A[j][i] * E[i] for i range(0, INPUT_LENGTH)] )
  #A[j][i] = softmax( D[j] * E[i] ) # softmax by row

  # Equation (7) with 'dot' score from Section 3.1 in the paper.
  # Note that we reuse Softmax-activation layer instead of writing tensor calculation
  attention = dot([decoder, encoder], axes=[2, 2])
  attention = Activation('softmax')(attention)

  context = dot([attention, encoder], axes=[2,1])
  decoder_combined_context = concatenate([context, decoder])

  # Has another weight + tanh layer as described in equation (5) of the paper
  output = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context) # equation (5) of the paper
  output = TimeDistributed(Dense(output_dict_size, activation="softmax"))(output) # equation (6) of the paper

  model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
  return model








# With encoding dictionary, transform the data into a matrix
def transform(encoding, data, vector_size):
    transformed_data = np.zeros(shape=(len(data), vector_size))
    for i in range(len(data)):
        for j in range(min(len(data[i]), vector_size)):
            transformed_data[i][j] = encoding[data[i][j]]
    return transformed_data

INPUT_LENGTH=19
OUTPUT_LENGTH=19

def generate(txt,model):
    encoder_input = transform(input_encoding, [txt.lower()], INPUT_LENGTH)
    decoder_input = np.zeros(shape=(len(encoder_input), OUTPUT_LENGTH))
    decoder_input[:,0] = START_CHAR_CODE
    for i in range(1, OUTPUT_LENGTH):
        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:,i] = output[:,i]
    return decoder_input

def decode(decoding, sequence):
    text = ''
    for i in sequence:
        if i == 0:
            break
        text += decoding[i]
    return text

def to_order(text,order):
    if order=="inorder":
        model = get_model()
        model.load_weights('weights/inorder_weights.hdf5')
    if order=="preorder":
        model = get_model()
        model.load_weights('weights/preorder_weights.hdf5')
    if order=="postorder":
        model = get_model()
        model.load_weights('weights/postorder_weights.hdf5')


    decoder_output = generate(text,model)
    return decode(output_decoding, decoder_output[0][1:])

seq=input("enter numbers")
ses=str(seq)
what_order=input("Enter the order inorder/postorder/preorder : ")
what_order=str(what_order)
print(to_order(seq,what_order))





