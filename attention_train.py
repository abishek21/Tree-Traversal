#### USAGE
## python attention_train.py --traindata path/to/traindata --loadvocab yes --model model_name.h5
### While training for first time pass --loadvocab no, This will let you to create the vocab file. --loadvocab yes loads the
### the vocab already built.

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense
from tensorflow.keras.layers import Activation, dot, concatenate
import argparse
import ast
ap=argparse.ArgumentParser()
ap.add_argument("-d","--traindata",required=True,
                help="path to input dataset")
ap.add_argument("-loadvocab","--loadvocab",required=False,
                help="path to input dataset")
ap.add_argument("-model","--model",required=False,
                help="path to input dataset")
args=vars(ap.parse_args())

df=pd.read_csv(args['traindata'])

df=df.astype(str)


training_input=df['X_data'].values.tolist()[:100000]


training_output=df['Y_data'].values.tolist()[:100000]



PADDING_CHAR_CODE=0
START_CHAR_CODE=1
input_dict_size,output_dict_size=19,19

if args["loadvocab"]=="yes":
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
    print("vocab loaded")

else:
    # Building character encoding dictionary
    def encode_characters(titles):
        count = 2
        encoding = {}
        decoding = {1: 'START'}
        r=set([c for title in titles for c in title.split(" ")])
        r.add(' ')
        r.add('0')
        for c in r:
            encoding[c] = count
            decoding[count] = c
            count += 1
        return encoding, decoding, count



    # Build encoding dictionary
    input_encoding, input_decoding, input_dict_size = encode_characters(training_input)
    output_encoding, output_decoding, output_dict_size = encode_characters(training_output)

    f = open("vocab/input_encoding.txt","w")
    f.write( str(input_encoding) )
    f.close()

    f = open("vocab/input_decoding.txt","w")
    f.write( str(input_decoding) )
    f.close()

    f = open("vocab/output_encoding.txt","w")
    f.write( str(output_encoding) )
    f.close()

    f = open("vocab/output_decoding.txt","w")
    f.write( str(output_decoding) )
    f.close()

# With encoding dictionary, transform the data into a matrix
def transform(encoding, data, vector_size):
    transformed_data = np.zeros(shape=(len(data), vector_size))
    for i in range(len(data)):
        for j in range(min(len(data[i]), vector_size)):
            transformed_data[i][j] = encoding[data[i][j]]
    return transformed_data

# Transform the data
encoded_training_input = transform(input_encoding, training_input, vector_size=19)
encoded_training_output = transform(output_encoding, training_output, vector_size=19)



training_encoder_input = encoded_training_input
training_decoder_input = np.zeros_like(encoded_training_output)
training_decoder_input[:, 1:] = encoded_training_output[:,:-1]
training_decoder_input[:, 0] = START_CHAR_CODE
training_decoder_output = np.eye(output_dict_size)[encoded_training_output.astype('int')]



INPUT_LENGTH=19
OUTPUT_LENGTH=19

encoder_input = Input(shape=(INPUT_LENGTH,))
decoder_input = Input(shape=(OUTPUT_LENGTH,))
encoder = Embedding(input_dict_size, 64, input_length=INPUT_LENGTH, mask_zero=True)(encoder_input)
# encoder = LSTM(64, return_sequences=True, unroll=True)(encoder)
encoder = LSTM(64, return_sequences=True, unroll=True)(encoder)
encoder_last = encoder[:, -1, :]

decoder = Embedding(output_dict_size, 64, input_length=OUTPUT_LENGTH, mask_zero=True)(decoder_input)
decoder = LSTM(64, return_sequences=True, unroll=True)(decoder, initial_state=[encoder_last, encoder_last])

# C[j] = sum( [A[j][i] * E[i] for i range(0, INPUT_LENGTH)] )
# A[j][i] = softmax( D[j] * E[i] ) # softmax by row

# Equation (7) with 'dot' score from Section 3.1 in the paper.
# Note that we reuse Softmax-activation layer instead of writing tensor calculation
attention = dot([decoder, encoder], axes=[2, 2])
attention = Activation('softmax')(attention)

context = dot([attention, encoder], axes=[2, 1])
decoder_combined_context = concatenate([context, decoder])

# Has another weight + tanh layer as described in equation (5) of the paper
output = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context)  # equation (5) of the paper
output = TimeDistributed(Dense(output_dict_size, activation="softmax"))(output)  # equation (6) of the paper

model = Model(inputs=[encoder_input, decoder_input], outputs=[output])


model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(x=[training_encoder_input, training_decoder_input], y=[training_decoder_output],
          verbose=1,
          batch_size=64,
          epochs=3)
model_path="model"+"/"+args["model"]
model.save(model_path)
#model.save_weights(args["model"])
print("model saved")
