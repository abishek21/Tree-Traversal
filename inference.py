import numpy as np
#import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense
from tensorflow.keras.layers import Activation, dot, concatenate
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


# With encoding dictionary, transform the data into a matrix
def transform(encoding, data, vector_size):
    transformed_data = np.zeros(shape=(len(data), vector_size))
    for i in range(len(data)):
        for j in range(min(len(data[i]), vector_size)):
            transformed_data[i][j] = encoding[data[i][j]]
    return transformed_data


INPUT_LENGTH = 19
OUTPUT_LENGTH = 19


def generate(txt, model):
    encoder_input = transform(input_encoding, [txt.lower()], INPUT_LENGTH)
    decoder_input = np.zeros(shape=(len(encoder_input), OUTPUT_LENGTH))
    decoder_input[:, 0] = START_CHAR_CODE
    for i in range(1, OUTPUT_LENGTH):
        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:, i] = output[:, i]
    return decoder_input


def decode(decoding, sequence):
    text = ''
    for i in sequence:
        if i == 0:
            break
        text += decoding[i]
    return text


def to_order(text, order):
    if order == "inorder":
        model = load_model("model/inorder_model.h5")
    if order == "preorder":
        model = load_model('model/preorder_model.h5')
    if order == "postorder":
        model = load_model('model/postorder_model.h5')

    decoder_output = generate(text, model)
    return decode(output_decoding, decoder_output[0][1:])



root_node=input("Enter root node : ")
root_node=str(root_node)
seq=input("enter 5 children with spaces : ")
seq=str(seq)
s=root_node+" "+seq
what_order=input("Enter the order inorder/postorder/preorder : ")
what_order=str(what_order)
print(to_order(s,what_order))





