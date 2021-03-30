import argparse
import sys
import os
import numpy as np
import pandas as pd
import time
from collections import Counter
#Preprocessing and evaluation
from process_data import parse_and_format

#Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint


#visualization
from tensorflow.keras.callbacks import TensorBoard

from attention_class import MultiHeadSelfAttention #https://apoorvnandan.github.io/2020/05/10/transformer-classifier/
#from lr_finder import LRFinder


import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A reinforcement learning algo for crypto trading.''')

parser.add_argument('--train_data', nargs=1, type= str, default=sys.stdin, help = 'Path to training data.')
#parser.add_argument('--variable_params', nargs=1, type= str, default=sys.stdin, help = 'Path to csv with variable params.')
#parser.add_argument('--param_combo', nargs=1, type= int, default=sys.stdin, help = 'Parameter combo.')
#parser.add_argument('--checkpointdir', nargs=1, type= str, default=sys.stdin, help = 'Path to checpoint directory. Include /in end')
#parser.add_argument('--save_model', nargs=1, type= int, default=sys.stdin, help = 'If to save model or not: 1= True, 0 = False')
#parser.add_argument('--checkpoint', nargs=1, type= int, default=sys.stdin, help = 'If to checkpoint or not: 1= True, 0 = False')
parser.add_argument('--num_epochs', nargs=1, type= int, default=sys.stdin, help = 'Num epochs (int)')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')

#from tensorflow.keras.backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
#sess = tf.Session(config=config)
#set_session(sess)  # set this TensorFlow session as the default session for Keras

#####FUNCTIONS and CLASSES#####

class EncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(EncoderBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim,num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, in_q,in_k,in_v, training): #Inputs is a list with [q,k,v]
        attn_output,attn_weights = self.att(in_q,in_k,in_v) #The weights are needed for downstream analysis
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(in_q + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output), attn_weights

class DecoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(DecoderBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim,num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, in_q,in_k,in_v, training): #Inputs is a list with [q,k,v]
        #Self-attention
        attn_output1,attn_weights1 = self.att(in_q,in_q,in_q) #The weights are needed for downstream analysis
        attn_output1 = self.dropout1(attn_output1, training=training)
        out1 = self.layernorm1(in_v + attn_output1)
        #Encoder-decoder attention
        attn_output2,attn_weights2 = self.att(out1,in_k,in_v) #The weights are needed for downstream analysis
        attn_output2 = self.dropout1(attn_output2, training=training)
        out2 = self.layernorm1(attn_output2 + attn_output1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out2 + ffn_output), attn_weights2

def create_model(maxlen, vocab_size, embed_dim,num_heads, ff_dim,num_layers,num_iterations):
    '''Create the transformer model
    '''

    historical_movement = layers.Input(shape=(maxlen,)) #Input historical course movement
    historical_decisions = layers.Input(shape=(maxlen,)) #Input historical trading decisions

    #Define the transformer
    encoder = EncoderBlock(embed_dim+4, num_heads, ff_dim)
    decoder = DecoderBlock(embed_dim+4, num_heads, ff_dim)
    #Iterate
    for i in range(num_iterations):
        #Encode
        for j in range(num_layers):
            x1, enc_attn_weights = encoder(x1,x1,x1) #q,k,v
        #Decoder
        for k in range(num_layers):
            x2, enc_dec_attn_weights = decoder(x2,x1,x1) #q,k,v - the k and v from the encoder goes into he decoder

       
    x2, enc_dec_attn_weights = decoder(x2,x1,x1) #q,k,v - the k and v from the encoder goes into he decoder
    preds = layers.Dense(6, activation="softmax")(x2) #Annotate
    #preds = layers.Reshape((maxlen,6),name='annotation')(x2)
    #pred_type = layers.Dense(4, activation="softmax",name='type')(x) #Type of protein
    #pred_cs = layers.Dense(1, activation="elu", name='pred_cs')(x)


    model = keras.Model(inputs=[historical_movement, historical_decisions], outputs=preds)
    #Optimizer
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True)

    opt = tf.keras.optimizers.Adam(learning_rate = lr_schedule,amsgrad=True)

    #Compile
    model.compile(optimizer = opt, loss= custom_loss)

    return model

######################MAIN######################
args = parser.parse_args()

#Get data
train_data = pd.read_csv(args.train_data[0])

#Get parameters
#variable_params=pd.read_csv(args.variable_params[0])
#param_combo=args.param_combo[0]
#checkpointdir = args.checkpointdir[0]
#save_model = bool(args.save_model[0])
#checkpoint = bool(args.checkpoint[0])
num_epochs = args.num_epochs[0]
outdir = args.outdir[0]

#Params
#net_params = variable_params.loc[param_combo-1]
#Fixed params
maxlen = 90*24  # Only consider the last 90 days, 24 hours


#Model
#Variable params
num_heads = 1 #int(net_params['num_heads']) #1  # Number of attention heads
ff_dim = 32 #int(net_params['ff_dim']) #32  # Hidden layer size in feed forward network inside transformer
num_layers = 1 #int(net_params['num_layers']) #1  # Number of attention heads

#Create model
model = create_model(maxlen, num_heads, ff_dim,num_layers)

#Summary of model
print(model.summary())
