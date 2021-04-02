import argparse
import sys
import os
import numpy as np
import pandas as pd
import time


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
parser = argparse.ArgumentParser(description = '''A reinforcement learning algo for crypto trading. Runs in tensorflow 2.''')

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
def get_batch(batch_size,movements_high,movements_low,maxlen):
    """
    Create batch of n inputs and targets dynamically
    """
    #Start index
    random_numbers = np.random.choice(len(movements_high)-maxlen,size=(batch_size,),replace=False) #without replacement

    # initialize vector for the targets
    inputs = np.zeros((batch_size,maxlen))
    targets=np.zeros((batch_size,maxlen))

    pdb.set_trace()
    #Get batch data
    #for i in range(len(random_numbers)): #sample_1444_t1.npy


    return inputs, np.array(targets)

def generate(batch_size, movements_high,movements_low,maxlen):
    """
    a generator for batches, so model.fit_generator can be used.
    """
    while True:
        inputs, targets = get_batch(batch_size,movements_high,movements_low,maxlen)


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


def custom_loss(y_true, y_pred): #y_true are the sought after Sharpe ratios
    '''Keras custom loss function
    '''
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)

    postion =y_pred[:,-1,1]+y_pred[:,-1,2] #momey + ownership
    std = tf.keras.backend.std(y_pred[:,:,0]) #std of movement
    delta = tf.maximum(postion/std,0.01) #y_true = inital capital


    #https://hackernoon.com/bitcoin-sharpe-ratio-the-risk-and-reward-of-investing-in-cryptocurrencies-i81e3xo8
    #The loss should be the return/std deviation of the return (the std of all historical trade outcomes) 
    #- will be similar to sharpe ratio. Want high return and low risk (std)
    return tf.keras.backend.mean(delta)


def calc_outcome(historical_movement,historical_decisions,future_movement,buy_sell_remain,percentage):
    '''Calcualte the outcome from the buy/sell decision
    '''
    buy_sell_remain = tf.math.argmax(buy_sell_remain,axis=-1)

    for j in range(batch_size):
        #Will need to iterate through the whole batch and update 
        #Not sure how to implement the different decisions simultaneously
        #Buy
        if buy_sell_remain==0:
            money_change = historical_movement[j,-1,2]*percentage #last_ownership*buy_percentage
            money = historical_movement[j,-1,1]-money_change #Money left = last money-last_ownership*buy_percentage
            ownership = historical_movement[j,-1,2]+money_change #New ownership = last ownership + buy
            #Update decisions
            historical_decisions[j,:-1,:]=historical_decisions[j,1:,:]
            historical_decisions[j,-1,0]=money_change
            historical_decisions[j,-1,1]=0
            historical_decisions[j,-1,2]=money_change*0.001

        #Sell
        if buy_sell_remain==1:
            money_change = historical_movement[j,-1,2]*percentage #last_ownership*sell_percentage
            money = historical_movement[j,-1,1]+money_change #Money left = last money+last_ownership*sell_percentage
            ownership = historical_movement[j,-1,2]-money_change  #New ownership = last ownership - sell
            #Update decisions
            historical_decisions[j,:-1,:]=historical_decisions[j,1:,:]
            historical_decisions[j,-1,0]=0
            historical_decisions[j,-1,1]=money_change
            historical_decisions[j,-1,2]=money_change*0.001


        #Update money
        historical_movement[j,:-1,1] = historical_movement[j,1:,1] 
        historical_movement[j,-1,1] = money
        #Update ownership
        historical_movement[j,:-1,2] = historical_movement[j,1:,2] 
        historical_movement[j,-1,2] = ownership

    #Future movement
    historical_movement[:,:-1,0]=historical_movement[:,1:,0] 
    historical_movement[:,-1,0]=future_movement


    return historical_movement



def create_model(maxlen, num_heads, ff_dim,num_layers,num_steps):
    '''Create the transformer model
    '''

    historical_movement = layers.Input(shape=(maxlen,3)) #Input historical course movement, money and ownership
    historical_decisions = layers.Input(shape=(maxlen,3)) #Input historical trading decisions: buy, sell, cost
    future_movement = layers.Input(shape=(maxlen,3)) #Input future course movement to calculate earnings/loss

    #Define the transformer
    encoder = EncoderBlock(3, num_heads, ff_dim)
    decoder = DecoderBlock(3, num_heads, ff_dim)

    

    for i in range(num_steps): #Go through n steps and evaluate the final earnings
        x1 = historical_movement
        x2 = historical_decisions
        #Encode
        for j in range(num_layers):
            x1, enc_attn_weights = encoder(x1,x1,x1) #q,k,v
        #Decoder
        for k in range(num_layers):
            x2, enc_dec_attn_weights = decoder(x2,x1,x1) #q,k,v - the k and v from the encoder goes into he decoder


        x2, enc_dec_attn_weights = decoder(x2,x1,x1) #q,k,v - the k and v from the encoder goes into he decoder
        buy_sell_remain = layers.Dense(3, activation="softmax")(x2) #Buy (dim 1)/Sell (dim 2)/Remain (dim 3)
        cat = layers.Concatenate()([x2,buy_sell_remain])
        percentage = layers.Dense(1, activation="softmax")(cat)
        #Update the historical movement each step - changing the course movement, money and ownership
        historical_movement = calc_outcome(historical_movement,historical_decisions,future_movement[i],buy_sell_remain,percentage)


    

    #At test time, the model has to be rewritten without the future_movement. Simply take the layer weights.
    model = keras.Model(inputs=[historical_movement, historical_decisions, future_movement], outputs=historical_movement)
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

#Need to initialize at many different time-points
#to not get stuck in a pattern (for generalization)

#Get data
train_data = pd.read_csv(args.train_data[0])
movements_high = train_data.High.values
movements_low = train_data.Low.values

#Get parameters
batch_size=32
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
maxlen = 90*24  # Only consider the last 90 days, 24 hours. Then trade for the next 90 days to maximize outcome
num_steps = maxlen

#Model
#Variable params
num_heads = 1 #int(net_params['num_heads']) #1  # Number of attention heads
ff_dim = 32 #int(net_params['ff_dim']) #32  # Hidden layer size in feed forward network inside transformer
num_layers = 1 #int(net_params['num_layers']) #1  # Number of attention heads

#Create model
model = create_model(maxlen, num_heads, ff_dim,num_layers,num_steps)

#Summary of model
print(model.summary())

#Fit
history = model.fit_generator(generate(batch_size,movements_high,movements_low,maxlen),
            steps_per_epoch=int(len(train_data)/batch_size*maxlen),
            epochs=num_epochs,
        callbacks=callbacks
    )
