import argparse
import sys
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

#Features with ta: https://github.com/bukosabino/ta
from ta import add_all_ta_features
from ta.utils import dropna

#Keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

#Scipy
from scipy.signal import savgol_filter

import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A price forecasting model for crypto trading. Runs in tensorflow 2.''')

parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to data.')
parser.add_argument('--variable_params', nargs=1, type= str, default=sys.stdin, help = 'Path to csv with variable params.')
parser.add_argument('--param_combo', nargs=1, type= int, default=sys.stdin, help = 'Parameter combo.')
parser.add_argument('--checkpointdir', nargs=1, type= str, default=sys.stdin, help = 'Path to checpoint directory. Include /in end')
parser.add_argument('--checkpoint', nargs=1, type= int, default=sys.stdin, help = 'If to checkpoint or not: 1= True, 0 = False')
parser.add_argument('--num_epochs', nargs=1, type= int, default=sys.stdin, help = 'Num epochs (int)')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')

##############FUNCTIONS AND CLASSES################
def format_data(df):
    '''Format and partition the data
    '''

    #Normalize data
    data = df[['close','momentum_ao','trend_dpo','volatility_bbm','volume_adi']]
    #Fill nans with 0
    data = data.fillna(0)
    data = data.values
    #Price
    closing_price = np.copy(data[1:,0])
    #Calc the hourly diff in price
    norm_price = (data[1:,0]-data[:-1,0])/data[:-1,0]
    #Assign
    data = data[1:,:]
    data[:,0]=norm_price
    #Normalize volatility and volume
    feature_max =np.max(np.absolute(data[:,1:]),axis=0)
    data[:,1:] = data[:,1:]/feature_max

    #Test partition evenly
    test_partition = []
    splits = np.arange(0,len(data),int(len(data)/5))
    for i in range(len(splits)-1):
        num = splits[i+1]-splits[i]
        test_partition.extend([i]*num)
    #Add last
    test_partition.extend([i]*(len(data)-len(test_partition)))
    feature_df = pd.DataFrame()
    feature_df['close'] = data[:,0]
    feature_df['momentum_ao'] = data[:,1]
    feature_df['trend_dpo'] = data[:,2]
    feature_df['volatility_atr'] = data[:,3]
    feature_df['volume_adi'] = data[:,4]
    feature_df['test_partition']=test_partition

    return feature_df

def get_train_batch(X_train_1, X_train_2, X_train_3, batch_size,lookback_window,forecast_window):
    '''Generate the train data
    '''

    #Get random selection
    sel1 = np.random.choice(len(X_train_1)-lookback_window-forecast_window,int(batch_size/3),replace=False)
    sel2 = np.random.choice(len(X_train_2)-lookback_window-forecast_window,int(batch_size/3),replace=False)
    sel3 = np.random.choice(len(X_train_3)-lookback_window-forecast_window,int(batch_size/3),replace=False)
    #Get x
    x_batch = []
    #Get y - need to sum the price diff from start to forecast
    y_batch = []
    for i in range(len(sel1)):
        x_batch.append(X_train_1[sel1[i]:sel1[i]+lookback_window,:])
        y_batch.append(np.sum(X_train_1[sel1[i]+lookback_window:sel1[i]+lookback_window+forecast_window,0])) #The first dimension (0) contains the price change
    for i in range(len(sel2)):
        x_batch.append(X_train_2[sel2[i]:sel2[i]+lookback_window,:])
        y_batch.append(np.sum(X_train_2[sel2[i]+lookback_window:sel2[i]+lookback_window+forecast_window,0]))
    for i in range(len(sel3)):
        x_batch.append(X_train_3[sel3[i]:sel3[i]+lookback_window,:])
        y_batch.append(np.sum(X_train_3[sel3[i]+lookback_window:sel3[i]+lookback_window+forecast_window,0]))

    return np.array(x_batch), np.array(y_batch)

def train_generator(X_train_1, X_train_2, X_train_3, batch_size,lookback_window,forecast_window):
    """
    a generator for batches
    """
    while True:
        pairs, targets = get_train_batch(X_train_1, X_train_2, X_train_3, batch_size,lookback_window,forecast_window)
        yield (pairs, targets)

def get_valid_batch(X_valid, batch_size, lookback_window,forecast_window):
    '''Generate the valid data
    '''

    #Get random selection
    sel = np.random.choice(len(X_valid)-lookback_window-forecast_window,int(batch_size),replace=False)
    #Get x
    x_batch = []
    #Get y
    y_batch = []
    for i in range(len(sel)):
        x_batch.append(X_valid[sel[i]:sel[i]+lookback_window,:])
        y_batch.append(np.sum(X_valid[sel[i]+lookback_window:sel[i]+lookback_window+forecast_window,0])) #The first dimension (0) contains the price change

    return np.array(x_batch), np.array(y_batch)

def valid_generator(X_valid, batch_size, lookback_window,forecast_window):
    """
    a generator for batches
    """
    while True:
        pairs, targets = get_valid_batch(X_valid, batch_size, lookback_window,forecast_window)
        yield (pairs, targets)



def create_model(filters, kernel_size, dilation_rate, num_res_blocks,lookback_window):
    '''Get the model
    '''
    #Model inputs
    inputs = layers.Input(shape=(lookback_window,5))
    #Initial convolution
    conv = layers.Conv1D(filters = filters, kernel_size = kernel_size, dilation_rate = dilation_rate, padding ="same")(inputs)

    #Resnet
    def resnet(x, num_res_blocks):
        """Builds a resnet with 1D convolutions of the defined depth.
        """
        # Instantiate the stack of residual units
        #Similar to ProtCNN, but they used batch_size = 64, 2000 filters and kernel size of 21
        for res_block in range(num_res_blocks):
            #block
            c1 = layers.Conv1D(filters = filters, kernel_size = kernel_size, dilation_rate = dilation_rate,  padding ="same")(x)
            b1 = layers.BatchNormalization()(c1) #Bacth normalize, focus on segment
            a1 = layers.Activation('relu')(b1)
            c2 = layers.Conv1D(filters = filters, kernel_size = kernel_size, dilation_rate = dilation_rate,  padding ="same")(a1)
            b2 = layers.BatchNormalization()(c2) #Bacth normalize, focus on segment
            #Skip connection
            s1 = tf.math.add(x, b2) #Skip connection
            x = layers.Activation('relu')(s1)

            return x


    #Apply resnet
    if num_res_blocks>=1:
        conv = resnet(conv, num_res_blocks)

    #Maxpool along sequence axis
    maxpool = layers.GlobalMaxPooling1D()(conv)
    #Flatten
    out = layers.Flatten()(maxpool)
    outputs = layers.Dense(1, activation="tanh")(out)

    model = tf.keras.Model(inputs, outputs)
    #Optimizer
    initial_learning_rate = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=500,
    decay_rate=0.96,
    staircase=True)

    opt = tf.keras.optimizers.Adam(learning_rate = 1e-3,amsgrad=True)

    #Compile
    model.compile(optimizer=opt,loss='mae')

    return model



#####################Problem definition#####################
#Parse args
args = parser.parse_args()
#Data
try:
    feature_df = pd.read_csv(args.datadir[0]+'feature_df.csv')
except:
    df = pd.read_csv(args.datadir[0]+'trade_df.csv')
    # Clean NaN values
    df = dropna(df)
    #Get features
    df = add_all_ta_features(df, open="open", high="high",
                            low="low", close="close", volume="volume")
    feature_df = format_data(df)
    feature_df.to_csv(args.datadir[0]+'feature_df.csv')


#Get parameters
variable_params=pd.read_csv(args.variable_params[0])
param_combo=args.param_combo[0]
checkpointdir = args.checkpointdir[0]
checkpoint = bool(args.checkpoint[0])
num_epochs = args.num_epochs[0]
outdir = args.outdir[0]

#Locate net params
net_params = variable_params.loc[param_combo-1]

#Variable params
filters = int(net_params['filters']) #First dense dim
batch_size = int(net_params['batch_size'])
kernel_size = int(net_params['kernel_size'])
dilation_rate = int(net_params['dilation_rate'])
num_res_blocks = int(net_params['num_res_blocks'])
lookback_window = int(net_params['lookback_window'])
forecast_window = int(net_params['forecast_window'])
test_partition = int(net_params['test_partition'])

#Get data
#Test indices
test_i = feature_df[feature_df.test_partition==test_partition].index
#Save losses
train_losses = []
valid_losses = []
#Go through all valid partitions
for valid_partition in np.setdiff1d(np.arange(5),test_partition):
    print('Validation partition',valid_partition)
    valid_i =feature_df[feature_df.test_partition==valid_partition].index
    train_i = np.setdiff1d(np.arange(len(feature_df)),np.concatenate([test_i,valid_i]))

    #Create model
    model = create_model(filters, kernel_size, dilation_rate, num_res_blocks,lookback_window)
    #Summary of model
    print(model.summary())
    callbacks = []
    #Checkpoint
    if checkpoint == True:
        print('Checkpoint directory exists...')
        checkpoint_path=checkpointdir+"tp_"+str(test_partition)+"_vp_"+str(valid_partition)+"_weights_{epoch:02d}.hdf5"
        checkpointer = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch')

        #Callbacks
        callbacks=[checkpointer]



    #Get train and valid data
    X_train = feature_df.loc[train_i]
    train_partitions = X_train.test_partition.unique()
    #Get the three train splits
    X_train_1 = X_train[X_train.test_partition==train_partitions[0]][['close','momentum_ao','trend_dpo','volatility_atr','volume_adi']].values
    X_train_2 = X_train[X_train.test_partition==train_partitions[1]][['close','momentum_ao','trend_dpo','volatility_atr','volume_adi']].values
    X_train_3 = X_train[X_train.test_partition==train_partitions[2]][['close','momentum_ao','trend_dpo','volatility_atr','volume_adi']].values

    #Get the valid data
    X_valid = feature_df.loc[valid_i,['close','momentum_ao','trend_dpo','volatility_atr','volume_adi']].values

    #Fit
    history = model.fit(train_generator(X_train_1, X_train_2, X_train_3, batch_size,lookback_window,forecast_window),
            epochs=num_epochs,
            steps_per_epoch = int(len(X_train)/batch_size),
            validation_data = valid_generator(X_valid, batch_size, lookback_window,forecast_window),
            validation_steps = int(len(X_valid)/batch_size),
            callbacks=callbacks
            )

    #Save loss
    train_losses.append(history.history['loss'])
    valid_losses.append(history.history['val_loss'])

pdb.set_trace()
#Save array of losses
outid = str(test_partition)+'_'+str(param_combo)
np.save(outdir+'train_losses_'+outid+'.npy',np.array(train_losses))
np.save(outdir+'valid_losses_'+outid+'.npy',np.array(valid_losses))
