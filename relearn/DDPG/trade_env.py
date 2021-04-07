from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
import pdb
tf.compat.v1.enable_v2_behavior()


class TradeEnvironment(object):
    '''Based on https://levelup.gitconnected.com/a-complex-reinforcement-learning-crypto-trading-environment-in-python-134f3faf0d7a
    '''
    def __init__(self, data, batch_size, fee = 0.001, look_back_window = 90*24, prediction_window=7*24): #90 days, 24 hours|7 days,24 hours
        #Buy/sell/hold + percentage
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.int32, minimum=0, name='action')
        #open, high, low, close, volume, investment, funds
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(batch_size,7,look_back_window), dtype=np.int32, minimum=0, name='observation')
        #State - the market movement: open, high, low, close, volume, investment, funds
        #Env variables
        self.data = data
        self._episode_ended = False
        self.batch_size = batch_size
        self.fee = fee #binance's commission fee per trade is 0.001
        self.look_back_window = look_back_window #how far back our agent will observe to make it's prediction
        self.prediction_window = prediction_window #how far ahead our agent will predict
        self.initial_investment = 0
        self.initial_funds = 10000
        #Get batch data and define state

        random_numbers = np.random.choice(len(self.data)-(self.look_back_window+self.prediction_window),size=(self.batch_size,),replace=False) #without replacement
        self._batch_data = np.zeros((self.batch_size,7,self.look_back_window+self.prediction_window)) #open, high, low, close, investment, funds
        for i in range(len(random_numbers)):
            self._batch_data[i,:5,:] = self.data[random_numbers[i]:random_numbers[i]+self.look_back_window+self.prediction_window].T

        #Assign investment and funds
        self._batch_data[:,5,:] =  self.initial_investment
        self._batch_data[:,6,:] = self.initial_funds
        self._state = self._batch_data[:,:,:self.look_back_window] #Start over - get start states



    def reset(self):
        """Return initial_time_step."""
        random_numbers = np.random.choice(len(self.data)-(self.look_back_window+self.prediction_window),size=(self.batch_size,),replace=False) #without replacement
        self._batch_data = np.zeros((self.batch_size,7,self.look_back_window+self.prediction_window)) #open, high, low, close, investment, funds
        for i in range(len(random_numbers)):
            self._batch_data[i,:5,:] = self.data[random_numbers[i]:random_numbers[i]+self.look_back_window+self.prediction_window].T

        #Assign investment and funds
        self._batch_data[:,5,:] =  self.initial_investment
        self._batch_data[:,6,:] = self.initial_funds
        self._state = self._batch_data[:,:,:self.look_back_window] #Start over - get start states

        return self._state

    def step(self, action, time_step):
        """Apply action and return new time_step."""

        #Containing data for look_back_window:prediction_window.
        #Shape = batch_size,6(open, high, low, close, volume, investment, funds),look_back_window+prediction_window
        investment = self._state[:,5,-1] #Last ownership, number of shares
        funds = self._state[:,6,-1] #Last funds
        action_type = np.array(action[:,0],dtype='int8')
        percentage = action[:,1]
        #percentage[np.isnan(percentage)]=0
        percentage[percentage>1]=1 #Max 1
        next_step_data = self._batch_data[:,:,self.look_back_window+time_step]
        next_step_price = next_step_data[:,3] #close price
        #Get buy
        buy_inds = np.argwhere(action_type==0)[:,0]
        if len(buy_inds)>0:
            #Buy
            current_price = self._state[buy_inds,1,-1] #High price at current time step - buy immediately
            amount = percentage[buy_inds]*funds[buy_inds]
            funds[buy_inds] -= amount
            investment[buy_inds] += amount*(1-self.fee)/current_price #Will buy 0.1 % less than paid for
            #Update next step data
            next_step_data[buy_inds,5]=investment[buy_inds]
            next_step_data[buy_inds,6]=funds[buy_inds]

        #Get sell
        sell_inds = np.argwhere(action_type==1)[:,0]
        if len(sell_inds)>0:
            #Sell
            current_price = self._state[sell_inds,2,-1] #Low price at current time step - sell immediately
            amount = percentage[sell_inds]*funds[sell_inds]
            funds[sell_inds] += amount*(1-self.fee) #Will obtain value of 0.1 % less than paid for
            investment[sell_inds] -= amount/current_price
            next_step_data[sell_inds,5]=investment[sell_inds]
            next_step_data[sell_inds,6]=funds[sell_inds]

        #Get hold
        hold_inds = np.argwhere(action_type==2)[:,0]
        if len(hold_inds)>0:
            #Copy previos to next step
            next_step_data[hold_inds,5]=self._state[hold_inds,5,-1]
            next_step_data[hold_inds,6]=self._state[hold_inds,6,-1]


        #Calculate new price at prediction_window ahead - makes the model have to figure out
        #longer positions to be successful
        #Update investment and funds based on market movement. Normalize with the period volatility to obtain a measure of risk
        #Should take the difference to the simple hold position - will make the agent have to outperform simply holding
        reward = (funds+investment*next_step_price) #/period_volatility
        if np.average(reward)<self.initial_funds*0.1:
            self._episode_ended = True

        #Update state
        new_state = np.zeros(self._state.shape)
        new_state[:,:,:-1]=self._state[:,:,1:]
        new_state[:,:,-1]=next_step_data
        self._state = new_state

        return self._state, np.expand_dims(reward,axis=1),self._episode_ended
