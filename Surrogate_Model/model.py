# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:45:14 2022

@author: nbrow
"""

from tensorflow.keras.layers import Input,Dense, Dropout 
from tensorflow.keras.models import Sequential
from tensorflow.keras. initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.models import Model

def FCC_model():

    inputs = Input(shape=(16,))
    t= Dense(81,activation='relu')(inputs)
    #t=Dropout(.2)(t)
    t= Dense(64,activation='relu')(t)
    t=Dropout(.2)(t)
    t=Dense(32,activation='relu')(t)
    t=Dense(16,activation='relu')(t)
    outputs = Dense(4,activation='tanh')(t)
   

    lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=5e-3,
    decay_steps=20000,
    decay_rate=0.9)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss='mean_squared_error')


    return model