# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 15:00:52 2022

@author: nbrow
"""

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import sys
import os
import numpy as np
import random
import matplotlib.pyplot as plt 
sys.path.insert(0,r'C:\Users\nbrow\OneDrive - Clemson University\Classwork\Doctorate Research\Python Coding\Unit_Cell_Design\VAE')
from autoencoder import VAE, Autoencoder
from FCC_Surrogate_model import FCC_model

model = FCC_model() # or create_plain_net()

#Specify the load type dataset you want to work with, Compression or Tension
Type='Compression'
#Select what you would like to do  Train, Test, or Config 
Action='Test'


#Load the appropriate naming features and normalizing values 
if Type=='Tension':
    save_name = 'UC_Tension_Surrogate_Model_Weights' # or 'cifar-10_plain_net_30-'+timestr
    load_name = "UC_Tension_Surrogate_Model_Weights"
    load_path="checkpoints/"+load_name+"/cp.ckpt"
    checkpoint_path = "checkpoints/"+save_name+"/cp.ckpt"

elif Type=='Compression':
    save_name = 'UC_Compression_Surrogate_Model_Weights' # or 'cifar-10_plain_net_30-'+timestr
    load_name = "UC_Compression_Surrogate_Model_Weights"
    load_path="checkpoints/"+load_name+"/cp.ckpt"
    checkpoint_path = "checkpoints/"+save_name+'/cp.ckpt'




file_count =150_000
if Type=='Compression':
    X_All=np.load('Surrogate_Model_NoisyLS_C_Data.npy')
    Y_All=np.load('Surrogate_Model_NoisyForce_C_Data.npy')
else:
    X_All=np.load('Surrogate_Model_NoisyLS_T_Data.npy')
    Y_All=np.load('Surrogate_Model_NoisyForce_T_Data.npy')


if Action=='Train': 
    history=model.fit(
        x=X_All,
        y=Y_All,
        epochs=150,
        verbose='auto',
        validation_split=.2,
        batch_size=128)
    Saved=False
    while Saved==False:
        try:
            model.save_weights(checkpoint_path)
            Saved=True
        except:
            'Nothing'
    plt.plot(history.history['loss'],label='Training Loss')
    plt.plot(history.history['val_loss'],label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(save_name+'_LossPlot.png')

elif Action=='Test':
    model.load_weights(checkpoint_path)
    Tess=[1,2,3,4]
    ML_Tot=model.predict(X_All)
    for It in range(0,len(Tess)):
        ML_pred=ML_Tot[Tess[It]]
        Truth=Y_All[Tess[It]]
        print('\n-----------')
        print('Ground Truth')
        print(Truth)
        print('\nML Prediction')
        print(ML_pred)
elif Action=='Config':
    hist,bin_edges = np.histogram(X_All,bins=100)
    plt.hist(X_All,bins=bin_edges)
    plt.xlabel('Normalized Coef All')
    #plt.xlim([-1,2])
    #plt.ylim([0,15000])
    plt.ylabel('# of Values')

 

    
    
    
    
    
    
    