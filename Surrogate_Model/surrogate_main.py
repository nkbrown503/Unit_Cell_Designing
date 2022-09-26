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
    save_name = 'UC_Tension_Surrogate_Model_Weights_9-5-22' # or 'cifar-10_plain_net_30-'+timestr
    load_name = "UC_Tension_Surrogate_Model_Weights_9-5-22"
    load_path="checkpoints/"+load_name+"/cp.ckpt"
    checkpoint_path = "checkpoints/"+save_name+"/cp.ckpt"
    Norm_Y=np.load('../Result_Files/Normalizing_Tension_Y_Value.npy')
    Norm_X=np.load('../Result_Files/Normalizing_Compression_X_Value.npy')
elif Type=='Compression':
    save_name = 'UC_Compression_Surrogate_Model_Weights_9-6-22' # or 'cifar-10_plain_net_30-'+timestr
    load_name = "UC_Compression_Surrogate_Model_Weights_9-6-22"
    load_path="checkpoints/"+load_name+"/cp.ckpt"
    checkpoint_path = "checkpoints/"+save_name+'/cp.ckpt'
    Norm_Y=np.load('../Result_Files/Normalizing_Compression_Y_Value.npy')
    Norm_X=np.load('../Result_Files/Normalizing_Compression_X_Value.npy')



file_count =150_000

LS_samp=np.load('../ML_Input_Noise_Files/LS_Design_C_1.npy')
 

#Load in the VAE
#_,encoder,_, _,_ = VAE(np.shape(UC_samp)[0],np.shape(UC_samp)[1],num_channels=1,latent_space_dim=16)
#_,encoder,_ = Autoencoder(np.shape(UC_samp)[0],np.shape(UC_samp)[1],num_channels=1,latent_space_dim=48)
#encoder.load_weights("../VAE/AE_encoder_48.h5") 

#-------Comment Below if Not First Time Training with the Given Dataset--------------
'''
#X_samp= encoder.predict(np.reshape(UC_samp,(1,20,60,1)))
Y_samp=np.load('../ML_Output_Noise_Files/UC_Design_C_1.npy')
X_All=np.zeros((file_count,LS_samp.shape[1]))
Y_All=np.zeros((file_count,Y_samp.shape[0]))
for i in range(0,file_count):
    sys.stdout.write('\rCurrently working on Iteration {}/{}...'.format(i,file_count))
    sys.stdout.flush()  
    Coef=np.load('../ML_Output_Noise_Files/UC_Design_C_{}.npy'.format(i+1))

    LS_Train=np.load('../ML_Input_Noise_Files/LS_Design_C_{}.npy'.format(i+1))[0:20,0:60]

    X_All[i,:]=LS_Train #encoder.predict(np.reshape(UC_Train,(1,20,60,1)))
    Y_All[i,:]=Coef
np.save('X_All_Compression_Set.npy',X_All)
np.save('Y_All_Compression_Set.npy',Y_All)
X_All_Stdev=np.zeros((48,))
X_All_Mean=np.zeros(48,)
for i in range(0,np.shape(X_All)[1]):
    X_All_Stdev[i]=np.std(X_All[:,i])
    X_All_Mean[i]=np.mean(X_All[:,i])
np.save('X_All_Compression_Stdev.npy',X_All_Stdev)
np.save('X_All_Compression_Mean.npy',X_All_Mean)
'''
#--------------Comment Above---------------------------------

#Load the training data which has already been produced running the commented lines above
if Type=='Tension':
    X_All=np.load('X_All_Tension_Set.npy')
    Y_All=np.load('Y_All_Tension_Set.npy')
    X_All_Stdev=np.load('X_All_Tension_Stdev.npy')
    X_All_Mean=np.load('X_All_Tension_Mean.npy')

elif Type=='Compression':
    X_All=np.load('X_All_Compression_Set.npy')
    Y_All=np.load('Y_All_Compression_Set.npy')
    X_All_Stdev=np.load('X_All_Compression_Stdev.npy')
    X_All_Mean=np.load('X_All_Compression_Mean.npy')


#Standardize the Latent Spaces used as the input for the surrogate model
for i in range(0,np.shape(X_All)[1]):
    X_All[:,i]=X_All[:,i]/X_All_Stdev[i]
X_All=X_All.reshape(X_All.shape[0],X_All.shape[1])


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

 

    
    
    
    
    
    
    