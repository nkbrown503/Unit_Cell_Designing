# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:14:41 2022

@author: nbrow
"""


import numpy as np
from autoencoder import VAE_SS_Curve
import os
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
import copy
from autoencoder import loss_func, loss_func_SS
import matplotlib.pyplot as plt 
Norm_Y=np.load('../Result_Files/Normalizing_Compression_Y_Value.npy')
Norm_X=np.load('../Result_Files/Normalizing_Compression_X_Value.npy')


def get_training_data(Use_Samples,Tot_Samples):
    x_train=np.zeros((Use_Samples,22))
    
    file_count =Use_Samples
    All_Trials=list(np.linspace(1,Tot_Samples,Tot_Samples).astype('int'))
    Train_Trials=random.sample(All_Trials,int(file_count*.90))
    All_Trials=[x for x in All_Trials if x not in Train_Trials]
    Test_Trials=random.sample(All_Trials, int(file_count*.10))
    x_train=np.zeros((len(Train_Trials),22))
    x_test=np.zeros((len(Test_Trials),22))
    for i in range(0,len(Train_Trials)):
        Results=abs(np.load('../Result_Files/UC_Design_AR3_C_Trial{}.npy'.format(Train_Trials[i]))[:11,:])
        Results[:,0]=Results[:,0]/abs(Norm_X)
        Results[:,1]=Results[:,1]/1e6
        while len(Results)!=11:
            Results=np.append(Results,[Results[-1,:]],axis=0)     
        x_train[i,:]=np.reshape(Results,(22,))
    for j in range(0,len(Test_Trials)):
        Results=abs(np.load('../Result_Files/UC_Design_AR3_C_Trial{}.npy'.format(Test_Trials[j]))[:11,:])
        Results[:,0]=Results[:,0]/abs(Norm_X)
        Results[:,1]=Results[:,1]/1e6
        while len(Results)!=11:
            Results=np.append(Results,[Results[-1,:]],axis=0)     
        x_test[j,:]=np.reshape(Results,(22,))     
    x_train = np.reshape(x_train, newshape=(x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, newshape=(x_test.shape[0], x_test.shape[1], 1))
    np.save('VAE_SS_Curve_Training_Set.npy',x_train)
    np.save('VAE_SS_Curve_Testing_Set.npy',x_test)
    return x_train,x_test


def train(SS_Size,latent_size, learning_rate):
    autoencoder,encoder,decoder, encoder_mu,encoder_log_variance = VAE_SS_Curve(SS_Size,latent_size)
    autoencoder.summary()
    autoencoder.compile(optimizer=Adam(lr=learning_rate), loss=loss_func_SS(encoder_mu, encoder_log_variance))

    return autoencoder, encoder,decoder


if __name__ == "__main__":

    num_channels=1
    Tot_Samples=10000
    Use_Samples=10000
    latent_space_dim=4
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 32
    EPOCHS = 150
    SS_Size=22
    save_name='VAE_SS_curve_training'
    Method='Test'
    #x_train,x_test = get_training_data(Use_Samples,Tot_Samples)
    x_train=np.load('VAE_SS_Curve_Training_Set.npy')
    x_test=np.load('VAE_SS_Curve_Testing_Set.npy')
    if Method=='Train':

        autoencoder,encoder,decoder = train(SS_Size,latent_space_dim,learning_rate=LEARNING_RATE)
        history=autoencoder.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_data=(x_test, x_test))
        plt.plot(history.history['loss'],label='Training Loss')
        plt.plot(history.history['val_loss'],label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.savefig(save_name+'_LossPlot.png')
        autoencoder.save("model")
        encoder.save_weights("VAE_SS_encoder.h5") 
        decoder.save_weights("VAE_SS_decoder.h5")
    elif Method=='Test':
        x_test=np.load('VAE_SS_Curve_Testing_Set.npy')
        autoencoder,encoder,decoder, _,_ = VAE_SS_Curve(SS_Size,latent_space_dim)
        encoder.load_weights("VAE_SS_encoder.h5") 
        decoder.load_weights("VAE_SS_decoder.h5")
        encoded_data = encoder.predict(x_test)
        
        decoded_data = decoder.predict(encoded_data)
        
        for i in range(0,10):
            Num=random.randint(0,1000)
            x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
            plt.plot(np.reshape(x_test[Num,:],(11,2)))
            plt.plot(np.reshape(decoded_data[Num,:],(11,2)))
