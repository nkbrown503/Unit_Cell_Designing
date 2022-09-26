# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 19:19:53 2022

@author: nbrow
"""

from FCC_Surrogate_model import FCC_model
import matplotlib.pyplot as plt 
import numpy as np 
import sys
sys.path.insert(0,r'C:\Users\nbrow\OneDrive - Clemson University\Classwork\Doctorate Research\Python Coding\Unit_Cell_Design\VAE')
from autoencoder import Autoencoder, VAE
import math
import random

surrogate_model= FCC_model() # or create_plain_nest()
_,AE_Encoder,_=Autoencoder(20,60,1,48)
AE_Encoder.load_weights("../VAE/AE_encoder_48.h5") 

Plot=True
avg_error=[]
It=5
Type='Tension'

if Type=='Tension':
    Norm_Y=np.load('../Result_Files/Normalizing_Tension_Y_Value.npy')
    Norm_X=np.load('../Result_Files/Normalizing_Compression_X_Value.npy')
    surrogate_model.load_weights('checkpoints/UC_Tension_Surrogate_Model_Weights_9-5-22/cp.ckpt')
    X_All_Stdev=np.load('../Constant_Values/X_All_Tension_Stdev.npy')
    X_All_Mean=np.load('../Constant_Values/X_All_Tension_Mean.npy')
elif Type=='Compression':
    Norm_Y=np.load('../Result_Files/Normalizing_Compression_Y_Value.npy')
    Norm_X=np.load('../Result_Files/Normalizing_Compression_X_Value.npy')
    surrogate_model.load_weights('checkpoints/UC_Compression_Surrogate_Model_Weights_9-6-22/cp.ckpt')
    X_All_Stdev=np.load('../Constant_Values/X_All_Compression_Stdev.npy')
    X_All_Mean=np.load('../Constant_Values/X_All_Compression_Mean.npy')

pred_list=[]
true_list=[]

Colors=['tab:blue','tab:orange','tab:green','tab:red','tab:cyan']

fig,ax= plt.subplots()
a=[5883,3193,1193,186,5128,2900,1824,258,3305,3627,1013,916,1268,4350,4323,4276,5332,2105,37,1132,2231,2737,3021,5835,3847,3104,5796,5949,1883,5718,18,3159,633,755,1444,5939,5631,4938,547,2402,5243,5183,393,2946,2128,2004,941,3703,2180,4738,757,2433,93,5835,65,5456,3658,2235,5704,493,2138,1005,109,780,456,2977,4127,3960,4009,3332,5685,4466,4611,3985,4078,2327,5881,2890,5614,4727,4688,5329,5126,5203,5519,3995,4168,3529,4413,4108,2991,4879,4125,1091,2978,353,657,3827,4670,4079,5042,2887,394]
for i in range(0,It):
    if Type=='Compression':
        Val=random.randint(1,6000)
    else:
        Val=random.randint(1,3100)
    #UC=[1111,3001,2972,2345,990]
    #Val=UC[i]
    #Val_List=[5883,3193,1193,186,5128,2900,1824,258,1013,916,1268,5332,2105,37,1132,2231,2737,3021,5835,3847,3104,5796,5949,1883,5718,18,3159,633,755,1444,5939,5631,4938,547,2402,5243,5183,393,2946,2128,2004,941,3703,2180,4738,757,2433,93,5835,65,5456,3658,2235,5704,493,2138,1005,109,780,456,2977,4127,3960,4009,3332,5685,4466,4611,3985,4078,2327,5881,2890,5614,4727,4688,5329,5126,5203,5519,3995,4168,3529,4413,4108,2991,4879,4125,1091,2978,353,657,3827,4670,4079]
    #Val=Val_List[i]
    print(Val)
    #Val=Val_list[i]
    sys.stdout.write('\rCurrently working on Iteration {}/{}...'.format(i,It))
    sys.stdout.flush()  
    if Type=='Tension':
        FileName_C='UC_Design_AR3_T_Trial{}'.format(int(Val))
        ML_input=np.load('../ML_Input_Noise_Files/UC_Design_T_{}.npy'.format(Val))[0:20,0:60]
        True_Force=np.load('../ML_Output_Noise_Files/UC_Design_T_{}.npy'.format(Val))    
        Force_Mean=np.load('../Constant_Values/Force_Mean_Tension_Values.npy')
        Force_stdev=np.load('../Constant_Values/Force_stdev_Tension_Values.npy')
    elif Type=='Compression':
        FileName_C='UC_Design_AR3_C_Trial{}'.format(int(Val))
        ML_input=np.load('../ML_Input_Noise_Files/UC_Design_C_{}.npy'.format(Val))[0:20,0:60]
        True_Force=np.load('../ML_Output_Noise_Files/UC_Design_C_{}.npy'.format(Val))
        Force_Mean=np.load('../Constant_Values/Force_Mean_Compression_Values.npy')
        Force_stdev=np.load('../Constant_Values/Force_Stdev_Compression_Values.npy') 
        

    ML_input=ML_input.reshape(1,20,60,1)
    
    ML_input=(AE_Encoder.predict(ML_input,verbose=0))/X_All_Stdev


    Pred_Force=surrogate_model.predict(np.reshape(ML_input,(1,48)),verbose=0)[0] 



    True_Force[1:]=(True_Force[1:]*Force_stdev[1:])+Force_Mean[1:]

    Pred_Force[1:]=(Pred_Force[1:]*Force_stdev[1:])+Force_Mean[1:]
    Pred_Force[0]=0

    if np.min(Pred_Force)>=0 and np.min(True_Force)>=0 and np.max(True_Force)<=1:
        pred_list=np.append(pred_list,Pred_Force)
        true_list=np.append(true_list,True_Force)
    Strain_val=np.linspace(0,1,11)
 
    #Pred_Fit_C=np.array([(Strain**3*Pred_Coef[0])+(Strain**2*Pred_Coef[1])+(Strain*Pred_Coef[2])+Pred_Coef[3] for Strain in Strain_val]) 
    #True_Fit_C=np.array([(Strain**3*True_Coef[0])+(Strain**2*True_Coef[1])+(Strain*True_Coef[2])+True_Coef[3] for Strain in Strain_val]) 

    c1 = np.mean([abs((i-j)/i)*100 for i,j in zip(True_Force[4:],Pred_Force[4:])])
    c2 = np.mean([abs((i-j)/j)*100 for i,j in zip(True_Force[4:],Pred_Force[4:])])
    c = np.mean([abs(i-j) for i,j in zip(True_Force,Pred_Force)])
    if True_Force[-1]>0.1:
        avg_error.append(c)
    
    if Plot:
        #fig,ax= plt.subplots()
        ax.plot(Strain_val,True_Force,'-',color='{}'.format(Colors[i]),label='True: UC {}'.format(Val))
        #ax.plot(Strain_val,Pred_Force,'--',color='{}'.format(Colors[i]),label='Prediction: UC {}'.format(Val))
        ax.plot(Strain_val,Pred_Force,'--',color='{}'.format(Colors[i]))
        ax.legend(loc='best',prop={'size': 8})
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_xlabel('Normalized Displacement')
        ax.set_ylabel('Normalized Force')


print('\nThe average mean % error is: {}%'.format(abs(np.mean(avg_error))))


#plt.scatter(pred_list,true_list)
#plt.xlabel('Real Normalized Force Value')
#plt.ylabel('Predicted Normalized Force Value')

#plt.plot([0,1],[0,1],'r-')


