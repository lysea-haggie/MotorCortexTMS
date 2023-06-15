# -*- coding: utf-8 -*-
"""
@author: Lysea Haggie lmun373@aucklanduni.ac.nz
"""

import matplotlib.pyplot as plt
import brian2 as b2
from brian2 import ms, pA
import numpy as np
from itertools import compress
#import parameters
import random
import scipy
from scipy.signal import find_peaks
import seaborn as sns
import re
import random
import pandas as pd
import igraph as ig
import os
import imageio
import pandas as pd
import itertools
from matplotlib import cm

#3D Spatial Plot
def spatial_plot(n_dict, n_groups, take_all = True, save_plot = False):
    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']
    l_names = list(n_dict.keys())
    print(l_names)
    for i in range(len(n_groups)):       
        #sample from group
        if take_all == False:
            idx_take = np.random.randint(n_groups[i].N, size = int(n_groups[i].N * 0.1))
            
            neurons_X = np.take(np.array(n_groups[i].X), idx_take)
            neurons_Y = np.take(np.array(n_groups[i].Y), idx_take)
            neurons_Z = np.take(np.array(n_groups[i].Z), idx_take)
        
            ax.scatter(neurons_X*1000, neurons_Y*1000, neurons_Z*1000, label = l_names[i], s = 7, alpha = 0.7, color = colours[i])
        
        if take_all == True:
            ax.scatter(n_groups[i].X*1000, n_groups[i].Y*1000, n_groups[i].Z*1000, label = l_names[i], s = 7, alpha = 0.7, color = colours[i])
    ax.legend(markerscale=4, loc='upper left')
    ax.set_xlabel('Distance (mm)')
    ax.set_ylabel('Distance (mm)')
    ax.set_zlabel('Distance (mm)')
    if save_plot == True:
        plt.savefig('../Figures/spatial3D', dpi=300, bbox_inches="tight")
    return

### Spontaneous Activity ColourMap
def cortex_colormap(statemon, save_plot = False):
    rdgy = cm.get_cmap('jet', 256)
    plt.rc('font', size=16)
    data1 = statemon.v[0:10000, 500:]
    fig, axs = plt.subplots(figsize=(10, 4), constrained_layout=True)
    psm = axs.pcolormesh(data1, cmap=rdgy, rasterized=True, vmin=-0.07, vmax=-0.05)
    fig.colorbar(psm, ax=axs)
    axs.set_xlabel('Time (ms)')
    axs.set_ylabel('Neurons')
    #plt.show()
    if save_plot == True:
        plt.savefig('../Figures/SpontaneousActivity_L23E.png', dpi=300)
    
    data2 = statemon.v[17000:27000, 500:]
    fig, axs = plt.subplots(figsize=(10, 4), constrained_layout=True)
    psm = axs.pcolormesh(data2, cmap=rdgy, rasterized=True, vmin=-0.07, vmax=-0.05)
    fig.colorbar(psm, ax=axs)
    axs.set_xlabel('Time (ms)')
    axs.set_ylabel('Neurons')
    #plt.show()
    if save_plot == True:
        plt.savefig('../Figures/SpontaneousActivity_L5E.png', dpi=300, bbox_inches="tight")

## TMS Depth Plot
def TMS_strength_depth(n_dict, n_groups, save_plot = False):

   Z = np.arange(0, 2.3, 0.01)
   w_function = 8 + 4*((2.3 - Z)/2.3)

   fig = plt.figure(figsize=(10,5))
   ax1 = fig.add_subplot(211)
   ax1.plot(Z, w_function)
   ax1.spines['right'].set_visible(False)
   ax1.spines['top'].set_visible(False)
   ax1.set_ylabel('TMS Strength (µA)', fontsize = 16)

   ax2 = fig.add_subplot(212, sharex = ax1)
   colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']
   l_names = list(n_dict.keys())
   for i in range(len(n_groups[:-1])):       
       #sample from group
        idx_take = np.random.randint(n_groups[i].N, size = int(n_groups[i].N * 0.1))
        neurons_X = np.take(np.array(n_groups[i].X), idx_take)
        neurons_Z = np.take(np.array(n_groups[i].Z), idx_take)
        ax2.scatter(neurons_Z*1000, neurons_X*1000, label = l_names[i], s = 7, alpha = 0.7, color = colours[i])
   fig.legend(markerscale=3, loc="right")
   ax2.set_xlabel('Depth (mm)', fontsize = 16)
   ax2.set_ylabel('Distance (mm)', fontsize = 16)
   plt.xticks(fontsize = 16)
   plt.yticks(fontsize = 16)
   if save_plot == True:
       plt.savefig('../Figures/TMSZStrength', dpi=300, bbox_inches="tight")

#TMS I-wave Plot
def TMS_plot(ratemonL5E, title_name = None, file_name = None, TMS_time = 150, save_plot = False):
    plt.rcParams['font.size'] = 16
    time_array = np.linspace(-5, 15, 200)
    #time_array = np.concatenate((zero_pad, times))
    
    plt.figure(figsize=(5, 5))
    plt.xlabel('Time following TMS (ms)') 
    plt.ylabel('Population Firing Rate (Hz)')
    #plt.ylim(-15,1000)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.plot(time_array, ratemonL5E.smooth_rate(window = 'gaussian', width=0.05*ms)[(TMS_time*10 - 50):(TMS_time*10 + 150)]/b2.Hz, 'k')
    #plt.plot(time_array, ratemonL5E.rate[1500:1600]/b2.Hz, 'k')
    if title_name != None:
        plt.title(title_name)
    #plt.show()
    if save_plot == True:
        if file_name == None:
            plt.savefig('../Figures/TMSfig.png', dpi=300, bbox_inches="tight")
        else:
            plt.savefig('../Figures/TMSfig_' + file_name.replace(" ", "_") + '.png', dpi=300, bbox_inches="tight")
    
    return

#Effect of increased inhibition
def TMS_inhibition_plot(ratemonL5E_original, ratemonL5E_inh, TMS_time = 150, save_plot = False):
    plt.rcParams['font.size'] = 16
    time_array = np.linspace(-5, 15, 200)
    #time_array = np.concatenate((zero_pad, times))
    
    plt.figure(figsize=(5, 5))
    plt.xlabel('Time following TMS (ms)') 
    plt.ylabel('Population Firing Rate (Hz)')
    #plt.ylim(-15,1000)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.plot(time_array, ratemonL5E_original.smooth_rate(window = 'gaussian', width=0.05*ms)[(TMS_time*10 - 50):(TMS_time*10 + 150)]/b2.Hz, 'k--', label = 'normal')
    plt.plot(time_array, ratemonL5E_inh.smooth_rate(window = 'gaussian', width=0.05*ms)[(TMS_time*10 - 50):(TMS_time*10 + 150)]/b2.Hz, 'r', label = 'increased inhibition')
    plt.legend(bbox_to_anchor=(0.6, 1), borderaxespad=0)
    #plt.plot(time_array, ratemonL5E.rate[1500:1600]/b2.Hz, 'k')
    #plt.show()
    if save_plot == True:
        plt.savefig('../Figures/TMSinhibitoryfig.png', dpi=300, bbox_inches="tight")
    
    return

#Spiking Behaviour - Raster & Frequency plots
#Raster Plot
def raster_plot(spikemon_list, n_groups, lname, filename=None, save_plot = False):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111) 
    colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']
    for idx, spikemon in enumerate(spikemon_list):
        neuron_num_array = np.random.randint(n_groups[idx].N, size=int(n_groups[idx].N*0.1))
        n_indexes=np.empty(0)
        for neuron_num in neuron_num_array:
            n_indexes = np.concatenate((n_indexes, np.where(np.array(spikemon.i[spikemon.t>50*ms]) == neuron_num)[0]))
        neuron_index = np.take(np.array(spikemon.i[spikemon.t>50*ms]), n_indexes.astype(int))
        neuron_times = np.take(np.array(spikemon.t[spikemon.t>50*ms]), n_indexes.astype(int))
        ax.scatter(neuron_times, neuron_index+(n_groups[idx].start), marker = '.', c = colours[idx])
    ax.invert_yaxis()
    ax.set_xlabel('Time (s)')
    tick_values = []
    for k in range(len(n_groups)):
        tick_values.append(int(np.median([n_groups[k].start, n_groups[k].stop])))
    ax.set_yticks(tick_values)
    ax.set_yticklabels(lname)
    ax.set_ylabel('Neuron Group')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if save_plot == True:
        if filename==None:
            plt.savefig('../Figures/raster', dpi=300, bbox_inches='tight')
        else:
            plt.savefig('../Figures/' + filename, dpi=300, bbox_inches="tight")

#Rates Plot
def rates_plot(ratemon_list, lname, filename=None, save_plot = False):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']
    for i in range(len(ratemon_list)):
        plt.subplot(len(ratemon_list), 1, i+1).set_xticks([])
        #plt.title('Spontaneous Neuron Firing')
        plt.plot(ratemon_list[i].t[1400:1700]/b2.ms, ratemon_list[i].smooth_rate(window = 'gaussian', width=0.05*ms)[1400:1700]/b2.Hz, color = colours[i], label = lname[i])
        #plt.axhline(y=np.mean(ratemon_list[i].smooth_rate(window = 'gaussian', width=0.1*ms)[500:]/b2.Hz), color = 'k' , linestyle = '--')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.annotate(lname[i], xy=(0.9,0.8),xycoords='axes fraction', fontsize=14)
        plt.tick_params(axis='y', which='major', labelsize=14)
        if i == 4:
            plt.ylabel('Frequency (hz)')
        if i == 7:
            plt.xlabel('Time')
    if save_plot == True:
        if filename==None:
            plt.savefig('../Figures/rates2', dpi=300, bbox_inches='tight')
        else:
            plt.savefig('../Figures/' + filename, dpi=300, bbox_inches="tight")
            
def strength_plot(filename, save_plot = False):
    #filename is Strength_Data
    df = pd.read_csv(filename)
    fig, axes = plt.subplots(1, 4, figsize=(15,5), sharex = True, sharey = True)
    df.plot(x='Strength', y='Amp1', ax=axes[0], xlabel='', label='d', color='k')
    df.plot(x='Strength', y='Amp2', ax=axes[1], xlabel='', label='I1', color='k')
    df.plot(x='Strength', y='Amp3', ax=axes[2], xlabel='', label='I2', color='k')
    df.plot(x='Strength', y='Amp4', ax=axes[3], xlabel='', label='I3', color='k')
    fig.add_subplot(1, 1, 1, frameon = False)
    plt.tick_params(labelcolor='none', bottom = False, left = False)
    axes[0].yaxis.set_label_text('Amplitude', fontsize=16)
    plt.xlabel('TMS Strength', fontsize=16)
    #plt.tight_layout()
    if save_plot == True:
        plt.savefig('../Figures/StrengthTMS.png', dpi=300)
    
def wave_plot(spikemon_list, n_layer, save_plot = False):
    ### Sort by most spiking to least spiking neurons
    L5E_spikemon = spikemon_list[4]
    mask = np.logical_and(L5E_spikemon.t>100*ms, L5E_spikemon.t<110*ms)
    plt.plot(np.array(L5E_spikemon.t[mask]), np.array(L5E_spikemon.i[mask]), 'k.')
    data = pd.DataFrame(list(zip(np.array(L5E_spikemon.i[mask]), np.array(L5E_spikemon.t[mask]))), columns = ['i', 't'])
    keys,values = data.sort_values(['i','t']).values.T
    ukeys,index=np.unique(keys,True)
    arrays=np.split(values,index[1:])
    N = n_layer[4]*9
    spk_neuron = pd.DataFrame({'i':range(0,N),'t':[[]]*N, 'num':[0]*N})
    spk_neuron.iloc[ukeys.astype(int),1] = arrays
    num = np.diff(index)
    num = np.append(num, len(keys) - max(index))
    spk_neuron.iloc[ukeys.astype(int),2] = num
    spk_neuron = spk_neuron.sort_values('num')
    spk_neuron['new_i'] = range(0,N)
    t_list = [list(x) for x in spk_neuron['t']]
    t_list_flat = [x for sublist in t_list for x in sublist]
    i_list = list(itertools.chain(*(itertools.repeat(elem, n) for elem, n in zip(spk_neuron['new_i'], spk_neuron['num']))))
    
    new_x = [(t-0.1)*1000 for t in t_list_flat]
    
    plt.figure(figsize=(8,5))
    plt.scatter(new_x, i_list, s=0.1, c='k')
    plt.xlabel('Time following TMS (ms)') 
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    if save_plot == True:
        plt.savefig('../Figures/L5Raster.png', dpi = 300, bbox_inches="tight")

def spatial_plot_gif(png_dir, sim_time, neurons, statemon):
    plt.rcParams['font.size'] = 14
    # make png path if it doesn't exist already
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    plt.set_cmap('rainbow')
    
    idx_take = np.random.randint(neurons[16200:27144].N, size = int(neurons[16200:27144].N * 0.5))

    neurons_X = np.take(np.array(neurons[16200:27144].X), idx_take)
    neurons_Y = np.take(np.array(neurons[16200:27144].Y), idx_take)
    neurons_Z = np.take(np.array(neurons[16200:27144].Z), idx_take)
    
    #for j in range(int((sim_time/0.1) - 500)):   
    for j in range(300):   
    #sample from group
        membrane_voltage = np.take(np.array(statemon.v[16200:27144, j+1400]), idx_take)
        
        ax.scatter(neurons_X*1000, neurons_Y*1000, neurons_Z*1000, s = 7, c = membrane_voltage, vmin = -0.12, vmax = -0.05, alpha = 0.7)
        
        #fig.colorbar(p)
            
        ax.set_xlabel('Distance (mm)')
        ax.set_ylabel('Distance (mm)')
        ax.set_zlabel('Distance (mm)')
        plt.savefig(png_dir +'frame_'+str(j)+'_.png', dpi=100, bbox_inches="tight")
        
    return

def make_gif(gif_name, png_dir):
    images, image_file_names = [],[]
    for file_name in os.listdir(png_dir):
        file_name.replace('PNG_dir', '')
        if file_name.endswith('.png'):
            image_file_names.append(file_name)       
    sorted_files = sorted(image_file_names, key=lambda y: int(y.split('_')[1]))
    # define some GIF parameters
    frame_length = 0.025 # seconds between frames Original=0.025
    end_pause = 4 # seconds to stay on last frame Original=2.5
    # loop through files, join them to image array
    for ii in range(0,len(sorted_files)):       
        file_path = os.path.join(png_dir, sorted_files[ii])
        if ii==len(sorted_files)-1:
            for jj in range(0,int(end_pause/frame_length)):
                images.append(imageio.imread(file_path))
        else:
            images.append(imageio.imread(file_path))
    # the duration is the time spent on each image (1/duration is frame rate)
    imageio.mimsave(gif_name, images,'GIF',duration=frame_length)

    
    
    