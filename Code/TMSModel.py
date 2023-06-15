# -*- coding: utf-8 -*-
"""
Function to run TMS model
@author: Lysea Haggie lmun373@aucklanduni.ac.nz
"""

import cortex as cortexFunctions
import cortex_parameters as params1
from cortex_parameters import *
import plotting as my_plt
import measures as my_measures
import matplotlib.pyplot as plt
import brian2 as b2
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import gc
from datetime import datetime

def run_TMS_model(circuit_type = 'cortical',  TMS_dir = 1, exc_weight = 1, inh_weight = 1, 
                   syn_num = None, ref = 2*b2.ms, exc_del = 1.5, inh_del = 0.8, TMS_prob = [0.25],
                   TMS_time = [150], TMS_type ='single', create_plots = False, save_plots = False, 
                   input_neurons = True):
    ##################### Create neurons and neuron groups #####################  
    
    b2.start_scope()
    gc.collect()
    
    #Table of connections ### Change this for DiLazzaro versus original
    if circuit_type == 'cortical':
        filename = 'Cortex_connections.csv'
    con_tab = pd.read_csv('../Connection_Tables/' + filename, delimiter=' ', index_col=False)
    
    print("Creating Cortex Model")
    ### Initialise Neuron Model ###
    cortex_model = cortexFunctions.cortex_neurons(params1.n_layer, params1.l_name, params1.eqs, 
                                            ref, threshold='v>theta', 
                                            reset =  params1.eqs_reset, reset_potential = params1.v_r)
    ### Create Neuron Groups ###   
    n_groups, n_dict = cortex_model.create_neurons(cols = True, n_cols = 9)
    ### Define Spatial arrangement ###
    cortex_model.spatial_loc(300, 300, 2300, 50, n_groups, n_dict, n_cols = 9)
    
    #print("Creating Synapses")
    
    #####################  Define Input #####################
    if input_neurons == True:
        syn_group = cortex_model.poission_input(n_groups, params1.bg_layer, 'Ie', params1.bg_freq, params1.w_ex)
    elif input_neurons == False:
        syn_group = []
    ##################### Create Synapses #####################
    ### Initialise Synapses ###
    cortex_model.init_synapses(params1.eqs_syn, params1.e_pre, params1.i_pre)
    
    ### Synapses ### ### ### ### ### ### ### ### ### ### ### ### ### Define connectivity type here ###
    syn_g, nsyn_array = cortex_model.create_synapses(n_dict, n_groups, con_tab, 'readin', ignore_syn = syn_num, del_tab = False, 
                                                     exc_weight = exc_weight, inh_weight = inh_weight, 
                                                     exc_delay = exc_del, exc_del_std = 0.5*exc_del, inh_delay = inh_del, inh_del_std = 0.5*inh_del) ### 'random', 'spatial'
    syn_group.append(syn_g)
    
    #####################  Define TMS #####################
    if TMS_type == 'single':
        TMS_synapse, TMS = cortexFunctions.TMS_stimulation(cortex_model.neurons, n_groups, TMS_prob, 
                                                TMS_time*b2.ms, TMS_dir)
        syn_group.append(TMS_synapse)
    
    ### Paired Pulse Different Strengths
    elif TMS_type == 'paired':
        TMS_synapses, TMS = cortexFunctions.TMS_paired_stimulation(cortex_model.neurons, n_groups, TMS_prob, 
                                                TMS_time*b2.ms, TMS_dir)
        syn_group.extend(TMS_synapses)
        
    ### Paired Pulse Inhibitory
    elif TMS_type == 'paired_inhibitory':
        TMS = []
        TMS_in_synapses, TMS_in = cortexFunctions.TMS_inhibitory_stimulation(cortex_model.neurons, n_groups, [TMS_prob[0]], 
                                                    [TMS_time[0]]*b2.ms, TMS_dir)
        syn_group.extend(TMS_in_synapses)
        TMS.append(TMS_in)
        TMS_synapses, TMS_cortex = cortexFunctions.TMS_stimulation(cortex_model.neurons, n_groups, [TMS_prob[1]], 
                                                    [TMS_time[1]]*b2.ms, TMS_dir)
        TMS.append(TMS_cortex)
        syn_group.extend(TMS_synapses)
    
    
    #####################  Run Model #####################
    print("Running Cortex Model")
    statemon, spikemon, ratemon, spikemon_list, ratemon_list = cortex_model.run_model(n_groups, syn_group, params1.simulation_time, TMS)
    
    #print("Plotting")
    if TMS_dir == 0 and create_plots == True:
        print('No TMS - plotting spontaneous activity')
        my_plt.spatial_plot(n_dict, n_groups[:-1], save_plot = save_plots)
        my_plt.raster_plot(spikemon_list, n_groups[:-1], params1.l_name[:-1], save_plot = save_plots)
        
    if TMS_dir == 1 and create_plots == True:
        print('TMS PA direction')
        my_plt.TMS_strength_depth(n_dict, n_groups, save_plot = save_plots)
        my_plt.raster_plot(spikemon_list, n_groups[:-1], params1.l_name[:-1], save_plot = save_plots)
        my_plt.rates_plot(ratemon_list, params1.l_name[:-1], save_plot = save_plots)
        my_plt.wave_plot(spikemon_list, params1.n_layer, save_plot = save_plots)
        
    return statemon, spikemon, ratemon_list #cortex_model.neurons #,nsyn_array