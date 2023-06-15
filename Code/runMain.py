# -*- coding: utf-8 -*-
"""
@author: Lysea Haggie lmun373@aucklanduni.ac.nz
"""
import cortex as cortexFunctions
import parameters as params1
from parameters import *
import plotting as my_plt
import measures as my_measures
import matplotlib.pyplot as plt
import ModelTMS
import numpy as np
import brian2 as b2

def main():
    
    # ##### Spontaneous Activity ####
    # statemon, _, _ = ModelTMS.run_main_model(TMS_dir = 0, create_plots = True)
    # my_plt.cortex_colormap(statemon)
    
    ## Regular PA TMS ####
    statemon, _,  ratemon_list_PA = ModelTMS.run_main_model(create_plots=False)
    my_plt.TMS_plot(ratemon_list_PA[4], save_plot=True, file_name = 'PA')
    
    # ##### AP Direction #####
    # _, _, ratemon_list_AP = ModelTMS.run_main_model(TMS_dir = -1, create_plots=True)
    # my_plt.TMS_plot(ratemon_list_AP[4], save_plot=True, file_name='AP')
        
    # ##### Increased inhibition ####
    # _, _, ratemon_list_inh = ModelTMS.run_main_model(inh_weight = 1.4, create_plots=True)
    # my_plt.TMS_inhibition_plot(ratemon_list_PA[4], ratemon_list_inh[4], save_plot=True)

    # ##### Refractory period #####
    # ref_array = np.arange(1, 3.25, 0.25)
    # for r in ref_array:
    #     _, _, ratemon_list_ref = ModelTMS.run_main_model(ref = r*b2.ms, create_plots=True)
    #     my_plt.TMS_plot(ratemon_list_ref[4], save_plot=True, title_name='Refractory Period = {}'.format(r), file_name='Ref{}'.format(r))
    
    # ##### Excitatory Delay #####
    # exc_del_array = np.arange(0.25, 2.5, 0.25)
    # for e in exc_del_array:
    #     _, _, ratemon_list_ref = ModelTMS.run_main_model(exc_del = e, create_plots=True)
    #     my_plt.TMS_plot(ratemon_list_ref[4], save_plot=True, title_name='Inhibitory Delay = {}'.format(e), file_name='InD{}'.format(e))
        
    # ##### Inhibitory Delay #####
    # inh_del_array = np.arange(0.25, 2.5, 0.25)
    # for i in inh_del_array:
    #     _, _, ratemon_list_ref = ModelTMS.run_main_model(inh_del = i, create_plots=True)
    #     my_plt.TMS_plot(ratemon_list_ref[4], save_plot=True, title_name='Excitatory Delay = {}'.format(i), file_name='ExD{}'.format(i))
        
    # ##### Strength Analysis #####
    # filename = 'Strength_Data'
    # #ModelTMS.TMS_Strength_Analysis(filename)
    # my_plt.strength_plot('../Data/'+ filename +'.csv', save_plot=True)
    
    # #### DiLazzaro Circuit #####
    # _, _, ratemon_list_DiLazzaro = ModelTMS.run_main_model(circuit_type = 'DiLazzaro', input_neurons = False, create_plots = True)
    # my_plt.TMS_plot(ratemon_list_DiLazzaro[4], file_name='DiLazzaro_1', save_plot = True)
    
    # ##### Paired Pulse #####
    # _, _, ratemon_list_paired = ModelTMS.run_main_model(TMS_type = 'paired', TMS_prob = [0.3, 0.25], TMS_time = [150, 152], create_plots=True)
    # my_plt.TMS_plot(ratemon_list_paired[4], save_plot = True, file_name='paired1.5')
    
    #### Paired Inhibitory Pulse #####
    #_, _, ratemon_list_paired = ModelTMS.run_main_model(TMS_type = 'paired_inhibitory', TMS_prob = [0.25, 0.25], TMS_time = [150, 151.3])
    #my_plt.TMS_plot(ratemon_list_paired[4], save_plot = True, file_name='pairedinhibtiory1.5')
    
   #  #### Make GIF ####
    # my_plt.spatial_plot_gif('PNG_dir_2/', 250, neurons, statemon)
    # my_plt.make_gif('TMS_L5E_2', 'PNG_dir_2')

    return
    

if __name__=="__main__":
    main()
 