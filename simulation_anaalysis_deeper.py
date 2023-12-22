# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:51:02 2023

@author: MATHIAS
"""


#%% libraries 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

#%%

clockwise_dt = pd.read_csv("clockwise_rotation_result_R.csv")

anti_clockwise_dt = pd.read_csv("anti_clockwise_rotation_result_R.csv")

dt_colnames = clockwise_dt.columns

###########################################################
#
#         q study
#
###########################################################

#%%

def retrieve_data(q = 1, position = 0, data = clockwise_dt ):
    """
    this return the subdata according to parameter q provided
    """
    q_frame_col = list()
    for c in data.columns[1:]:
        if float(c.split(", ")[position]) == q : q_frame_col.append(c)
        
    q_dataframe = data[q_frame_col]
    #{"ica_exp", "ica_logcosh", "kernel_ica_kgv", "kernel_ica_kcca"}
    
    ica_exp = list()
    ica_logcosh = list()
    kernel_ica_kgv = list()
    kernel_ica_kcca = list()
    
    for l in q_frame_col:
        if l.split(", ")[-1] == "ica_exp" : ica_exp.append(list(q_dataframe[l]))
        if l.split(", ")[-1] == "ica_logcosh" : ica_logcosh.append(list(q_dataframe[l]))
        if l.split(", ")[-1] == "kernel_ica_kgv" : kernel_ica_kgv.append(list(q_dataframe[l]))
        if l.split(", ")[-1] == "kernel_ica_kcca" : kernel_ica_kcca.append(list(q_dataframe[l]))
        
    list_ica_exp = list()
    list_ica_logcosh = list()
    list_kernel_ica_kgv = list()
    list_kernel_ica_kcca = list()
    
    for j in ica_exp:
        for o in j:
            list_ica_exp.append(o)
    for j in ica_logcosh:
        for o in j:
            list_ica_logcosh.append(o)
    for j in kernel_ica_kgv:
        for o in j:
            list_kernel_ica_kgv.append(o)
    for j in kernel_ica_kcca:
        for o in j:
            list_kernel_ica_kcca.append(o)
                
        
    return_dict = {"ica_exp" : list_ica_exp, 
                  "ica_logcosh" : list_ica_logcosh, 
                  "kernel_ica_kgv" : list_kernel_ica_kgv, 
                  "kernel_ica_kcca" : list_kernel_ica_kcca,
                   }
    result = pd.DataFrame(return_dict)
    
    return result
#%%
q_1 = retrieve_data()

q_1["x"] = q_1.index

#%%

sns.lineplot(data = q_1, x = "x", y = "ica_exp")
sns.lineplot(data = q_1, x = "x", y = "ica_logcosh")


#%%
n_1000 = retrieve_data(1100, 2)
#%%

def frame_retrieve(parameter = [1, 1.01, 1.025, 1.05], position = 1,
                   data = clockwise_dt):
    """
    
    """
    # q_col = [1, 1.01, 1.025, 1.05]
    
    q_line = ["ica_exp", "ica_logcosh", "kernel_ica_kgv", "kernel_ica_kcca"]
    
    frame_mean = pd.DataFrame(data=0, columns=parameter, index = q_line)
    
    frame_median = pd.DataFrame(data=0, columns=parameter, index = q_line)
    
    for u in parameter:
        frame_mean[u] = retrieve_data(q = u, position = position, data=data).mean()
        frame_median[u] = retrieve_data(q = u, position = position, data=data).median()
    
    return frame_mean, frame_median
#%%

frame = frame_retrieve()

#%%

def plot_functions(parameter = [1, 1.01, 1.025, 1.05], 
                   position = 0,
                   data = clockwise_dt ):
    """
    plot of mean and median 
    """
    
    frame = frame_retrieve(parameter = parameter, position = position, data = data)
    
    for_label = frame[0].columns
    
    plt.rcParams['figure.figsize'] = [8, 4]
    
    plt.subplot(121)
    
    
       
    plt.plot(frame[0].index, frame[0][for_label[0]], 'gx', label=for_label[0])
    plt.plot(frame[0].index, frame[0][for_label[1]], 'r*', label=for_label[1])
    plt.plot(frame[0].index, frame[0][for_label[2]], 'b+', label=for_label[2])
    plt.plot(frame[0].index, frame[0][for_label[3]], 'c2', label=for_label[3])
    plt.xticks(rotation = 45)
    plt.grid()
    plt.title("mean")
    plt.legend()
    
    plt.subplot(122)
    plt.plot(frame[1].index, frame[1][for_label[0]], 'gx', label=for_label[0])
    plt.plot(frame[1].index, frame[1][for_label[1]], 'r*', label=for_label[1])
    plt.plot(frame[1].index, frame[1][for_label[2]], 'b+', label=for_label[2])
    plt.plot(frame[1].index, frame[1][for_label[3]], 'c2', label=for_label[3])
    
    plt.xticks(rotation = 45)
    plt.grid()
    plt.title("median")
    plt.legend()
    
    plt.suptitle("Evolution of the mean and median of Amari error accross q values")
    plt.tight_layout()


#%%
plot_functions()

plt.savefig("pdf_amari_q.pdf")

#%%


def plot_functions_3(parameter = [1, 1.01, 1.025, 1.05], 
                   position = 0,
                   data = clockwise_dt, 
                   title = "Evolution of the mean and median of Amari error accross sample size values"  ):
    """
    plot of mean and median 
    """
    
    frame = frame_retrieve(parameter = parameter, position = position, data = data)
    
    for_label = frame[0].columns
    
    plt.rcParams['figure.figsize'] = [8, 4]
    
    plt.subplot(121)
    
    
       
    plt.plot(frame[0].index, frame[0][for_label[0]], 'gx', label=for_label[0])
    plt.plot(frame[0].index, frame[0][for_label[1]], 'r*', label=for_label[1])
    plt.plot(frame[0].index, frame[0][for_label[2]], 'b+', label=for_label[2])
    # plt.plot(frame[0].index, frame[0][for_label[3]], 'c2', label=for_label[3])
    plt.xticks(rotation = 45)
    plt.grid()
    plt.title("mean")
    plt.legend()
    
    plt.subplot(122)
    plt.plot(frame[1].index, frame[1][for_label[0]], 'gx', label=for_label[0])
    plt.plot(frame[1].index, frame[1][for_label[1]], 'r*', label=for_label[1])
    plt.plot(frame[1].index, frame[1][for_label[2]], 'b+', label=for_label[2])
    #plt.plot(frame[1].index, frame[1][for_label[3]], 'c2', label=for_label[3])
    
    plt.xticks(rotation = 45)
    plt.grid()
    plt.title("median")
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
plot_functions_3(parameter = [1100, 2100, 4100], 
                   position = 2)
plt.savefig("pdf_size.pdf")
#%%
plot_functions_3(parameter = [5, 10, 15], 
                   position = 1,
                   title="Evolution of the mean and median of Amari error accross df values")

plt.savefig("pdf_df.pdf")
#%%  



# clockwise schema

plot_functions(parameter = [1, 1.01, 1.025, 1.05], data = clockwise_dt)
#%%
plot_functions_3(parameter = [1100, 2100, 4100], 
                   position = 2, data=clockwise_dt)
#%%
plot_functions_3(parameter = [5, 10, 15], 
                   position = 1, data= clockwise_dt,
                   title="Evolution of the mean and median of Amari error accross df values")
#%%


# Anti-clockwise analysis



plot_functions_3(parameter = [0.99009901, 0.97560976, 0.95238095], 
                 data=anti_clockwise_dt, 
                 title= "Evolution of the mean and median of Amari error accross \n angle q values (anti-clockwise rotation)")
plt.savefig("pdf_amari_anti.pdf")
#%%
plot_functions_3(parameter = [1100, 2100, 4100], 
                   position = 2, data=anti_clockwise_dt,
                   title= "Evolution of the mean and median of Amari error accross \n sample sizes values (anti-clockwise rotation)")
plt.savefig("pdf_size_anti.pdf")
#%%
plot_functions_3(parameter = [5, 10, 15], 
                   position = 1,
                   title="Evolution of the mean and median of Amari error accross \n df values (anti-clockwise rotation)",
                   data=anti_clockwise_dt)
plt.savefig("pdf_df_anti.pdf")

#%%
    
new_df_concrete = pd.DataFrame(data=0.0, 
                      columns = ["q", "df", "n", "ica_exp", "ica_logcosh", 
                                 "kernel_ica_kgv", "kernel_ica_kcca"],
                      index = np.arange(clockwise_dt.shape[0]))

#%%

