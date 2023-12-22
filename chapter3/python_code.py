# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 09:28:56 2023

@author: MATHIAS
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.decomposition import KernelPCA, FastICA

# np.random.seed(1109)

import warnings
warnings.filterwarnings("ignore")

clockwise_dt = pd.read_csv("clockwise_rotation_result_R.csv")

anti_clockwise_dt = pd.read_csv("anti_clockwise_rotation_result_R.csv")

new_df = pd.DataFrame(data=0.0, 
                      columns = ["q", "df", "n", "ica_exp",
                      "ica_logcosh", "kernel_ica_kgv", 
                      "kernel_ica_kcca"],
                      index = np.arange(36))

new_df_kernel = pd.DataFrame(data=0.0, 
                      columns = ["q", "df", "n", 
                      "kernel_ica_kgv", "kernel_ica_kcca"],
                      index = np.arange(36))


new_df2 = pd.DataFrame(data=0.0, 
                      columns = ["q", "df", "n", "ica_exp",
                      "ica_logcosh", "kernel_ica_kgv",
                      "kernel_ica_kcca"],
                      index = np.arange(36))

new_df2_kernel = pd.DataFrame(data=0.0, 
                      columns = ["q", "df", "n",
                      "kernel_ica_kgv", "kernel_ica_kcca"],
                      index = np.arange(36))
last_col_name = new_df.columns[-4:]

last_col_name_kernel = new_df_kernel.columns[-2:]

def filling_new_df(clockwise_dt = clockwise_dt, new_df = new_df,
                   new_df_kernel = new_df_kernel ) : 
    
    """
    this function is used to fill up the new_df 
    to be used for plotting
    
    """
    
    # b is the index of the row to fill in new df
    b = 0
    
    # columns names of the provided dataframe
    clockwise_dt_columns = list(clockwise_dt.columns)[1:]
    
    
    # loopping throughout the columns of the provided dataframe
    for o in range(0, len(clockwise_dt_columns), 4):
        
        # getting the concerned columns about the next operations
        col_in_action = clockwise_dt_columns[o : o+4]
        
        col_in_action_kernel = [col_in_action[2], col_in_action[3]]
        # col_in_action_logcosh = [col_in_action[1], col_in_action[3]]
        
        # extracting infos to fill the new df
        col_in_action_extract = col_in_action[0].split(", ")
        
        # extracting values of q, df, and n
        new_df.iloc[b, :] = [
        float(col_in_action_extract[0].split(" = ")[-1]) ,
        int(col_in_action_extract[1].split(" = ")[-1]),
        int(col_in_action_extract[2].split(" = ")[-1])-100, 
        0, 0, 0, 0]
        
        new_df_kernel.iloc[b, :] = [
        float(col_in_action_extract[0].split(" = ")[-1]) ,
        int(col_in_action_extract[1].split(" = ")[-1]), 
        int(col_in_action_extract[2].split(" = ")[-1])-100,
        0, 0]

         
        # extract the data to use
        first_step = clockwise_dt[col_in_action]
        
        first_step_kernel = clockwise_dt[col_in_action_kernel]
                
        # apply idxmin and count values
        second_step = first_step.idxmin(axis = 1).value_counts()
        
        second_step_kernel = first_step_kernel.idxmin(
                                axis = 1).value_counts()
        
        
        
        # little manipulation in order to fill the reste of 
        # the new_df columns 
        third_step = pd.DataFrame(second_step) 
        
        third_step_kernel = pd.DataFrame(second_step_kernel) 
       
        
        third_step["order"] = third_step.index
        
        third_step_kernel["order"] = third_step_kernel.index
        
        
        order_of_values = dict()
        
        order_of_values_kernel = dict()
        
        
        
        # trying to keep the order to insure that no messing up 
        # will be observed
        for i in range(len(third_step.index)):
            order_of_values[third_step["order"][i].split(", ")[-1]] = \
            third_step["count"][i]
        
        
        for i in range(len(third_step_kernel.index)):
            order_of_values_kernel[third_step_kernel["order"][i].split(
            ", ")[-1]] = third_step_kernel["count"][i]
         
        
        
         
        # filling the corresponding data into the corresponding cell
        # of new_df
        for j in last_col_name:
            for k in order_of_values.keys():
                if j == k : new_df.loc[b, j] = order_of_values[k]
                
        for j in last_col_name_kernel:
            for k in order_of_values_kernel.keys():
                if j == k : new_df_kernel.loc[b, j] =\
                order_of_values_kernel[k]
            

        # changing to the next index of rows of new_df to fill        
        b +=1  
                

filling_new_df()

filling_new_df(anti_clockwise_dt, new_df2, new_df_kernel=new_df2_kernel)


#%%
plt.rcParams['figure.figsize'] = [20, 9]
sns.lineplot(data=new_df, x = new_df.index, y = "kernel_ica_kgv", 
    color = "c", label = "kernel_ica_kgv")
sns.lineplot(data=new_df, x = new_df.index, y = "kernel_ica_kcca", 
    color = "r", label = "kernel_ica_kcca")
plt.ylabel("")
plt.xlabel("Simulation parameters variation (q, df, n)", 
    fontsize = "xx-large")
plt.xticks(fontsize = "xx-large")
plt.yticks(fontsize = "xx-large")
plt.title("kernelICA algorithms performance from simulation \n clockwise rotation", fontsize = 30, fontstyle = "italic") 
 # xx-large
plt.legend(loc = "lower center", fontsize = "xx-large", reverse = True) 
#["ica_logcosh", "ica_exp", "kernel_ica_kgv", "kernel_ica_kcca"],
# colors = ["c", 'r']
plt.savefig("pdf_try.pdf")


plt.show()


#%%

plt.rcParams['figure.figsize'] = [20, 10]
sns.lineplot(data=new_df, x = new_df.index, y = "ica_logcosh", 
color = "c", label = "ica_logcosh", linewidth = 2)
sns.lineplot(data=new_df, x = new_df.index, y = "ica_exp", 
color = "r", label = "ica_exp", linewidth = 2)
sns.lineplot(data=new_df, x = new_df.index, y = "kernel_ica_kgv", 
color = "b", label = "kernel_ica_kgv")
sns.lineplot(data=new_df, x = new_df.index, y = "kernel_ica_kcca",
color = "g", label = "kernel_ica_kcca")
plt.ylabel("")
plt.xlabel("Simulation parameters variation (q, df, n)", 
fontsize = "xx-large")
plt.xticks(fontsize = "xx-large")
plt.yticks(fontsize = "xx-large")
plt.title("ICA algorithms performance from simulation \n clockwise rotation", fontsize = 30, fontstyle = "italic") #"xx-large"
plt.legend(loc = "center", fontsize = "xx-large", reverse = True, 
bbox_to_anchor=(0.5, 0.2, 0.7, 0.5)) 
#["ica_logcosh", "ica_exp", "kernel_ica_kgv", "kernel_ica_kcca"],
# colors = ["c", 'r']
plt.show()

#%% anti-clockwise rotation analysis

plt.rcParams['figure.figsize'] = [20, 9]
sns.lineplot(data=new_df2_kernel, x = new_df2_kernel.index, 
y = "kernel_ica_kgv", color = "c", label = "kernel_ica_kgv")
sns.lineplot(data=new_df2_kernel, x = new_df2_kernel.index,
y = "kernel_ica_kcca", color = "r", label = "kernel_ica_kcca")
plt.ylabel("")
plt.xlabel("Simulation parameters variation (q, df, n)", 
fontsize = "xx-large")
plt.xticks(fontsize = "xx-large")
plt.yticks(fontsize = "xx-large")
plt.title("kernelICA algorithms performance from simulation \n anti-clockwise rotation", fontsize = 30, fontstyle = "italic") 
#"xx-large"
plt.legend(loc = "lower center", fontsize = "xx-large", reverse = True)
#["ica_logcosh", "ica_exp", "kernel_ica_kgv", "kernel_ica_kcca"], 
# colors = ["c", 'r']
plt.savefig("pdf_try2.pdf")
plt.show()

#%%
plt.rcParams['figure.figsize'] = [20, 10]
sns.lineplot(data=new_df2, x = new_df2.index, 
y = "ica_logcosh", color = "c", label = "ica_logcosh", linewidth = 2)
sns.lineplot(data=new_df2, x = new_df2.index, 
y = "ica_exp", color = "r", label = "ica_exp", linewidth = 2)
sns.lineplot(data=new_df2, x = new_df2.index, y = "kernel_ica_kgv",
color = "b", label = "kernel_ica_kgv")
sns.lineplot(data=new_df2, x = new_df2.index, y = "kernel_ica_kcca",
color = "g", label = "kernel_ica_kcca")
plt.ylabel("")
plt.xlabel("Simulation parameters variation (q, df, n)", 
fontsize = "xx-large")
plt.xticks(fontsize = "xx-large")
plt.yticks(fontsize = "xx-large")
plt.title("ICA algorithms performance from simulation \n anti-clockwise rotation", fontsize = 30, fontstyle = "italic") 
#"xx-large"
plt.legend(loc = "center", fontsize = "xx-large", reverse = True,
           bbox_to_anchor=(0.5, 0.2, 0.7, 0.5)) 
#["ica_logcosh", "ica_exp", "kernel_ica_kgv", "kernel_ica_kcca"],
# colors = ["c", 'r']
plt.show()