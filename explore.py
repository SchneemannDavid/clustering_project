
# regular imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import math
import env

## Visualization Functions for exploring zillow data ##

#-----------------------------------------------------------#

# Function which plots a categorical and continuous var

def plot_categorical_and_continuous_vars(df, categorical, continuous):
    df_sample = df.sample(3000)
    #plt.figure()
    #sns.countplot(x=categorical, data=df_sample)
    plt.figure()
    sns.swarmplot(x=categorical, y=continuous, data=df_sample)
    #plt.figure()
    #sns.boxplot(x=categorical, y=continuous, data=df_sample)
    #plt.figure()
    #sns.violinplot(x=categorical, y=continuous, data=df_sample)
    plt.figure()
    sns.barplot(x=categorical, y=continuous, data=df_sample)

# Function which plots two continuous vars

def plot_continuous_and_continuous_vars(df, continuous1, continuous2):
    df_sample = df.sample(3000)
    #plt.figure()
    #sns.relplot(x=continuous1, y=continuous2, data=df_sample, kind='scatter')
    plt.figure()
    sns.lmplot(x=continuous1, y=continuous2, data=df_sample, scatter=True, hue=None)
    #plt.figure()
    #sns.jointplot(x=continuous1, y=continuous2, data=df_sample, kind='scatter')




# Creating a plot loop that iterates through all combinations of variables
def iteration_plot_categorical_and_continuous_vars(train, cat_cols, cont_cols):
    train_sample = train.sample(5000)
    for cont in cont_cols:
        for cat in cat_cols:
            fig = plt.figure(figsize= (20, 10))
            fig.suptitle(f'{cont} vs {cat}')
            

            plt.subplot(131)
            sns.stripplot(data=train_sample, x = cat, y = cont)
           

            plt.subplot(1, 3, 3)
            sns.boxplot(data = train_sample, x = cont, hue = cat)
            
            
            plt.subplot(1, 3, 2)
            sns.barplot(data = train_sample, x = cat, y = cont)

