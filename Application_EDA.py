#imports
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os


##Data Exploration
def explore_data():
    
    #general info of the data and how it is organized
    print("Data Information:")
    print(data.info, '\n') 

    #list each column heading and type of data it contains
    print("Columns/Metrics and Associated Data Types:")
    print(data.dtypes, '\n')

    #Since "trader" and "trader_label" are the only columns with objects rather than numbers,
    #sample some data from both of those columns to see what they contain
    print("Data of \'trader_label\' column:")
    print(data['trader_label'], '\n')
    print("Data of \'trader\' column:")
    print(data['trader'], '\n') 
    #data is nums, aside from "trader" (unique strings of characters identifying traders) 
    #and "trader_label" (descriptive strings: awful, bad, good, sharp)

    #checks for null values
    print("Number of null values in each column:")
    print(data.isnull().sum(), '\n') 
    #gets number of traders w/ 0 or 1 transaction
    low_tx_traders = (data["transaction_count"] <= 1).sum()
    print(f"Traders with 0 or 1 transactions: {low_tx_traders}")
    #nulls are present in all columns related to std (standard deviation)
    #seems to be bc of performing stdev operation on only 0 or 1 transaction 
    #(number of traders w 0 or 1 transactions matches number of nulls)

    #Check that the topic values for a given trader add up to 1, suggesting that they represent
    #the trader's involvement in each topic
    sum = 0
    for c in data.columns:
        if c.startswith("topic_"):
            sum += data[c][0]
    print("Topic totals for a trader: ", sum)


##Helper method to filter outliers by IQR out of a given dataset
def removeOutliersByIQR(outlier_data):

    #compute IQR
    Q1 = np.percentile(outlier_data, 25)
    Q3 = np.percentile(outlier_data, 75)
    IQR = Q3 - Q1

    #filter data by dropping values outside IQR-based range
    filtered_data = outlier_data[(outlier_data >= (Q1 - 1.5 * IQR)) & (outlier_data <= (Q3 + 1.5 * IQR))]
    return filtered_data


#Initial data insights
def insights():
    
    #General Trader Insights
    print("Number of traders: ", len(data))
    print(data["trader_label"].value_counts(), "\n")

    #PnL Insights
    print("PnL Statistics: \n", data["trader_pnl"].describe())
    print("Number of profitable traders (by PnL): ", (data["trader_pnl"] > 0).sum(), "\n")

    #PPV (PnL per volume) insights
    print("PPV Statistics: \n", data["trader_ppv"].describe(), "\n")

    #Trading activity insights
    highvol_threshold = data['trader_volume'].quantile(0.9)
    highvol_traders = data[data['trader_volume'] >= highvol_threshold]
    high_volume_profitable = (highvol_traders['trader_pnl'] > 0).sum()
    print("High-volume traders (top 10%): ", len(highvol_traders))
    print("Profitable high-volume traders: ", high_volume_profitable, "\n")

    #Current activity insights
    print("Average transactions per day: ", data["transactions_per_day"].mean())
    print("Average volume per day: ", data["volume_per_day"].mean())
    print("Average markets per day: ", data["markets_per_day"].mean(), "\n")

    #Timing insights
    timing_cols = ['mean_time', 'std_time', 'mean_time_vw', 'std_time_vw']
    print("Timing statistics:")
    print(data[timing_cols].describe(), "\n")


#Basic plots showing distributions of key metrics
def simple_plots():
    
    #define the key columns/metrics
    key_cols = ["trader_pnl", "trader_ppv", "trader_volume", "transaction_count", "price_levels_consumed", "markets_per_day"]

    #plot distribution of each key metric
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Distributions of Key Statistics')
    axes = axes.flatten()
    for i, col in enumerate(key_cols):
        if col == "price_levels_consumed" or col == "markets_per_day":
            axes[i].set_yscale('log')
            sns.histplot(data[col], bins=100, ax=axes[i])
        else:
            sns.histplot(removeOutliersByIQR(data[col]), bins=100, ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    plt.show()


#Plots based on trader topic focus
def topic_plots():
    
    #Get topic columns
    topic_cols = [c for c in data.columns if c.startswith("topic_")]

    # Compute and plot the average topic share between all traders
    topic_means = data[topic_cols].mean().sort_values(ascending=False)
    plt.figure(figsize=(12,6))
    sns.barplot(x=topic_means.values, y=topic_means.index)
    plt.title("Average Topic Share Across Traders")
    plt.show()

    # Plot topic focus by trader label
    topic_by_label = data.groupby("trader_label")[topic_cols].mean().T
    topic_by_label.plot(kind="bar", figsize=(15,6))
    plt.title("Average Topic Share by Trader Label")
    plt.show()


#Plots based on timing statistics
def timing_plots():
    
    #get columns related to trade timings
    timing_cols = ['mean_time', 'std_time', 'mean_time_vw', 'std_time_vw']
    
    #Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Distributions of Timing Statistics')
    axes = axes.flatten()
    for i, col in enumerate(timing_cols):
        sns.histplot(data[col], bins=100, ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    plt.show()


#Correlation information
def corr():
    
    #Compute correlation data
    corr = data.select_dtypes(include=["number"]).corr()

    #Plot correlation heatmap
    plt.figure(figsize=(24,16))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Trader Features")
    plt.show()

    #Get metrics having strongest positive/negative correlation with PnL
    corr_with_pnl = corr["trader_pnl"].sort_values(ascending=False)
    print("Strongest positive correlation with PnL:")
    for col, corr_val in corr_with_pnl.head(11).items():
        if(col != "trader_pnl"):
            print(col, corr_val)
    
    print("\nStrongest negative correlation with PnL:")
    for col, corr_val in corr_with_pnl.tail(10).items():
        print(col, corr_val)

    #Method to get pairs of metrics with high correlation
    def high_corr_pairs(corr_matrix, threshold=0.8):
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        return sorted(pairs, key=lambda x: -abs(x[2]))

    #Determine pairs of metrics with high correlation
    pairs = high_corr_pairs(corr, threshold=0.8)
    print("\nPairs of metrics with high correlation:")
    for p in pairs:
        print(f"{p[0]} and {p[1]}: corr = {p[2]:.2f}")


#plots for metrics by trader label
def label_plots():
    
    #select key columns to plot
    plot_cols = ["trader_pnl", "transactions_per_day", "volume_per_day", "markets_per_day", 
                 "price_levels_consumed", "mean_delta", "mean_time", "trader_ppv", "mean_tx_value"]

    #generate plots
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'Box Plots by Trader Label')
    axes = axes.flatten()
    for i, col in enumerate(plot_cols):
        sns.boxplot(x="trader_label", y=col, data=data, order=['awful', 'bad', 'good', 'sharp'], showfliers=False, ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    plt.show()


#clear terminal
os.system('cls')

#load data from parquet file to Pandas DataFrame
data = pd.read_parquet('C:/Users/adity/Downloads/Important/VSCode/SmithInvestmentFund/SIFdata.parquet')

#call methods
explore_data()
insights()
simple_plots()
topic_plots()
timing_plots()
corr()
label_plots()