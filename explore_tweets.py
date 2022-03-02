import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from scipy import stats
import nltk

def plot_stacked_all(word_counts):
    '''
    Takes in word_counts df with an 'All' column and produces stacked bar chart for each category for top num_top words in all classes
    '''
    (word_counts.sort_values('All', ascending=False)
     .head(20).sort_values('All', ascending=True)
     .apply(lambda row: row/row['All'], axis = 1)
     .drop(columns = 'All')
     #.sort_values(by = 'DFW')
     .plot.barh(stacked = True, width = 1, ec = 'lightgrey')
    )
    plt.title('Proportion of Total Count for Each City for 20 Most-Commonly Occuring Words')
    plt.legend(bbox_to_anchor= (1.03,1))
    plt.xlabel('Proportion')
    plt.ylabel('Word')
    plt.show()

def plot_stacked_bar(word_counts, category, num_top = 20, cmap = None):
    '''
    Takes in word_counts df with an 'All' column and produces stacked bar chart for each category for top num_top words specified by category argument
    '''
    plt.figure(figsize=(16, 9))
    plt.rc('font', size=16)
    col_list = list(word_counts.columns)
    ordered_cols = []
    ordered_cols.append(category)
    for col in col_list:
      if col != category:
        ordered_cols.append(col)
    word_counts = word_counts[ordered_cols]
    (word_counts.sort_values(by=category, ascending=False)
     .head(num_top)
     .apply(lambda row: row / row['All'], axis=1)
     .drop(columns='All')
     .sort_values(by=category)
     .plot.barh(stacked=True, width=1, ec='lightgrey', cmap = cmap, alpha = 1))
    plt.legend(bbox_to_anchor= (1.03,1))
    plt.title(f'Proportions of Most Commonly-Occuring {num_top} {category} Words\n')
    plt.xlabel('Proportion')
    plt.ylabel('Word')
    # make tick labels display as percentages
    # plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter('{:.0%}'.format))
    plt.show()

def bigram_count(words_list, top_num = 20, city_name = None):
    '''
    This function takes in a words_list
    Creates bigrams
    Plots the counts on a bar chart 
    Optional arguements to change customization
    - top_num: default 20, shows most common number of bigrams
    '''
    
    # create bigrams
    ngrams = pd.Series(nltk.bigrams(words_list.split())).value_counts().head(top_num)

    # plot bigrams on left subplot
    ngrams.sort_values(ascending = True).plot.barh(alpha = .7, width = .9)
    plt.title(f'Top {top_num} Bigrams: {city_name}')
    plt.show()

def trigram_count(words_list, top_num = 20, city_name = None):
    '''
    This function takes in a words_list
    Creates trigrams
    Plots the counts on a bar chart 
    Optional arguements to change customization
    - top_num: default 10, shows most common number of trigrams
    '''

    # create bigrams
    ngrams = pd.Series(nltk.trigrams(words_list.split())).value_counts().head(top_num)

    
    # plot bigrams on left subplot
    ngrams.sort_values(ascending = True).plot.barh(alpha = .7, width = .9)
    plt.title(f'Top {top_num} Trigrams: {city_name}')
    plt.show()