#!/usr/bin/env python
# coding: utf-8

# 
# # Project: Investigate TMDB movie data
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# # <a id='intro'></a>
# ## Introduction
# 
# For this Data Analyst project, I selected the TMDb movie dataset from kaggle to investigate. According to kaggle introduction page, the data contains information that are provided from The Movie Database (TMDb). It collects 5000+ movies and their rating and basic move information, including user ratings and revenue data.
# The potiental problem that can be discussed in the dataset:
# 
# ### The potiental problem that can be discussed in the dataset:
# 
# Accroding Kaggle data overview, the dataset provides some metrics that measure how successful these movies are. These metrics include popularity, revenue and vote average. It also contains some basic information corresponding to the movie like cast, director, keywords, runtime, genres, etc. Any of the basic information can be a key to a success movie. More specificly, these factors can be classified to two categrories as follows:
# Metrics for Evaluating the Success Movie
# 
# #### Metrics for Evaluating the Success Movie
#     popularity
#     revenue
#     vote average score
# 
# #### Potential Key to Affect the Success of a Movie
# 
#     Budget
#     Cast
#     Director
#     Tagline
#     Keywords
#     Runtime
#     Genres
#     Production Companies
#     Release Date
#     Vote Average
# 
# Since the dataset is featured with the rating of movies as mentioned above, it contains plentiful information for exploring the properties that are associated with successful movies, which can be defined by high popularity, high revenue and high rating score movies. Besides, the dataset also contains the movie released year, so it also can let us to explore the trend in these movie metrics. Therefore, the qestions I am going to explore are including three parts:
# 
# #### Research Part 1: General Explore
# 
#     Question 1: Popularity Over Years
#     Question 2: The distribution of revenue in different popularity levels in recent five years.
#     Question 3: The distribution of revenue in different score rating levels in recent five years.
# 
# #### Research Part 2 : Find the Properties are Associated with Successful Movies
# 
#     Question 1: What kinds of properties are associated with movies that have high popularity?
#     Question 2: What kinds of properties are associated with movies that have high voting score?
#     
# #### Research Part 3 Top Keywords and Genres Trends by Generation
# 
#     Question 1: Number of movie released year by year
#     Question 2: Keywords Trends by Generation
#     Question 3: Genres Trends by Generation
# 

# <a id='wrangling'></a>
# ## Data Wrangling
# 
# ### General Dataset Properties
# 
# #### First, let's look what the dataset looks like for preceeding to investigate.
# 

# In[2]:


# Import statements for all of the packages that I plan to use.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')


# Load the data and print out a few lines. Perform operations to inspect data
# Types and look for instances of missing or possibly errant data.

# In[3]:


df = pd.read_csv("https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dd1c4c_tmdb-movies/tmdb-movies.csv")
df.head(1)


# In[4]:



#see the column info and null values in the dataset
df.info()


#     From the table above, there are totally 10866 entries and total 21 columns. And there exists some null value in the cast, director, overview and genres columns. But some columns are with a lot of null value rows like homepage, tagline, keywords and production_companies, especially the homepage and tagline columns are even not necessary for answering the question, so I decide to drop both of the columns on the stage.
# 
#     Let's see some descriptive statistics for the data set.

# In[5]:


df.describe()


# In[6]:


#Let's take a look at some zero budget and revenue data.

df_budget_zero = df.query('budget == 0')
df_budget_zero.head(3)


# In[7]:



df_revenue_zero = df.query('revenue == 0')
df_revenue_zero.head(3)


# In[8]:


df_budget_0count =  df.groupby('budget').count()['id']
df_budget_0count.head(2)


# In[9]:


df_revenue_0count =  df.groupby('revenue').count()['id']
df_revenue_0count.head(2)


# In[10]:


df_runtime_0count =  df.groupby('runtime').count()['id']
df_runtime_0count.head(2)


# 
# Cleaning Decision Summary
# 
#    1. Drop unnecessary columns for answering those questions : homepage, tagline, imdb_id, overview,budget_adj, revenue_adj.
#    2. Drop duplicates.
#    3. Drop null values columns that with small quantity of nulls : cast, director, and genres.
#    4. Replace zero values with null values in the budget and revenue column.
#    5. Drop zero values columns that with small quantity of zeros : runtime.
# 
# 
# 
# ### Data Cleaning 
# 
# First, according to the previous decision, let's drop unncessary columns : imdb_id, homepage, tagline, overview.

# After discussing the structure of the data and any problems that need to be
# cleaned, perform those cleaning steps in the second part of this section.
# Drop extraneous columns

# In[11]:


col = ['imdb_id', 'homepage', 'tagline', 'overview', 'budget_adj', 'revenue_adj']
df.drop(col, axis=1, inplace=True)


# In[17]:


# see if these columns are dropped.
df.head(1)


# In[12]:


df.drop_duplicates(inplace=True)


# In[20]:


cal2 = ['cast', 'director', 'genres']
df.dropna(subset = cal2, how='any', inplace=True)


# In[13]:


df.isnull().sum()


# In[14]:


#Then, replace zero values with null values in the budget and revenue column.
df['budget'] = df['budget'].replace(0, np.NaN)
df['revenue'] = df['revenue'].replace(0, np.NaN)
# see if nulls are added in budget and revenue columns
df.info()


# In[15]:


#Finally, drop columns with small quantity of zero values : runtime.
df.query('runtime != 0', inplace=True)
df.query('runtime == 0')


# 
# Cleaning Result Summary:
# 
#     From the table bellow, we can see that the data in each column are almost clear without too many null values. And my clearning goal is also to keep the data integrity from the original one. Although there are some null values in `keywords` and `production companies` columns, it is still useful for analysis, and in fact the number of their null values are not very huge, so I just kept both of them. The data now with 10703 entries and 17 columns.
# 

# In[16]:


df.info()


# In[17]:


#And from the table bellow, after transfer all zero values to null values in `budget` and `revenue` data, we can see that both the distribution of budget and revenue are much better, without too concentrate on the zero value or small values. And after deleting the zero values of runtime, we can see the minimum value of runtime is more reasonable.
df.describe()


# # <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# 
# ## Research Part 1: General Explore
# 
#     Question 1: Popularity Over Years.
#     Question 2: The distribution of popularity in different revenue levels in recent five years.
#     Question 3: The distribution of score rating in different revenue levels in recent five years.
# 
# ## Research Part 2 : Find the Properties are Associated with Successful Movies
# 
#     Question 1: What kinds of properties are associated with movies that have high popularity?
#     Question 2: What kinds of properties are associated with movies that have high voting score?
#     
# ## Research Part 3 Top Keywords and Genres Trends by Generation
# 
#     Question 1: Number of movie released year by year.
#     Question 2: Keywords Trends by Generation.
#     Question 3: Genres Trends by Generation.
#     

# # Research Question 1 (General Explore)
# 
# 
# 
# ## Question 1: Popularity Over Years
# 
# To explore this question, let's take a look of the dataset

# In[18]:


df.head(2)


# In[19]:


# computing the mean for popularity
p_mean = df.groupby('release_year').mean()['popularity']
p_mean.tail()


# In[20]:


# computing the median for popularity
p_median = df.groupby('release_year').median()['popularity']
p_median.tail()


# In[21]:


# building the index location for x-axis
index_mean = p_mean.index
index_median = p_median.index


# In[22]:


#set style
sns.set_style('whitegrid')
#set x, y axis data
#x1, y1 for mean data; x2, y2 for median data
x1, y1 = index_mean, p_mean
x2, y2 = index_median, p_median
#set size
plt.figure(figsize=(9, 4))
#plot line chart for mean and median
plt.plot(x1, y1, color = 'g', label = 'mean')
plt.plot(x2, y2, color = 'r', label = 'median')
#set title and labels
plt.title('Popularity Over Years')
plt.xlabel('Year')
plt.ylabel('Popularity');
#set legend
plt.legend(loc='upper left')


# From the figure above, we can see that the trend of popularity mean is upward year to year, and the peak is in the 2015, while the trend of popularity median is slightly smoother in recent years. We still can conclude that on average, popularity over years is going up in recent years. The trend is reasonable due to the eaiser access of movie information nowadays. Moreover, in the Internet age, people can easily search and gether movie information, even watch the content through different sources. Maybe it is such the backgroud that boost the movie popularity metrics.

# ## Question 2: The distribution of popularity in different revenue levels in recent five years.
# 
# The movies popularity is growing up in recently years, but how about the popularity in different revenue levels? will popularity be more higher in high revenue level? In this research I don't dicuss the revenue trend since it is affected by many factors like inflation. Although the database contains the adjusted data but I just want the analysis be more simple. Moreever, if I find out the movie revenue trend is growing up, it still can't infer that the trend up is related to popularity just by looking the revenue trend line chart year by yaer.
# 
# Hence, it leads me that what to find out the distribution of popularity look like in terms of different revenue levels. Which means I can see the what popularity with which revenue levels. Dou to the revenue data contains wide range, to be more specific, I divided the revenue data into five levels: Low', 'Medium', 'Moderately High', 'High' according to their quartile. Also I choose the recent five years data to dicuss in order to focus on the current data feature.
# 
# #### For the further usage of the level-diveded procedure with quartile, I build a cut_into_quantile function to divided data into four levels according to their quartile: 'Low', 'Medium', 'Moderately High', 'High'.
# 
# #### The cut_into_quantile function- general use.
# 

# In[24]:


# quartile function
def cut_into_quantile(dfname ,column_name):
# find quartile, max and min values
    min_value = dfname[column_name].min()
    first_quantile = dfname[column_name].describe()[4]
    second_quantile = dfname[column_name].describe()[5]
    third_quantile = dfname[column_name].describe()[6]
    max_value = dfname[column_name].max()
# Bin edges that will be used to "cut" the data into groups
    bin_edges = [ min_value, first_quantile, second_quantile, third_quantile, max_value]
# Labels for the four budget level groups
    bin_names = [ 'Low', 'Medium', 'Moderately High', 'High'] 
# Creates budget_levels column
    name = '{}_levels'.format(column_name)
    dfname[name] = pd.cut(dfname[column_name], bin_edges, labels=bin_names, include_lowest = True)
    return dfname


# ####  Since I want to explore the data by year to year in the question, so to avoide the different level affecting among each year's revenue, I divide revenue levels by with each year's revenue quartile.
# 
# 

# In[25]:


#choose the recent five years 
dfyear =[2011,2012,2013,2014,2015]
#creat a empty dataframe,df_q2
df_q2 = pd.DataFrame()

#for each year, do the following procedure
for year in dfyear:
    dfn = df.query('release_year == "%s"' % year) # first filter dataframe with the selected year 
    dfn2 = cut_into_quantile(dfn,'revenue') #apply the cut_into_quantile with the selected frame, store it to dfn2 
    df_q2 = df_q2.append(dfn2) #append dfn2 to df_q2
df_q2.info()


# Now we can see we create a revenue_levels column with the same rows with revenue.
# 
# Then use the dataset to explore the popularity in each level each year.
# 

# In[26]:


# grouping the dataframe I created above with each revenue levels in each year, finding the popularity meadian
dfq2_summary = df_q2.groupby(['release_year','revenue_levels']).median()
dfq2_summary.tail(8)


# In[27]:


# Setting the positions and width for the bars
pos = list(range(len(dfq2_summary.query('revenue_levels =="Low"'))))
width = 0.2 

# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))

# Create a bar with Low data, in position pos,
plt.bar(pos, 
        #using 'Low' data,
        dfq2_summary.query('revenue_levels =="Low"')['popularity'], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#EE3224', 
        # with label Low
        label= 'Low') 

# Create a bar with Medium data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos], 
        #using Medium data,
        dfq2_summary.query('revenue_levels =="Medium"')['popularity'],
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#F78F1E', 
        # with label Medium
        label='Medium') 

# Create a bar with Moderately High data,
# in position pos + some width buffer,
plt.bar([p + width*2 for p in pos], 
        #using Moderately High data,
        dfq2_summary.query('revenue_levels =="Moderately High"')['popularity'], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#FFC222', 
        # with label Moderately High
        label='Moderately High') 

# Create a bar with High data,
# in position pos + some width buffer,
plt.bar([p + width*3 for p in pos], 
        #using High data,
        dfq2_summary.query('revenue_levels =="High"')['popularity'], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#4fb427', 
        # with label High
        label='High')

ax.set_ylabel('popularity')

ax.set_title('Popularity in Different Revenue Levels in Recent Five Years')

ax.set_xticks([p + 1.5 * width for p in pos])

ax.set_xticklabels([2011,2012,2013,2014,2015])

plt.legend( loc='upper left')
plt.grid()
plt.show()


# #### We can see that movies with higher revenue level are with higher popularity in recent five years.
# 
# We can see that revenue level has postive relation with popularity. The result is reasonable since it makes me think of if movie producer wants to make high revenue movies, the first thing they always is to promote it and make it popular. So according the result from the previous question, I infer that a high revenue movie is always with a higher popularity than movies with lower revenue levels. So if we define success of a movie is it's revenue, one property it has is the high popularity.
# 
# #### But what about the score rating distribution in different revenue levels of movies? Does high revenue level movie has the property of high score rating? Let's explore on the next question.
# 

# ### Question 3: The distribution of revenue in different score rating levels in recent five years.
# 
# Use the same procedure on Question 2 to explore this question.
# 

# In[28]:


# group the dataframe we created above with each revenue levels in each year, find the vote_average mean
dfq2_summary = df_q2.groupby(['release_year','revenue_levels']).mean()
dfq2_summary.tail(4)


# In[38]:


# group the dataframe we created above with each revenue levels in each year, find the vote_average mean
dfq2_summary = df_q2.groupby(['release_year','revenue_levels']).mean()
dfq2_summary.tail(4)


# In[29]:


# Setting the positions and width for the bars
pos = list(range(len(dfq2_summary.query('revenue_levels =="Low"'))))
width = 0.2 

# Plotting the bars
fig, ax = plt.subplots(figsize=(12,3))

# Create a bar with Low data, in position pos,
plt.bar(pos, 
        #using 'Low' data,
        dfq2_summary.query('revenue_levels =="Low"')['vote_average'], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#EE3224', 
        # with label Low
        label= 'Low') 

# Create a bar with Medium data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos], 
        #using Medium data,
        dfq2_summary.query('revenue_levels =="Medium"')['vote_average'],
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#F78F1E', 
        # with label Medium
        label='Medium') 

# Create a bar with Moderately High data,
# in position pos + some width buffer,
plt.bar([p + width*2 for p in pos], 
        #using Moderately High data,
        dfq2_summary.query('revenue_levels =="Moderately High"')['vote_average'], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#FFC222', 
        # with label Moderately High
        label='Moderately High') 

# Create a bar with High data,
# in position pos + some width buffer,
plt.bar([p + width*3 for p in pos], 
        #using High data,
        dfq2_summary.query('revenue_levels =="High"')['vote_average'], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#4fb427', 
        # with label High
        label='High')

ax.set_ylabel('vote average')

ax.set_title('Vote Average Score in Different Revenue Levels in Recent Five Years')

ax.set_xticks([p + 1.5 * width for p in pos])

ax.set_xticklabels([2011,2012,2013,2014,2015])

plt.ylim(3, 10)

plt.legend(loc='upper left')
plt.grid()
plt.show()


# From the chart above, we can see that there is no big difference of movie rating between each revenue level. So it can be concluded that the high revenue movies don't have the significant high score rating.
# 

# 
# ### Part 1 Question Explore Summary
# 
#      1.Movie popularity trend is growing from 1960, I infer that it is with the background that nowadays movie information and rating system are more accessible by Internet with different channels.
#      2.Movies with higher revenue level are with higher popularity in recent five years. In other words, a high revenue movie always with a higher popularity. So on the next part, I will explore: What's properties that are associated with high popularity movies?
#      3.Movies with higher revenue level don't have the significant high score rating than other revenue levels in recent five years. So on the next part, I will explore: What's properties that are associated with high rating movies?

# ### Research Question 2  (Find the Properties are Associated with Successful Movies)
# 
# 
#    #### Question 1: What kinds of properties are associated with movies that have high popularity?
#         What's the budget level movie are associated with movies that have high popularity?
#         What's the runtime level are associated with movies that have high popularity on average?
#         What's casts, directors, keywords, genres and production companies are associated with high popularity?
# 
#    #### Question 2: What kinds of properties are associated with movies that have high voting score?
#         What's the budget level are associated with movies that have high voting score?
#         What's the runtime level are associated with movies that have high voting score?
#         What's the directors, keywords, genres are associated with voting score?
# 
# ### Function and research sample prepare
# 
# In the dataset, the potential properties associated with movies can be runtime, budget, cast, director, keywords, genres, production companies. These data are including two types: quantitative data and categorical data. Both runtime and budget data are quantitative data; the others are categorical data.
# 
#    For quantitative data, since the data is quantitative, I can devide the data into various levels and find the properties in all range of movies success, I choose to use the whole dataset and then divided runtime and budget into four levels according to their quartile: 'Low', 'Medium', 'Moderately High', 'High' in all time range. And then find out what's the runtime and budget level with higher degree of movies popularity/voting score.
# 
#    For categorical data, which are cast, director, keywords and genres, since we are not necessary to discuss all the range of of movies success(which is also difficult to dicuss), I just focus on the high popularity or high rating, so I filter the top 100 popular/ high voting score movies data in each year, and then count the number of occurrences in every category every year to find their properties. Forthermore, in case that the top frequent occurrences are also appeared in the worst popular/ high voting score movies, I also filter the worst 100 popular/ high voting score movies in every year and then compare the result to top 100's. If the top frequent occurrences also appear in the worst movies, I am going to include these factors as properties associated with top movies as well as worst movies. Besides, these data are contain the pipe (|) characters so first I have to spilt them. </b>
# 
# ### The function is the same I ued in the Part 1 Question. So I just past it again below.
# 

# #### A)The cut_into_quantile function- general use.
# 
# The function is the same I ued in the Part 1 Question. So I just past it again below.

# In[30]:


# quartile function
def cut_into_quantile(dfname ,column_name):
# find quartile, max and min values
    min_value = dfname[column_name].min()
    first_quantile = dfname[column_name].describe()[4]
    second_quantile = dfname[column_name].describe()[5]
    third_quantile = dfname[column_name].describe()[6]
    max_value = dfname[column_name].max()
# Bin edges that will be used to "cut" the data into groups
    bin_edges = [ min_value, first_quantile, second_quantile, third_quantile, max_value]
# Labels for the four budget level groups
    bin_names = [ 'Low', 'Medium', 'Moderately High', 'High'] 
# Creates budget_levels column
    name = '{}_levels'.format(column_name)
    dfname[name] = pd.cut(dfname[column_name], bin_edges, labels=bin_names, include_lowest = True)
    return dfname


# #### B) Split pipe (|) characters and then count their number of appeared times, then find the top three factor.

# In[31]:


# split pipe characters and count their number of appeared times
#argument:dataframe_col is the target dataframe&column; num is the number of the top factor
def find_top(dataframe_col, num=3):
    # split the characters in the input column 
    #and make it to a list
    alist = dataframe_col.str.cat(sep='|').split('|')
    #transfer it to a dataframe
    new = pd.DataFrame({'top' :alist})
    #count their number of appeared times and
    #choose the top3
    top = new['top'].value_counts().head(num)
    return top


# 
# ### B. Sample prepare-- Filter Top 100 and Worst 100 movies in each year as the research sample.
# 
# #### A) Select Top 100 popular movies in every year.
# 

# In[32]:


# Select Top 100 popular movies.
# fisrt sort it by release year ascending and popularity descending
df_top_p = df.sort_values(['release_year','popularity'], ascending=[True, False])
#group by year and choose the top 100 high
df_top_p = df_top_p.groupby('release_year').head(100).reset_index(drop=True)
#check, it must start from 1960, and with high popularity to low
df_top_p.head(2)


# #### B) Select Top 100 high revenue movies in every year.

# In[89]:


# Select Top 100 high revenue movies.
# fisrt sort it by release year ascending and revenue descending
df_top_r = df.sort_values(['release_year','revenue'], ascending=[True, False])
#group by year and choose the top 100 high
df_top_r = df_top_r.groupby('release_year').head(100).reset_index(drop=True)
#check, it must start from 1960, and with high revenue to low
df_top_r.head(2)


# #### C) Select Top 100 high score rating movies in every year.

# In[34]:


# Select Top 100 high scorer ating movies.
# fisrt sort it by release year ascending and high scorer ating descending
df_top_s = df.sort_values(['release_year','vote_average'], ascending=[True, False])
#group by year and choose the top 100 high
df_top_s = df_top_s.groupby('release_year').head(100).reset_index(drop=True)
#check, it must start from 1960, and with high scorer ating to low
df_top_s.head(2)


# #### D) To compare to results, I also create three subdataset for the last 100 movies.

# In[35]:


# the last 100 popular movies in every year
df_low_p = df.sort_values(['release_year','popularity'], ascending=[True, True])
df_low_p = df_low_p.groupby('release_year').head(100).reset_index(drop=True)
# the last 100 high revenue movies in every year
df_low_r = df.sort_values(['release_year','revenue'], ascending=[True, True])
df_low_r = df_low_r.groupby('release_year').head(100).reset_index(drop=True)
# the last 100 score rating movies in every year
df_low_s = df.sort_values(['release_year','vote_average'], ascending=[True, True])
df_low_s = df_low_s.groupby('release_year').head(100).reset_index(drop=True)


# ## Question 1: What kinds of properties are associated with movies that have high popularity?
# 
#    1. What's the budget level movie are associated with movies that have high popularity?
#    2. What's the runtime level are associated with movies that have high popularity on average?
#    3. What's casts, directors, keywords, genres and production companies are associated with high popularity? </b>
# 
# ## 1.1 What's the budget level movie are associated with movies that have high popularity?
# 
# First, divided budget data into four levels with it's quartile: 'Low', 'Medium', 'Moderately High', 'High' and create a level column.
# 

# In[36]:


# use cut_into_quantile function to build a level column
df = cut_into_quantile(df,'budget')
df.head(1)


# From the table above, I built a budget_levels columns.

# In[37]:


# Find the mean and median popularity of each level with groupby
result_mean = df.groupby('budget_levels')['popularity'].mean()
result_mean


# In[38]:


result_median = df.groupby('budget_levels')['popularity'].median()
result_median


# In[39]:


#Visualing
# the x locations for the groups
ind = np.arange(len(result_mean))  
# the width of the bars
width = 0.5       
ind


# In[40]:


# plot bars
#set style
sns.set_style('darkgrid')
bars = plt.bar(ind, result_mean, width, color='g', alpha=.7, label='mean')

# title and labels
plt.ylabel('popularity')
plt.xlabel('budget levels')
plt.title('Popularity with Budget Levels')
locations = ind  # xtick locations，345...
labels = result_median.index  
plt.xticks(locations, labels)
# legend
plt.legend()


# 
# From the figure above, we can see that movies with higher popularity are with higher budget level. The result is reasonable since movies with higher popularity may has a higher promoting advertising cost. And with the high promotion level people always have more probability to get know these movies.
# 
# ## 1.2 What's the runtime level are associated with movies that have high popularity on average?
# 
# Divided runtime data into four levels with it's quartile: 'Short', 'Medium', 'Moderately Long', 'Long'.
# 

# In[42]:


df = cut_into_quantile(df,'runtime')
df.head(1)


# In[43]:


# Find the mean popularity of each level with groupby
result_mean = df.groupby('runtime_levels')['popularity'].mean()
result_mean


# In[44]:


# Find the median popularity of each level with groupby
result_median = df.groupby('runtime_levels')['popularity'].median()
result_median


# In[45]:


ind = np.arange(len(result_median))  # the x locations for the groups
width = 0.5       # the width of the bars


# In[46]:


# plot bars
bars = plt.bar(ind, result_median, width, color='#1ea2bc', alpha=.7, label='median')

# title and labels
plt.ylabel('popularity')
plt.xlabel('runtime levels')
plt.title('Popularity with Runtime Levels')
locations = ind  # xtick locations，345...
labels = result_median.index  
plt.xticks(locations, labels)
# legend
plt.legend()


# We can see that the higher popularity movies has longer run time.

# 
# ## 1.3 What's casts, directors, keywords, genres and production companies are associated with high popularity?
# 
# #### First, choose the dataset-df_top_p. It is the dataframe about top 100 popular movies in each year.
# 

# In[50]:


df_top_p.head(2)


# #### Then, find the three highest occurrences in each category among the top 100 popular movies. And store the result table into variables in order to create a summary table.

# In[51]:


# find top three cast
a = find_top(df_top_p.cast)
# find top three director
b = find_top(df_top_p.director)
# find top three keywords
c = find_top(df_top_p.keywords)
# find top three genres
d = find_top(df_top_p.genres)
# find top three production companies
e = find_top(df_top_p.production_companies)


# #### Use the result above to create a summary dataframe.

# In[52]:


df_popular = pd.DataFrame({'popular_cast': a.index, 'popular_director': b.index, 'popular_keywords': c.index, 'popular_genres': d.index, 'popular_producer': e.index})
df_popular


# #### Finally, find the three highest occurrences in each category among the 100 unpopular movies.

# In[54]:


# call the dataset wiht the 100 unpopular movies in each year
df_low_p.head(2)


# In[55]:


# find top three cast among the among the 100 unpopular movies
na = find_top(df_low_p.cast)
# find top three director among the among the 100 unpopular movies
nb = find_top(df_low_p.director)
# find top three keywords among the among the 100 unpopular movies
nc = find_top(df_low_p.keywords)
# find top three genres among the among the 100 unpopular movies
nd = find_top(df_low_p.genres)
# find top three production companiess among the among the 100 unpopular movies
ne = find_top(df_low_p.production_companies)


# In[56]:


df_unpopular = pd.DataFrame({'unpopular_cast': na.index, 'unpopular_director': nb.index, 'unpopular_keywords': nc.index, 'unpopular_genres': nd.index, 'unpopular_producer': ne.index})
df_unpopular


# #### Now, we get the two table that list the properties occurred the most among the top 100 popular movies each year, among the top 100 unpopular movies each year respectively.
# 
# #### Now we can campare the two tables and find out What's casts, directors, keywords, genres and production companies are associated with high popularity.
# 

# In[78]:


# compare
df_popular


# From the tabbles above, we can find that cast Michael Caine is appeared in both popular and unpopular movies; director Woody Allen and Clint Eastwood are appeared in both popular and unpopular movies; all three genres Drama, Comedy, Thriller are appeared in both popular and unpopular movies; sex is appeared in both popular and unpopular movies; all three producer Universal Pictures, Warner Bros, Paramount Pictures are appeared in both popular and unpopular movies. The summary are as follows:
# 
#     Cast associated with high popularity movies: Robert De Niro and Bruce Willis. It's really reasonable because I have seen a lot of promoted movies content which are performed by them in my country. On average I think they do have the huge popularity in past years!
#     Director associated with high popularity movies: Steven Spielberg. It's no doubt that he got the first place since he has won so many awards and honors for his high quality and popular work!
#     Both of the most popular and unpopular movies are associated three mainly genres: Drama, Comedy, and Thriller. I just can infer that these genres are common in the movie industry.
#     Keywords associated with high popularity movies: based on novel and dystopia. It' also no doubt it comes out the result. Especially the based on novel movies, since nowadays tons of movies are made based on novel like Harry Potter, The Hunger Games etc, and they were also famous in my country.
#     Producer associated with high popularity movies and unpopularity movies: Warner Bros., Universal Pictures and  Paramount Pictures. The three giants of movie indusry did produce such a various quality movies!
# 
# 
# ### Question 2: What kinds of properties are associated with movies that have high voting score?
# 
#    1. What's the budget level are associated with movies that have high voting score?
#    2. What's the runtime level are associated with movies that have high voting score?
#    3. What's the directors, keywords, genres are associated with voting score? </b>
# 
# Use the same procedure with Research 2, Question 1 to answer these questions.
# 
# 
# ## 2.1 What's the budget level are associated with movies that have high voting score?
# 
# First, use the dataframe with budget level I have created in the previous question. Then find the mean and median of vote_average group by different budget level.
# 

# In[57]:


# Find the mean and median voting score of each level with groupby
result_mean = df.groupby('budget_levels')['vote_average'].mean()
result_mean


# In[58]:


result_median = df.groupby('budget_levels')['vote_average'].median()
result_median


# #### Let's use the mean table above to visualize it.

# In[59]:


# plot bars
#set style
sns.set_style('darkgrid')
ind = np.arange(len(result_mean))  # the x locations for the groups
width = 0.5       # the width of the bars

# plot bars
plt.subplots(figsize=(8, 6))
bars = plt.bar(ind, result_median, width, color='y', alpha=.7, label='mean')

# title and labels
plt.ylabel('rating')
plt.xlabel('budget levels')
plt.title('Rating with Budget Levels')
locations = ind  # xtick locations，345...
labels = result_median.index  
plt.xticks(locations, labels)
# legend
plt.legend( loc='upper left')


# We can see that there is no big difference in average voting score at different budget levels. So from the result, maybe high budget of a movie is not necessary to a good quality of movie!
# 
# 
# ## 2.2 What's the runtime level are associated with movies that have high voting score?
# 
# First, use the dataframe with runtime level I have created in the previous question. Then find the mean and median of vote_average group by different runtime level.
# 

# In[60]:


# Find the mean popularity of each level with groupby
result_mean = df.groupby('runtime_levels')['vote_average'].mean()
result_mean


# In[61]:


result_median = df.groupby('runtime_levels')['vote_average'].median()
result_median


# In[62]:


sns.set_style('darkgrid')
ind = np.arange(len(result_mean))  # the x locations for the groups
width = 0.5       # the width of the bars

# plot bars
bars = plt.bar(ind, result_median, width, color='g', alpha=.7, label='mean')

# title and labels
plt.ylabel('rating')
plt.xlabel('runtime levels')
plt.title('Rating with Runtime Levels')
locations = ind  # xtick locations，345...
labels = result_median.index  
plt.xticks(locations, labels)
# legend
plt.legend()


# We can see that there is no big difference in average voting score in different runtime levels. So from the result, maybe long runtime of a movie is not necessary to a good quality of movie!
# 
# 
# ## 2.3 What's the directors, keywords, genres are associated with voting score?
# 
# First, choose the dataset-df_top_s. It is the dateframe about top 100 high voting score movies in each year.
# 

# In[64]:


df_top_s.head(2)


# #### Then, find the three highest occurrences in each category among the top 100 high voting score movies. And store the result table into variables in order to create a summary table.

# In[65]:


# find top three director
a = find_top(df_top_s.director)
# find top three keywords
b = find_top(df_top_s.keywords)
# find top three genres
c = find_top(df_top_s.genres)


# #### Use the result above to create a summary table.

# In[66]:


#create a summary dataframe.
df_high_score = pd.DataFrame({'high_score_director': a.index, 'high_score_keywords': b.index, 'high_score_genres': c.index})
df_high_score


# #### Finally, find the three highest occurrences in each category of the worst 100 rating score movies.

# In[67]:


# call the dataset wiht the 100 low rating movies in each year
df_low_s.head(2)


# In[68]:


# find top three director among the among the 100 low rating movies
na = find_top(df_low_s.director)
# find top three keywords among the among the 100 low rating movies
nb = find_top(df_low_s.keywords)
# find top three genres among the among the 100 low rating movies
nc = find_top(df_low_s.genres)


# Use the result above to create a summary table.

# In[69]:


df_low_score = pd.DataFrame({'low_score_director': na.index, 'low_score_keywords': nb.index, 'low_score_genres': nc.index})
df_low_score


# In[70]:


# compare
df_high_score


# #### After summing up both tables above, we can find that:
# 
#     Martin Scorsese and Clint Eastwood have made top quality movies on average over the past years from 1960.
#     The top quality movies have the keywords with based on novel and woman director over the past years from 1960. The based on novel keyword are also with the top popular movies, but the result of woman director amazed me! 
#     
#     
# ## Part 2 Question Explore Summary
# 
#     For the properties are associated with high popularity movies, they are high budget levels and longer run time. And cast associated with high popularity movies are Robert De Niro and Bruce Willis; director associated with high popularity movies are Steven Spielberg; genres associated with high popularity movies are drama, comedy, and thriller but they also appeared in the most unpopular movies; keywords associated with high popularity movies are based on novel and dystopia; producer associated with high popularity movies are Warner Bros., Universal Pictures and Paramount Pictures, but they are also appeared in the most unpopular movies.
# 
#     Each level in both runtime and budget don't have obvious different high rating score. In other words, the low budget level or the low budget may still have a high rating. And Martin Scorsese and Clint Eastwood have made top quality movies on average over the past years from 1960; the top quality movies have the keywords with based on novel and woman director over the past years from 1960.
# 
# 
# 

# ## Research Part 3 Top Keywords and Genres Trends by Generation
# 
#     Question 1: Number of movie released year by year
#     Question 2: Keywords Trends by Generation
#     Question 3: Genres Trends by Generation </b>
# 
# #### In question 1, I am going to find out the number of movie released year by year.
# 
# In question 2 and 3, I am going to find out what's the keyword and genre appeared most by generation? To do this:
# 
#     Step one: group the dataframe into five generations: 1960s, 1970s, 1980s, 1990s and 2000s
#     Step two: use the find_top function to count out the most appeared keyword and genre in each generation dataframe. 

# ## Question 1: Number of movie released year by year
# 
# First, use group by release year and count the number of movie released in each year.
# 

# In[80]:


movie_count = df.groupby('release_year').count()['id']
movie_count.head()


# Then visualize the result.

# In[81]:


#set style
sns.set_style('darkgrid')
#set x, y axis data
# x is movie release year
x = movie_count.index
# y is number of movie released
y = movie_count
#set size
plt.figure(figsize=(10, 5))
#plot line chart 
plt.plot(x, y, color = 'g', label = 'mean')
#set title and labels
plt.title('Number of Movie Released year by year')
plt.xlabel('Year')
plt.ylabel('Number of Movie Released');


# We can see that the number of movie released are increasing year by year. And the it is the accelerated growth since the curve is concave upward.

# ## Question 2: Keywords Trends by Generation
# 
# First, sort the movie release year list to group the dataframe into generation.
# 

# In[83]:


# sort the movie release year list.
dfyear= df.release_year.unique()
dfyear= np.sort(dfyear)
dfyear


# Then, build the generation catagory of 1960s, 1970s, 1980s, 1990s and 2000s.

# In[84]:


# year list of 1960s
y1960s =dfyear[:10]
# year list of 1970s
y1970s =dfyear[10:20]
# year list of 1980s
y1980s =dfyear[20:30]
# year list of 1990s
y1990s = dfyear[30:40]
# year list of afer 2000
y2000 = dfyear[40:]


# #### Then for each generation dataframe, use the find_top to find out the most appeared keywords, then combine this result to a new datafram.

# In[85]:


# year list of each generation
times = [y1960s, y1970s, y1980s, y1990s, y2000]
#generation name
names = ['1960s', '1970s', '1980s', '1990s', 'after2000']
#creat a empty dataframe,df_r3
df_r3 = pd.DataFrame()
index = 0
#for each generation, do the following procedure
for s in times:
    # first filter dataframe with the selected generation, and store it to dfn
    dfn = df[df.release_year.isin(s)] 
    #apply the find_top function with the selected frame, using the result create a dataframe, store it to dfn2 
    dfn2 = pd.DataFrame({'year' :names[index],'top': find_top(dfn.keywords,1)})
     #append dfn2 to df_q2
    df_r3 = df_r3.append(dfn2)
    index +=1
df_r3


# 
# Now, we get the keywords of most filmed movies in each generation. We can see that in 1960s and 1970s, the top keywords was based on novel, which means movies with the keyword based on novel are released most according the dataset. In 1980s, the top keyword was nudity, what a special trend! In 1990s, independent film became the top keyword. And after 2000, the movie with the feature woman director were released most. It's sounds great!
# 

# 
# 
# Now let's visualize the result.
# 

# In[86]:


# Setting the positions
generation = ['1960s', '1970s', '1980s', '1990s', 'after2000']
keywords = df_r3.index
y_pos = np.arange(len(generation))
fig, ax = plt.subplots()
# Setting y1: the keywords number
y1 = df_r3.top
# Setting y2 again to present the right-side y axis labels
y2 = df_r3.top
#plot the bar
ax.barh(y_pos,y1, color = '#007482')
#set the left side y axis ticks position
ax.set_yticks(y_pos)
#set the left side y axis tick label
ax.set_yticklabels(keywords)
#set left side y axis label
ax.set_ylabel('keywords')

#create another axis to present the right-side y axis labels
ax2 = ax.twinx()
#plot the bar
ax2.barh(y_pos,y2, color = '#27a5b4')
#set the right side y axis ticks position
ax2.set_yticks(y_pos)
#set the right side y axis tick label
ax2.set_yticklabels(generation)
#set right side y axis label
ax2.set_ylabel('generation')
#set title
ax.set_title('Keywords Trends by Generation')


# One more thing, we can see that the number of the keywords appeared changes from 16 to 347 by generation, and it is resonable since the trend is consistent with the number of movie released.

# # Question 3: Genres Trends by Generation
# 
# Use the same procedure as Question 2, first use the find_top to find out the most appeared genres, then combine this result to a new datafram.
# 

# In[87]:


# year list of each generation
times = [y1960s, y1970s, y1980s, y1990s, y2000]
#generation name
names = ['1960s', '1970s', '1980s', '1990s', 'after2000']
#creat a empty dataframe,df_r3
df_r3 = pd.DataFrame()
index = 0
#for each generation, do the following procedure
for s in times:
    # first filter dataframe with the selected generation, and store it to dfn
    dfn = df[df.release_year.isin(s)] 
    #apply the find_top function with the selected frame, using the result create a dataframe, store it to dfn2 
    dfn2 = pd.DataFrame({'year' :names[index],'top': find_top(dfn.genres,1)})
     #append dfn2 to df_q2
    df_r3 = df_r3.append(dfn2)
    index +=1
df_r3


# Visualize the result.

# In[88]:


# Setting the positions
generation = ['1960s', '1970s', '1980s', '1990s', 'after2000']
genres = df_r3.index
y_pos = np.arange(len(generation))
fig, ax = plt.subplots()
# Setting y1: the genre number
y1 = df_r3.top
# Setting y2 again to present the right-side y axis labels
y2 = df_r3.top
#plot the bar
ax.barh(y_pos,y1, color = '#007482')
#set the left side y axis ticks position
ax.set_yticks(y_pos)
#set the left side y axis tick label
ax.set_yticklabels(genres)
#set left side y axis label
ax.set_ylabel('genres')

#create another axis to present the right-side y axis labels
ax2 = ax.twinx()
#plot the bar
ax2.barh(y_pos,y2, color = '#27b466')
#set the right side y axis ticks position
ax2.set_yticks(y_pos)
#set the right side y axis tick label
ax2.set_yticklabels(generation)
#set right side y axis label
ax2.set_ylabel('generation')
#set title
ax.set_title('Genres Trends by Generation')


# We can see that the genre Drama are the most filmed in almost all generation. Only the 1980s are dominated by the comedy type.

# # Part 3 Question Explore Summary
# 
#    1. The number of movie released are increasing year by year. And the it is in the accelerated growth trend.
#    2. In 1960s and 1970s, the top keywords was based on novel, which means movies with the keyword based on novel are released most according the dataset. In 1980s, the top keyword was nudity. In 1990s, independent film became the top keyword. And after 2000, the movie with the feature woman director were released most.
#     
#    3. The genre Drama are the most filmed in almost all generation. Only the 1980s are dominated by the comedy type.
# 
# 

# 
# 
# ## Conclusions:
# 
# The goal in the research is primary to explore three parts questions:
# 
# #### Part one: General Explore
# 
#     At part one, I explored some general questions. The result turned out that the movie popularity trend is growing from 1960 on average. Moreever, I focused on the movies which are with high revenue. I found movies with higher revenue level are with higher popularity in recent five years on average. Besides, movies with higher revenue level don't have the significant high score rating in recent five years. And this results made me want to learn more: What's properties that are associated with high popularity movies? What's properties that are associated with high high voting score?
# 
# #### Part two: Find the Properties are Associated with Successful Movies
# 
#     At this part, I first found out the properties that are associated with high popularity movies. They were with high budget levels and longer run time. And cast associated with high popularity movies are Robert De Niro and Bruce Willis; director associated with high popularity movies are Steven Spielberg; genres associated with high popularity movies are drama, comedy, and thriller but they also appeared in the most unpopular movies; keywords associated with high popularity movies are based on novel and dystopia; producer associated with high popularity movies are Warner Bros., Universal Pictures and Paramount Pictures, but they are also appeared in the most unpopular movies.
# 
#     And the I found out the properties that are associated with high high voting score. Each level in both runtime and budget don't have obvious different high rating score. In other words, the low budget level or the low budget may still have a high rating. And Martin Scorsese and Clint Eastwood have made top quality movies on average over the past years from 1960; the top quality movies have the keywords with based on novel and woman director over the past years from 1960.
# 
# 
# #### Part three: Top Keywords and Genres Trends by Generation
# 
#     In this part, I explored the number of movie released trend year by year. Then explored the keywords and genres trends, with group the dataframe into five generations: 1960s, 1970s, 1980s, 1990s and 2000s.
# 
#     The number of movie released are increasing year by year. And the it is in the accelerated growth trend. Besides, In 1960s and 1970s, the top keywords was based on novel, in 1980s, the top keyword was nudity. In 1990s, independent film became the top keyword. And after 2000, the movie with the feature woman director were released most. Further more, the genre Drama are the most filmed in almost all generation. Only the 1980s are dominated by the comedy type.
# 
#     To sum up, I did find a lot of interesting information among the dataset, just hope that I can dig more! But there are still some limitations.

# 
# # Limitation
# 
# 1. Data quality: althought I assume the zero values in revenue and budget column are missing, there are still a lot of unreasonable small/big value in the both of the columns. Also, the metrics about rating or popularity are not defined clearly, and the basis of them may be changing year by year.
# 2. Although the the popularity doesn't have the upperbound , it actually have the high probability of having outliers. But I choose to retain the data to keep the data originalty. Maybe there are still the reason that I should take it into account.
# 3. Units of revenue and budget column: I am not sure that the budgets and revenues all in US dollars?
# 4. The inflation effect: I used the revenue and budget data to explore, but I didn't use the adjusted data, although it is provided the adjusted data based on the year 2010.
# 5. In my reseach one, although I discussed the distribution of popularity in different revenue levels in recent five years, but I just cut the revenue levels based on it's quantile. I didn't find out the whole revenue distributin in the fisrt, so there may be exist risks that the high revenue level still cover a wide of range, and may affect the final result. Besides, in the part, I just discuss data in the recent five year, maybe in other year there are some different distribution.
# 6. In research two, I dicussed the properties are associated with successful movies. The successful I defined here are high popularity and high voting score. But I didn't find the properties of high revenue since I just assume the high revenue level are with higher popularity, which is I found in research one, so it makes me just leave out the finding the properties of high revenue movie. But I think there must be some other factor that are associated with high revenue movies.
# 7. In research two, I dicussed the budget level and runtime level properties, but I just cut both of them based on the whole time quantile data not year by year. Also, to cut them into four levels based on quantile still rough.
# 8. The categorical data, when I analysed them, I just split them one by one, and count them one by one. But the thing is, there must be some effect when these words combine. For example, the keyword based on novel is popular, but what truly keyword that makes the movie sucess maybe the based on novel&adventure.
# 9. I didn't count number of votes into consideration, so the rating score may be a bias whe the vote number is few.
# 
# 

# In[90]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




