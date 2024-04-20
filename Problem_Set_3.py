#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Question 1
# Step 1. Import the necessary libraries

import numpy as np


# In[2]:


import pandas as pd


# In[5]:


# Question 1
# Step 2. Import the dataset from this address.

# Step 3. Assign it to a variable called users

users = pd.read_csv(r'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', sep='|')
users.head(20)


# In[6]:


# Question 1
# Step 4. Discover what is the mean age per occupation

users[["age", "occupation"]].groupby("occupation").mean()


# In[7]:


# Question 1
# Step 5. Discover the Male ratio per occupation and sort it from the most to the least

def conv_gender_to_numeric(x):
    if x == 'M':
        return 1
    if x == 'F':
        return 0
    
users['gender_num'] = users['gender'].apply(conv_gender_to_numeric)

ratio = users.groupby('occupation').gender_num.sum() / users.occupation.value_counts() * 100 

ratio.sort_values(ascending = False)


# In[8]:


# Question 1
# Step 6. For each occupation, calculate the minimum and maximum ages

users.groupby('occupation').age.agg(['min', 'max'])


# In[9]:


# Question 1
# Step 7. For each combination of occupation and sex, calculate the mean age

users.groupby(['occupation', 'gender']).age.mean()


# In[10]:


# Question 1
# Step 8. For each occupation present the percentage of women and men

gender_occup_count = users.groupby(['occupation', 'gender']).agg({'gender': 'count'})
occup_count = users.groupby(['occupation']).agg('count')
occup_gender_percent = gender_occup_count.div(occup_count, level = "occupation") * 100
occup_gender_percent.loc[: , 'gender']


# In[ ]:





# In[ ]:





# In[ ]:


# Question 2 Euro Teams

# Step 1. Import the necessary libraries

import numpy as np

import pandas as pd


# In[11]:


# Question 2 Euro Teams
# Step 2. Import the dataset from this address
# Step 3. Assign it to a variable called euro12


euro12 = pd.read_csv(r'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv')
euro12.head(20)


# In[12]:


# Question 2 Euro Teams
# Step 4. Select only the Goal column


euro12['Goals']


# In[13]:


# Question 2 Euro Teams
# Step 5. How many team participated in the Euro2012?

teams_participated = euro12['Team'].count()
print(teams_participated, "teams participated in Euro2012")


# In[14]:


# Question 2 Euro Teams
# Step 6. What is the number of columns in the dataset?

columns = len(euro12.axes[1])
print("Number of Columns: ", columns)


# In[15]:


# Question 2 Euro Teams
# Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them to a dataframe called discipline

discipline= pd.DataFrame(euro12, columns = ['Team','Red Cards','Yellow Cards'])
discipline


# In[16]:


# Question 2 Euro Teams
# Step  8. Sort the teams by Red Cards, then to Yellow Cards

discipline.sort_values(by=['Red Cards', 'Yellow Cards'])
discipline


# In[17]:


# Question 2 Euro Teams
# Step 9. Calculate the mean Yellow Cards given per Team

Mean_Yellow_Cards = discipline['Yellow Cards'].mean()
print("Mean Yellow Cards given per Team:", Mean_Yellow_Cards)


# In[18]:


# Question 2 Euro Teams
# Step 10. Filter teams that scored more than 6 goals

goals_over_6 = euro12['Goals'] > 6
euro12[goals_over_6]


# In[19]:


# Question 2 Euro Teams
# Step 11. Select the teams that start with G

euro12[euro12.Team.str.startswith('G')]


# In[20]:


# Question 2 Euro Teams
# Step 12. Select the first 7 columns


euro12.head(7)


# In[21]:


# Question 2 Euro Teams
# Step 13. Select all columns except the last 3

euro12.loc[:, ~euro12.columns.isin(['Subs on', 'Subs off', 'Players Used'])]  


# In[22]:


# Question 2 Euro Teams
# Step 14. Present only the Shooting Accuracy from England, Italy and Russia

euro12.loc[euro12.Team.isin(['England', 'Italy', 'Russia']), ['Team','Shooting Accuracy']]


# In[ ]:





# In[ ]:





# In[24]:


# Question 3 Housing 
#Step 1. Import the necessary libraries

import numpy as np
import pandas as pd
import random


# In[25]:


# Question 3 Housing 
#Step  2. Create 3 differents Series, each of length 100, as follows:
# The first a random number from 1 to 4

first_series = [[random.randint(1, 4)] for i in range(100)]
df_first = pd.DataFrame(first_series)
df_first


# In[26]:


# The second a random number from 1 to 3

second_series = [[random.randint(1, 3)] for i in range(100)]
df_second = pd.DataFrame(second_series)
df_second


# In[27]:


# The third a random number from 10,000 to 30,000

third_series = [[random.randint(10000, 30000)] for i in range(100)]
df_third = pd.DataFrame(third_series)
df_third


# In[28]:


# Question 3 Housing 
#Step 3. Create a DataFrame by joinning the Series by column

df = pd.concat([df_first, df_second, df_third], axis=1)
df


# In[29]:


# Question 3 Housing 
#Step 4. Change the name of the columns to bedrs, bathrs, price_sqr_meter

df.columns = ['bedrs', 'bathrs', 'price_sqr_meter']
df


# In[30]:


# Question 3 Housing 
#Step 5. Create a one column DataFrame with the values of the 3 Series and assign it to 'bigcolumn'

df_new = df.bedrs.astype(str).str.cat(df.bathrs.astype(str)).str.cat(df.price_sqr_meter.astype(str))
df_new.columns = ['bigcolumn']
df_new


# In[31]:


# Question 3 Housing 
#Step 6. Ops it seems it is going only until index 99. Is it true?

Soution: True

print(df_new)


# In[32]:


# Question 3 Housing 
#Step 7. Reindex the DataFrame so it goes from 0 to 299

df_new.reset_index()
df_new.reindex(index=range(0,299))


# In[ ]:





# In[ ]:





# In[41]:


# Question 4
# Wind Statistics 
#The data have been modified to contain some missing values, 
#identified by NaN.Using pandas should make this exercise easier, in particular for the bonus question.
#You should be able to perform all of these operations without using a for loop orother looping construct.
#The data in 'wind.data' has the following format:Yr Mo Dy RPT VAL ROS KIL SHA BIR DUB CLA MUL CLO BELMAL61 1 1 15.04 
#14.96 13.17 9.29 NaN 9.87 13.67 10.25 10.83 12.58 18.50 15.0461 1 2 14.71 NaN 10.83 6.50 12.62 7.67 11.50 10.04 9.79 9.67 
#17.54 13.8361 1 3 18.50 16.88 12.33 10.13 11.17 6.17 11.25 NaN 8.50 7.67 12.75 12.71
#The first three columns are year, month, and day. 
#The remaining 12 columns areaverage windspeeds in knots at 12 locations in Ireland on that day.
# Step 1. Import the necessary libraries

import pandas as pd


# In[42]:


#Question 4
# Step 2. Import the dataset from the attached file wind.txt

url = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/06_Stats/Wind_Stats/wind.data"


# In[43]:


#Question 4
# Step 3. Assign it to a variable called data and replace the first 3 columns by a proper datetime index

data = pd.read_csv(url, delim_whitespace=True, parse_dates=[[0, 1, 2]])


# In[44]:


#Question 4
# Step 4. Year 2061? Fix the year and apply the function
def fix_year(x):
    year = x.year - 100 if x.year > 1989 else x.year
    return pd.Timestamp(year, x.month, x.day)
 
data['Yr_Mo_Dy'] = data['Yr_Mo_Dy'].apply(fix_year)


# In[45]:


#Question 4
# Step 5. Set the right dates as the index

data.set_index('Yr_Mo_Dy', inplace=True)


# In[46]:


# Question 4
# Step 6. Compute how many values are missing for each location over the entirerecord.They should be ignored in all calculations below.

missing_values_per_location = data.isnull().sum()


# In[47]:


# Question 4
# Step 7. Compute how many non-missing values there are in total.

total_non_missing_values = data.notnull().sum().sum()


# In[48]:


# Question 4
# Step 8. Calculate the mean windspeeds of the windspeeds over all the locations andall the times.A single number for the entire dataset.

mean_windspeed = data.mean().mean()


# In[49]:


#Question 4
# Step 9. Create a DataFrame called loc_stats and calculate the min, max, mean, and standard deviations of the windspeeds at each location over all the days
loc_stats = data.describe(percentiles=[])


# In[50]:


#Question 4
# Step 10. Create a DataFrame called day_stats and calculate the min, max, mean, and standard deviations of the windspeeds across all the locations at each day
day_stats = pd.DataFrame()
day_stats['min'] = data.min(axis=1)
day_stats['max'] = data.max(axis=1)
day_stats['mean'] = data.mean(axis=1)
day_stats['std'] = data.std(axis=1)


# In[51]:


#Question 4
# Step 11. Find the average windspeed in January for each location
january_avg_windspeed = data[data.index.month == 1].mean()


# In[52]:


#Question 4
# Step 12. Downsample the record to a yearly frequency for each location
yearly_data = data.resample('Y').mean()


# In[53]:


#Question 4
# Step 13. Downsample the record to a monthly frequency for each location
monthly_data = data.resample('M').mean()


# In[54]:


#Question 4
# Step 14. Downsample the record to a weekly frequency for each location
weekly_data = data.resample('W').mean()


# In[55]:


#Question 4
# Step 15. Calculate the min, max, mean, and standard deviations of the windspeeds across all locations for each week for the first 52 weeks
weekly_stats_first_52_weeks = weekly_data.iloc[:52].agg(['min', 'max', 'mean', 'std'])
 
# Print or utilize the results as needed
print("Missing values per location:")
print(missing_values_per_location)
print("\nTotal non-missing values:", total_non_missing_values)
print("\nMean windspeeds of the entire dataset:", mean_windspeed)
print("\nLocation statistics:")
print(loc_stats)
print("\nDay statistics:")
print(day_stats)
print("\nAverage windspeed in January for each location:")
print(january_avg_windspeed)
print("\nYearly frequency data:")
print(yearly_data)
print("\nMonthly frequency data:")
print(monthly_data)
print("\nWeekly frequency data:")
print(weekly_data)
print("\nWeekly statistics for the first 52 weeks:")
print(weekly_stats_first_52_weeks)


# In[ ]:





# In[ ]:





# In[17]:


# Question 5
# Step 1. Import the necessary libraries
import pandas as pd


# In[18]:


# Question 5
# Step 2. Import the dataset from this address.
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv"


# In[19]:


# Question 5
# Step 3. Assign it to a variable called chipo.
chipo = pd.read_csv(url, sep='\t')


# In[20]:


# Question 5
# Step 4. See the first 10 entries
print("First 10 entries:\n", chipo.head(10))


# In[21]:


# Question 5
# Step 5. What is the number of observations in the dataset?
print("\nNumber of observations:", len(chipo))


# In[22]:


# Question 5
# Step 6. What is the number of columns in the dataset?
print("\nNumber of columns:", len(chipo.columns))


# In[23]:


# Question 5
# Step 7. Print the name of all the columns.
print("\nColumn names:", chipo.columns.tolist())


# In[24]:


# Question 5
# Step 8. How is the dataset indexed?
print("\nIndex:", chipo.index)


# In[25]:


# Question 5
# Step 9. Which was the most-ordered item?
most_ordered_item = chipo.groupby('item_name').sum().sort_values(by='quantity', ascending=False).head(1).index[0]
print("\nMost-ordered item:", most_ordered_item)


# In[26]:


# Question 5
# Step 10. For the most-ordered item, how many items were ordered?
most_ordered_item_quantity = chipo.groupby('item_name').sum().loc[most_ordered_item, 'quantity']
print("\nNumber of items ordered for the most-ordered item:", most_ordered_item_quantity)


# In[27]:


# Question 5
# Step 11. What was the most ordered item in the choice_description column?
most_ordered_choice_description = chipo.groupby('choice_description').sum().sort_values(by='quantity', ascending=False).head(1).index[0]
print("\nMost ordered item in the choice_description column:", most_ordered_choice_description)


# In[28]:


# Question 5
# Step 12. How many items were orderd in total?
total_items_ordered = chipo['quantity'].sum()
print("\nTotal items ordered:", total_items_ordered)


# In[29]:


# Question 5
# Step 13.• Turn the item price into a float

chipo['item_price'] = chipo['item_price'].apply(lambda x: float(x[1:]))

#• Check the item price type
#• Create a lambda function and change the type of item price
#• Check the item price typeStep 

print("\nType of item price after conversion:", chipo['item_price'].dtype)


# In[30]:


#Question 5
#14. How much was the revenue for the period in the dataset?
revenue = (chipo['quantity'] * chipo['item_price']).sum()
print("\nRevenue for the period:", revenue)


# In[32]:


# Question 5
# Step 15. How many orders were made in the period?Step 
num_orders = chipo['order_id'].nunique()
print("\nNumber of orders made in the period:", num_orders)


# In[33]:


#Question 5
#16. What is the average revenue amount per order?
avg_revenue_per_order = revenue / num_orders
print("\nAverage revenue amount per order:", avg_revenue_per_order)


# In[34]:


# Question 5
# Step 17. How many different items are sold?

num_unique_items = chipo['item_name'].nunique()
print("\nNumber of different items sold:", num_unique_items)


# In[ ]:





# In[ ]:





# In[1]:


# Question 6
# Create a line plot showing the number of marriages and divorces per capita in the U.S. between 1867 and 2014. 
# Label both lines and show the legend.Don't forget to label your axes!

import pandas as pd
import matplotlib.pyplot as plt
 
# Load the dataset
url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/national-marriage-divorce-rates-1867-2014.csv"
data = pd.read_csv(url)
 
# Filter the data for the years between 1867 and 2014
filtered_data = data[(data['Year'] >= 1867) & (data['Year'] <= 2014)]
 
# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(filtered_data['Year'], filtered_data['Marriages_per_1000'], label='Marriages per 1000 people')
plt.plot(filtered_data['Year'], filtered_data['Divorces_per_1000'], label='Divorces per 1000 people')
 
# Labeling
plt.xlabel('Year')
plt.ylabel('Per Capita')
plt.title('Number of Marriages and Divorces per Capita in the U.S. (1867-2014)')
plt.legend()
 
# Show plot
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


# Question 7
# Create a vertical bar chart comparing the number of marriages and divorces percapita in the U.S.
# between 1900, 1950, and 2000.Don't forget to label your axes!



# In[ ]:





# In[ ]:





# In[6]:


# Question 8
# Create a horizontal bar chart that compares the deadliest actors in Hollywood. 
# Sort the actors by their kill count and label each bar with the corresponding actor's name. Don't forget to label your axes!

import matplotlib.pyplot as plt
 
# Hypothetical data (actor names and their kill counts)
actors = ['John Wick', 'The Terminator', 'Rambo', 'Sarah Connor', 'James Bond']
kill_counts = [300, 200, 150, 120, 100]
 
# Sort the actors and kill counts by kill count in descending order
sorted_data = sorted(zip(actors, kill_counts), key=lambda x: x[1], reverse=True)
sorted_actors, sorted_kill_counts = zip(*sorted_data)
 
# Create a horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(sorted_actors, sorted_kill_counts, color='skyblue')
 
# Labeling
plt.xlabel('Kill Count')
plt.ylabel('Actor')
plt.title('Deadliest Actors in Hollywood')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
 
# Annotate each bar with the corresponding actor's name
for i, (actor, kill_count) in enumerate(zip(sorted_actors, sorted_kill_counts)):
    plt.text(kill_count, i, f' {actor}', va='center', ha='left')
 
# Show plot
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# In[ ]:





# In[ ]:





# In[5]:


# Question 9
# Create a pie chart showing the fraction of all Roman Emperors that were assassinated.

# Make sure that the pie chart is an even circle, labels the categories, and shows thepercentage breakdown of the categories.

import matplotlib.pyplot as plt
 
# Hypothetical data (total number of Roman Emperors and number of assassinated Roman Emperors)
total_emperors = 50
assassinated_emperors = 12
 
# Calculate the fraction of assassinated Roman Emperors
fraction_assassinated = assassinated_emperors / total_emperors
 
# Calculate the fraction of non-assassinated Roman Emperors
fraction_survived = 1 - fraction_assassinated
 
# Labels for the categories
labels = ['Assassinated', 'Survived']
 
# Sizes for each category (as percentages)
sizes = [fraction_assassinated * 100, fraction_survived * 100]
 
# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
 
# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')
 
# Title
plt.title('Fraction of Roman Emperors Assassinated')
 
# Show plot
plt.show()


# In[ ]:





# In[ ]:





# In[4]:


# Question 10
#Create a scatter plot showing the relationship between the total revenue earned byarcades 
# and the number of Computer Science PhDs awarded in the U.S. between 2000 and 2009.
# Don't forget to label your axes!Color each dot according to its year

import matplotlib.pyplot as plt
import numpy as np
 
# Hypothetical data (total revenue earned by arcades and number of Computer Science PhDs awarded)
years = np.arange(2000, 2010)
revenue_arcades = [1000000, 1100000, 1200000, 1250000, 1300000, 1350000, 1400000, 1450000, 1500000, 1550000]
cs_phds_awarded = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
 
# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(revenue_arcades, cs_phds_awarded, c=years, cmap='viridis', alpha=0.8)
 
# Colorbar
cbar = plt.colorbar()
cbar.set_label('Year')
 
# Labels and title
plt.xlabel('Total Revenue Earned by Arcades')
plt.ylabel('Number of Computer Science PhDs Awarded')
plt.title('Relationship between Arcade Revenue and Computer Science PhDs')
 
# Show plot
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




