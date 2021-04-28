#Standard
import pandas as pd
from pandas import Series, DataFrame

#Plotting
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# Set up the Titanic 'Train' csv file as a DataFrame
titanicDataFrame = pd.read_csv('train.csv')

# Increase output of columns to view entire DataFrame
pd.set_option('display.max_columns', 12)

# View preview of data
print(titanicDataFrame.head())

# Get overall info for DataSet
print(titanicDataFrame.info())

sns.countplot('Sex',data=titanicDataFrame)
sns.countplot('Age',data=titanicDataFrame)
sns.countplot('Pclass',data=titanicDataFrame,hue='Sex')
plt.show()

# Function adding discrepancy between adult and child passengers
def male_female_child(passenger):
    # Take age and sex
    age, sex = passenger
    # Compare the age, otherwise leave the sex
    if age < 16:
        return 'child'
    else:
        return sex

# Define new column in DataFrame called 'person'
titanicDataFrame['person'] = titanicDataFrame[['Age','Sex']].apply(male_female_child,axis=1)
print(titanicDataFrame[0:10])

#Create new countplot with child category included
sns.countplot('Pclass',data=titanicDataFrame,hue='person')
plt.show()

# Create histogram to look at distribution of ages to get a better picture of passenger demographic
titanicDataFrame['Age'].hist(bins=70)
print(titanicDataFrame['person'].value_counts())
plt.show()

# Visualize data with FacetGrid by plotting mutiple kde plots on one plot
# First FacetGrid will examine distribution of male and female adult passengers and their age
figure = sns.FacetGrid(titanicDataFrame, hue='Sex',aspect=4)
figure.map(sns.kdeplot,'Age',shade=True)
oldest = titanicDataFrame['Age'].max()
figure.set(xlim=(0,oldest))
figure.add_legend()
plt.show()

# Second FacetGrid shows same distribution as first but with children included in data
figure = sns.FacetGrid(titanicDataFrame, hue='person',aspect=4)
figure.map(sns.kdeplot,'Age',shade=True)
oldest = titanicDataFrame['Age'].max()
figure.set(xlim=(0,oldest))
figure.add_legend()
plt.show()

# Third FacetGrid examines distribution of age according to what class each passenger was in
figure = sns.FacetGrid(titanicDataFrame, hue='Pclass',aspect=4)
figure.map(sns.kdeplot,'Age',shade=True)
oldest = titanicDataFrame['Age'].max()
figure.set(xlim=(0,oldest))
figure.add_legend()
plt.show()

#######################################################################
# What Deck were passengers on and how does that relate to their class?
print(titanicDataFrame.head())

# Drop all NaN values from Cabin column
deck = titanicDataFrame['Cabin'].dropna()
print(deck.head())

# Only need first letter of deck to to classify its level
# Grab letter of deck level with for loop
levels = []
for level in deck:
    # For loop to grab the first letter
    levels.append(level[0])

# Reset DataFrame and use countplot to view
cabin_dataFrame = DataFrame(levels)
cabin_dataFrame.columns = ['Cabin']
cabin_dataFrame = cabin_dataFrame[cabin_dataFrame.Cabin != 'T']
sns.countplot('Cabin',data=cabin_dataFrame, order=['A','B','C','D','E','F','G'], palette='YlGn_d')
plt.show()

#######################################################
# Where did the passengers come from?
# Checking data on website states that C,Q,or S in 'Embarked' column
# stands for origins of Cherbourg, Queenstown, and Southhampton

# Make a countplot to look at results NOTE: order argument is used to deal w/ null values
sns.countplot('Embarked',data=titanicDataFrame,hue='Pclass',order=['C','Q','S'])
plt.show()

#######################################################
# Who was alone and who was with family?

# Add new column to DataFrame to define alone
titanicDataFrame['Alone'] = titanicDataFrame.Parch + titanicDataFrame.SibSp
print(titanicDataFrame['Alone'])

# Now know if alone column > 0, then passenger wasn't alone
# Change column to reflect 'alone' and 'with family'
titanicDataFrame['Alone'].loc[titanicDataFrame['Alone'] > 0] = 'With Family'
titanicDataFrame['Alone'].loc[titanicDataFrame['Alone'] == 0] = 'Alone'
# View head of DataFrame to confirm changes
print(titanicDataFrame.head())

# Use countplot to get a visualization of newly added data
sns.countplot('Alone',data=titanicDataFrame,palette='Blues')
plt.show()

########################################################
# Final question: What factors helped someone survive the sinking?

# Create new 'Survivor' column with yes/no instead of 0/1
titanicDataFrame['Survivor'] = titanicDataFrame.Survived.map({0: 'no', 1: 'yes'})

# Quick countplot to see relationship between survived vs dead
sns.countplot('Survivor',data=titanicDataFrame,palette='Oranges')
plt.show()

# Create factorplot to see survival rate between class types
sns.factorplot('Pclass','Survived',data=titanicDataFrame)
plt.show()

# Create factorplot to see survival rate between class types and gender
sns.factorplot('Pclass','Survived',hue='person',data=titanicDataFrame)
plt.show()

# Did age have a factor in death rates?
# Use linear plot to track age versus survival
sns.lmplot('Age','Survived',data=titanicDataFrame)
plt.show()

# Use same linear plot but use hue to distinguish between classes
ages = [10,20,30,50,70,80]
sns.lmplot('Age','Survived',hue='Pclass',data=titanicDataFrame,palette='cool',x_bins=ages)
plt.show()

# Create another linear plot to relate gender with age and survival ratio
sns.lmplot('Age','Survived',hue='Sex',data=titanicDataFrame,palette='winter_d',x_bins=ages)
plt.show()

########################################################################
# Did the deck have an effect on survival rate?
cabin_dataFrame = pd.concat([cabin_dataFrame, titanicDataFrame['Sex']],axis=1)
cabin_dataFrame = pd.concat([cabin_dataFrame, titanicDataFrame['Survived']], axis=1)
print(cabin_dataFrame.head())
sns.factorplot('Cabin','Survived',data=cabin_dataFrame,hue='Sex',palette='winter',order=['A','B','C','D','E','F','G'])
plt.show()
