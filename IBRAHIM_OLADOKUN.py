#!/usr/bin/env python
# coding: utf-8

# # Probability of default: credit scoring model

# Building a model that borrowers can use to help make the best financial decisions.

# The following variables are contained in the csv Dataset given:
# 
#     VARIABLE NAMES                  :                           DESCRIPTIONS
# SeriousDlqin2yrs                    :     Person experienced 90 days past due delinquency or worse (Target variable / label)
# 
# RevolvingUtilizationOfUnsecuredLines:  Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits
# 
# age Age of borrower in years
# 
# NumberOfTime30-59DaysPastDueNotWorse: Number of times borrower has been 30-59 days past due but no worse in the last 2 years.
# 
# DebtRatio: Monthly debt payments, alimony,living costs divided by monthy gross income
# 
# MonthlyIncome: Monthly income
# 
# NumberOfOpenCreditLinesAndLoans: Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards)
# 
# NumberOfTimes90DaysLate: Number of times borrower has been 90 days or more past due.
# 
# NumberRealEstateLoansOrLines: Number of mortgage and real estate loans including home equity lines of credit
# 
# NumberOfTime60-89DaysPastDueNotWorse: Number of times borrower has been 60-89 days past due but no worse in the last 2 years.
# 
# NumberOfDependents: Number of dependents in family excluding themselves (spouse, children etc.)
# 

# I will be using a random forest classifier for two reasons: firstly, because it would allow me to quickly and easily change the output to a simple binary classification problem. Secondly, because the predict_proba functionality allows me to output a probability score (probability of 1), this score is what i will use for predicting the probability of 90 days past due delinquency or worse in 2 years time.
# 
# Furthermore, I will predominantly be adopting a quantiles based approach in order to streamline the process as much as possible so that hypothetical credit checks can be returned as easily and as quickly as possible.

# In[1]:


# Load in our libraries
import pandas as pd
import numpy as np
import re
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score

sns.set(style='white', context='notebook', palette='deep')
pd.options.display.max_columns = 100


# # Exploratory Data Analysis

# In[3]:


train = pd.read_csv("cs-training.csv")
test = pd.read_csv("cs-test.csv")


# In[6]:


test.head()


# In[7]:


train.shape


# In[8]:


train.describe()


# In[9]:


train.info()


# In[10]:


train.isnull().sum()


# SeriousDlqin2yrs is the target variable (label), it is binary.
# 
# The training set contains 150,000 observations of 11 numerical features and 1 label.
# 
# 
# 
# NumberOfDependents column  and MonthlyIncome column  contains NaN values, It is suspected that other variables contains errors (Age)

# In[11]:


test.isnull().sum()


# The test data also contains several NaN values

# # Target distribution

# In[13]:


ax = sns.countplot(x = train.SeriousDlqin2yrs ,palette="Set3")
sns.set(font_scale=1.5)
ax.set_ylim(top = 150000)
ax.set_xlabel('Financial difficulty in 2 years')
ax.set_ylabel('Frequency')
fig = plt.gcf()
fig.set_size_inches(10,5)
ax.set_ylim(top=160000)

plt.show()


# Our target variable distribution of the above plot is very skewed i.e the right and left disribution are shaped differently from each other

# # Detecting outliers

# In[14]:


def detect_outliers(df,n,features):
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers

# detect outliers from Age, SibSp , Parch and Fare
# These are the numerical features present in the dataset
Outliers_to_drop = detect_outliers(train,2,["RevolvingUtilizationOfUnsecuredLines",
                                            "age",
                                            "NumberOfTime30-59DaysPastDueNotWorse",
                                            "DebtRatio",
                                            "MonthlyIncome",
                                            "NumberOfOpenCreditLinesAndLoans",
                                            "NumberOfTimes90DaysLate",
                                            "NumberRealEstateLoansOrLines",
                                            "NumberOfTime60-89DaysPastDueNotWorse",
                                            "Unnamed: 0",
                                            "NumberOfDependents"])


# In[17]:


train.loc[Outliers_to_drop]


# 3527 outliers  were detected in the training set, which represents 2.53% of our training data.I will drop these outliers

# In[18]:


train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)


# # Merging datasets

# In[21]:


train_len = len(train)
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)


# In[22]:


dataset.shape


# In[24]:


dataset = dataset.rename(columns={'Unnamed: 0': 'Unknown',
                                  'SeriousDlqin2yrs': 'Target',
                                  'RevolvingUtilizationOfUnsecuredLines': 'UnsecLines',
                                  'NumberOfTime30-59DaysPastDueNotWorse': 'Late3059',
                                  'DebtRatio': 'DebtRatio',
                                  'MonthlyIncome': 'MonthlyIncome',
                                  'NumberOfOpenCreditLinesAndLoans': 'OpenCredit',
                                  'NumberOfTimes90DaysLate': 'Late90',
                                  'NumberRealEstateLoansOrLines': 'PropLines',
                                  'NumberOfTime60-89DaysPastDueNotWorse': 'Late6089',
                                  'NumberOfDependents': 'Deps'})

train = train.rename(columns={'Unnamed: 0': 'Unknown',
                                  'SeriousDlqin2yrs': 'Target',
                                  'RevolvingUtilizationOfUnsecuredLines': 'UnsecLines',
                                  'NumberOfTime30-59DaysPastDueNotWorse': 'Late3059',
                                  'DebtRatio': 'DebtRatio',
                                  'MonthlyIncome': 'MonthlyIncome',
                                  'NumberOfOpenCreditLinesAndLoans': 'OpenCredit',
                                  'NumberOfTimes90DaysLate': 'Late90',
                                  'NumberRealEstateLoansOrLines': 'PropLines',
                                  'NumberOfTime60-89DaysPastDueNotWorse': 'Late6089',
                                  'NumberOfDependents': 'Deps'})

test = test.rename(columns={'Unnamed: 0': 'Unknown',
                                  'SeriousDlqin2yrs': 'Target',
                                  'RevolvingUtilizationOfUnsecuredLines': 'UnsecLines',
                                  'NumberOfTime30-59DaysPastDueNotWorse': 'Late3059',
                                  'DebtRatio': 'DebtRatio',
                                  'MonthlyIncome': 'MonthlyIncome',
                                  'NumberOfOpenCreditLinesAndLoans': 'OpenCredit',
                                  'NumberOfTimes90DaysLate': 'Late90',
                                  'NumberRealEstateLoansOrLines': 'PropLines',
                                  'NumberOfTime60-89DaysPastDueNotWorse': 'Late6089',
                                  'NumberOfDependents': 'Deps'})


# # Exploring variables

# In[25]:


# Correlation matrix
g = sns.heatmap(train.corr(),annot=False, fmt = ".2f", cmap = "coolwarm")


# This shows that the Target has the highest correlation with age, previous late payments, and the number of dependants

# # Exploring UnsecLines

# In[27]:


dataset.UnsecLines.describe()


# In[28]:


dataset.UnsecLines = pd.qcut(dataset.UnsecLines.values, 5).codes


# In[29]:


# Exploring UnsecLines feature vs Target
g  = sns.factorplot(x="UnsecLines",y="Target",data=dataset,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")


# This shows that there is an almost exponential relationship between this variable and our target

# # Exploring Age

# In[31]:


# Exploring Age vs Survived
g = sns.FacetGrid(dataset, col='Target')
g = g.map(sns.distplot, "age")


# In[32]:


dataset.age = pd.qcut(dataset.age.values, 5).codes


# In[33]:


# Exploring age feature vs Target
g  = sns.factorplot(x="age",y="Target",data=dataset,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")


# The age has an inverse relationship to default risk

# # Exploring Late3059

# In[35]:


# Explore UnsecLines feature vs Target
g  = sns.factorplot(x="Late3059",y="Target",data=dataset,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")


# In[36]:


for i in range(len(dataset)):
    if dataset.Late3059[i] >= 6:
        dataset.Late3059[i] = 6


# In[38]:


# Exploring UnsecLines feature vs Target
g  = sns.factorplot(x="Late3059",y="Target",data=dataset,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")


# Due to very high standard deviations i decided to group customers who have 6 or more late payments together. Which shows that this has boosted the predictive capacity and reduced the variance of Late3059

# # Exploring DebtRatio

# In[40]:


# Exploring Age vs Survived
g = sns.FacetGrid(dataset, col='Target')
g = g.map(sns.distplot, "DebtRatio")


# In[41]:


dataset.DebtRatio = pd.qcut(dataset.DebtRatio.values, 5).codes


# In[42]:


# Explore DebtRatio feature quantiles vs Target
g  = sns.factorplot(x="DebtRatio",y="Target",data=dataset,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")


# In[43]:


dataset.MonthlyIncome.isnull().sum()


# In[44]:


g = sns.heatmap(dataset[["MonthlyIncome","Unknown","UnsecLines","OpenCredit","PropLines"]].corr(),cmap="BrBG",annot=True)


# In[45]:


g = sns.heatmap(dataset[["MonthlyIncome","age","DebtRatio","Deps","Target"]].corr(),cmap="BrBG",annot=True)


# In[46]:


g = sns.heatmap(dataset[["MonthlyIncome","Late3059","Late6089","Late90"]].corr(),cmap="BrBG",annot=True)


# MonthlyIncome has no strong correlation with any other variable so the NaN values cannot be accurately estimated. Thus, i will fill the NaN with the median value

# In[47]:


dataset.MonthlyIncome.median()


# In[48]:


#Fill Embarked nan values of dataset set with 'S' most frequent value
dataset.MonthlyIncome = dataset.MonthlyIncome.fillna(dataset.MonthlyIncome.median())


# In[49]:


dataset.MonthlyIncome = pd.qcut(dataset.MonthlyIncome.values, 5).codes


# In[50]:


# Exploring DebtRatio feature quantiles vs Target
g  = sns.factorplot(x="MonthlyIncome",y="Target",data=dataset,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")


# # Exploring OpenCredit

# In[51]:


dataset.OpenCredit.describe()


# In[52]:


dataset.OpenCredit = pd.qcut(dataset.OpenCredit.values, 5).codes


# In[53]:


# Explore DebtRatio feature quantiles vs Target
g  = sns.factorplot(x="OpenCredit",y="Target",data=dataset,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")


# # Exploring Late90

# In[54]:


dataset.Late90.describe()


# In[55]:


# Explore DebtRatio feature quantiles vs Target
g  = sns.factorplot(x="Late90",y="Target",data=dataset,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")


# In[56]:


for i in range(len(dataset)):
    if dataset.Late90[i] >= 5:
        dataset.Late90[i] = 5


# In[57]:


# Exploring DebtRatio feature quantiles vs Target
g  = sns.factorplot(x="Late90",y="Target",data=dataset,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")


# # Exploring PropLines

# In[58]:


dataset.PropLines.describe()


# In[59]:


# Explore DebtRatio feature quantiles vs Target
g  = sns.factorplot(x="PropLines",y="Target",data=dataset,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")


# In[60]:


for i in range(len(dataset)):
    if dataset.PropLines[i] >= 6:
        dataset.PropLines[i] = 6


# In[61]:


# Exploring DebtRatio feature quantiles vs Target
g  = sns.factorplot(x="PropLines",y="Target",data=dataset,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")


# # Exploring Late6089

# In[62]:


# Exploring Late6089 feature quantiles vs Target
g  = sns.factorplot(x="Late6089",y="Target",data=dataset,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")


# In[63]:


for i in range(len(dataset)):
    if dataset.Late6089[i] >= 3:
        dataset.Late6089[i] = 3


# In[64]:


# Exploring Late6089 feature quantiles vs Target
g  = sns.factorplot(x="Late6089",y="Target",data=dataset,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")


# # Exploring Deps

# In[65]:


dataset.Deps.describe()


# In[66]:


dataset.Deps = dataset.Deps.fillna(dataset.Deps.median())


# In[67]:


dataset.Deps.isnull().sum()


# In[68]:


# Explore DebtRatio feature quantiles vs Target
g  = sns.factorplot(x="Deps",y="Target",data=dataset,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")


# In[69]:


for i in range(len(dataset)):
    if dataset.Deps[i] >= 4:
        dataset.Deps[i] = 4


# In[71]:


# Exploring DebtRatio feature quantiles vs Target
g  = sns.factorplot(x="Deps",y="Target",data=dataset,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")


# # Final NaN check

# In[72]:


dataset.info()


# In[74]:


dataset.head()


# Building binary/dummy variables

# In[76]:


dataset = pd.get_dummies(dataset, columns = ["UnsecLines"], prefix="UnsecLines")
dataset = pd.get_dummies(dataset, columns = ["age"], prefix="age")
dataset = pd.get_dummies(dataset, columns = ["Late3059"], prefix="Late3059")
dataset = pd.get_dummies(dataset, columns = ["DebtRatio"], prefix="DebtRatio")
dataset = pd.get_dummies(dataset, columns = ["MonthlyIncome"], prefix="MonthlyIncome")
dataset = pd.get_dummies(dataset, columns = ["OpenCredit"], prefix="OpenCredit")
dataset = pd.get_dummies(dataset, columns = ["Late90"], prefix="Late90")
dataset = pd.get_dummies(dataset, columns = ["PropLines"], prefix="PropLines")
dataset = pd.get_dummies(dataset, columns = ["Late6089"], prefix="Late6089")
dataset = pd.get_dummies(dataset, columns = ["Deps"], prefix="Deps")


# In[77]:


dataset.head()


# In[78]:


dataset.head()


# In[79]:


dataset.shape


# # Building our credit scoring model

# In[82]:


train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels=["Target"],axis = 1,inplace=True)


# In[83]:


test.shape


# In[84]:


## Separate train features and label 

train["Target"] = train["Target"].astype(int)

Y_train = train["Target"]

X_train = train.drop(labels = ["Target", "Unknown"],axis = 1)


# In[85]:


clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(X_train, Y_train)


# In[86]:


features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)


# In[87]:


features.plot(kind='barh', figsize=(20, 20))


# In[92]:


parameters = {'n_estimators': 1000, 'random_state' : 20}
    
model = RandomForestClassifier(**parameters)
model.fit(X_train, Y_train)


# In[93]:


test.head()


# In[94]:


results_df = pd.read_csv("cs-test.csv")


# In[95]:


results_df = results_df.drop(["RevolvingUtilizationOfUnsecuredLines",
                             "age",
                             "NumberOfTime30-59DaysPastDueNotWorse",
                             "DebtRatio",
                             "MonthlyIncome",
                             "NumberOfOpenCreditLinesAndLoans",
                             "NumberOfTimes90DaysLate",
                             "NumberRealEstateLoansOrLines",
                             "NumberOfTime60-89DaysPastDueNotWorse",
                             "NumberOfDependents"], axis=1)


# In[97]:


DefaultProba = model.predict_proba(test.drop(["Unknown"], axis=1))
DefaultProba = DefaultProba[:,1]
results_df.SeriousDlqin2yrs = DefaultProba

results_df = results_df.rename(columns={'Unnamed: 0': 'Id',
                                        'SeriousDlqin2yrs': 'Probability'})


# In[99]:


results_df.head()


# In[100]:


results_df.to_csv("TEST_CREDIT_SCORE.csv", index=False)


# This model lead to an accuracy rate of 0.800498 on Kaggle's unseen test data.
# 
# I deem this accuracy rate to be acceptable given that i used a relatively simple quantile based approach and in light of the fact that no parameter optimization was undertaken.
# 

# In[109]:


results_df.to_csv(r"C:\Users\Lenovo Core i7\Desktop\REnmoneycsv\ttttt.csv",index =False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




