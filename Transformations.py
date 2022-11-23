import pandas
import seaborn as sns

data = pandas.read_csv('C:/Users/Shashi/Documents/DataSets/calories_consumed.csv')

data.shape

data.info()

data.dtypes


data.mean()

data['Calories Consumed'].mean()

data.median()

data.mode()


data.var()

data.std()


data.skew()


data.kurt()


import matplotlib.pyplot as plt

plt.boxplot(data['Calories Consumed'])
plt.boxplot(data['Weight gained (grams)'])
plt.boxplot(data)

plt.hist(data)

plt.scatter(x = ['Calories_Consumed'], y = ['Weight_gained'])


# Duplicated values

duplicate = data.duplicated()
duplicate.sum()


# Zero Variance

data.var() == 0


# Discretization

# Binaryization


data['Weight_gained_new'] = pandas.cut(data['Weight gained (grams)'], 
                                         bins = [min(data['Weight gained (grams)']),
                                                 data['Weight gained (grams)'].mean(), max(data['Weight gained (grams)'])],include_lowest=True,labels=['Low','High'])
                                          

data['Calories Consumed'] = pandas.cut(data['Calories Consumed'],
                                        bins=[min(data['Calories Consumed']), data['Calories Consumed'].mean(), max(data['Calories Consumed'])],
                                          include_lowest=True,
                                           labels=["Low", "High"])


# Missing Values

data.isna().sum()



# Dummy value creation for column Calories Consumed

data_new = pandas.get_dummies(data)

data_new1 = pandas.get_dummies(data, drop_first=True)

plt.boxplot(data_new1)



# Normal Q Q Plot

import scipy.stats as stats
import pylab

stats.probplot(data['Calories Consumed'], dist='norm', plot = pylab)
stats.probplot(data['Weight gained (grams)'], dist='norm', plot = pylab)


# Transformation to make Weight gained (grams) variable normal

import numpy as np

stats.probplot(np.log(data['Weight gained (grams)']), dist = "norm", plot = pylab)


fitted_data, fitted_lambda = stats.boxcox(data['Weight gained (grams)'])


fig, ax = plt.subplots(1, 2)

sns.distplot(data['Weight gained (grams)'], hist = False, kde = True,
              kde_kws = {'shade' : True, 'linewidth' : 2},
               label = 'Non-Normal', color = 'blue', ax = ax[0])
 

sns.distplot(fitted_data, hist = False, kde = True,
              kde_kws = {'shade' : True, 'linewidth' : 2},
               label = 'Normal', color = 'blue', ax = ax[1])


plt.legend(loc = 'upper right')


fig.set_figheight(5)
fig.set_figwidth(10)

# Transform data

print(f"lambda value used for transformation : {fitted_lambda}")

# YeoJhonson Transformation

from feature_engine import transformation

tf = transformation.YeoJohnsonTransformer(variables=('Weight gained (grams)'))

data_tf = tf.fit_transform(data)

prob = stats.probplot(data_tf['Weight gained (grams)'], dist = stats.norm, plot = pylab)



























