# Imporing the Libraries

import pandas
    
data = pandas.read_csv('C:/Users/Shashi/Documents/DataSets/OnlineRetail.csv', encoding = 'unicode_escape')

data.dtypes

data.info()

data.UnitPrice = data.UnitPrice.astype('int64')


data.shape


data.describe()



data.dtypes

data.duplicated().shape



data1 = data.drop_duplicates()
data1 = data.drop_duplicates(keep = 'last')
data.drop_duplicates(keep = False)

data1.Quantity.shape

data1.shape

data1.Quantity.mean()
data1.Quantity.median()
data1.Quantity.mode()


data1.UnitPrice.shape

data1.UnitPrice.mean()
data1.UnitPrice.median()
data1.UnitPrice.mode()


data1.CustomerID.shape

data1.CustomerID.mean()
data1.CustomerID.median()
data1.CustomerID.mode()
data1.describe()


# data1.boxplot(figsize = (10,10) )

data1.var()

data1.std()



data1.skew()


data1.kurt()



import matplotlib.pyplot as plt

#plt.boxplot(data)

plt.boxplot(data1.UnitPrice)

plt.hist(data1.UnitPrice)


plt.boxplot(data1.Quantity)
plt.hist(data1.Quantity)


plt.boxplot(data1.CustomerID)
plt.hist(data1.CustomerID)



plt.scatter(x = data1['UnitPrice'], y = data1['Quantity'] )

plt.scatter(x = data1['UnitPrice'], y = data1['CustomerID'] )

plt.scatter(x = data1['Quantity'], y = data1['CustomerID'] )

plt.scatter(x = data1['Quantity'], y = data1['UnitPrice'] )

plt.scatter(x = data1['CustomerID'], y = data1['UnitPrice'] )

plt.scatter(x = data1['CustomerID'], y = data1['Quantity'] )



#plt.scatter(x = data1['Country'], y = data1['InvoiceDate'] )




# Outlier's

import seaborn as sns

import numpy as np

iqr = data['UnitPrice'].quantile(0.75) - data['UnitPrice'].quantile(0.25)

upper_limit = data['UnitPrice'].quantile(0.75) + (iqr * 1.5)
lower_limit = data['UnitPrice'].quantile(0.25) - (iqr * 1.5)

Outlier_data = np.where(data['UnitPrice'] > upper_limit, True, np.where(data['UnitPrice'] < lower_limit, True, False))

data_trimmed = data.loc[~(Outlier_data), ]


data.shape, data_trimmed.shape

# Checking for outlier in box plot with seaborn 

sns.boxplot(data_trimmed.UnitPrice)


Out_data = np.where(data['Quantity'] > upper_limit, True, np.where(data['Quantity'] > lower_limit, True, False))

data_trim = data.loc[~(Out_data), ]

data.shape, data_trim.shape

sns.boxplot(data_trim.Quantity)


# Replace

data['data_replace'] = pandas.DataFrame(np.where(data['UnitPrice'] > upper_limit, upper_limit, np.where(data['UnitPrice'] < lower_limit, lower_limit, data['UnitPrice'])))


sns.boxplot(data.data_replace)


data['data_replaced'] = pandas.DataFrame(np.where(data['Quantity'] > upper_limit, upper_limit, np.where(data['Quantity'] < lower_limit, lower_limit, data['Quantity'])))

sns.boxplot(data.data_replaced)


# Winsorizor

from feature_engine.outliers import Winsorizer


winsor_iqr = Winsorizer(capping_method='iqr',
                           tail= 'both',
                            fold=1.5,
                             variables=('UnitPrice'))

data_i = winsor_iqr.fit_transform(data[['UnitPrice']],)

sns.boxplot(data_i.UnitPrice)

win_iqr = Winsorizer(capping_method='iqr',
                        tail= 'both',
                         fold=1.5,
                          variables=('Quantity'))

data_q = win_iqr.fit_transform(data[['Quantity']], )

sns.boxplot(data_q.Quantity)


# gaussian method

winsor_gaussian = Winsorizer(capping_method='gaussian',
                           tail= 'both',
                            fold=3,
                             variables=('UnitPrice'))


data_g = winsor_gaussian.fit_transform(data[['UnitPrice']], )

sns.boxplot(data_g.UnitPrice)


win_gaussian = Winsorizer(capping_method='gaussian',
                           tail= 'both',
                            fold=3,
                             variables=('Quantity'))


data_gu = win_gaussian.fit_transform(data[['Quantity']], )

sns.boxplot(data_gu.Quantity)


# Percentile
# values with 95%, 5%


winsor_percentile = Winsorizer(capping_method='quantiles',
                              tail= 'both',
                               fold=0.05,
                                variables=('UnitPrice'))

data_p = winsor_percentile.fit_transform(data[['UnitPrice']],)


sns.boxplot(data_p.UnitPrice)


win_percentile = Winsorizer(capping_method='quantiles',
                              tail= 'both',
                               fold=0.05,
                                variables=('Quantity'))

data_per = win_percentile.fit_transform(data[['Quantity']],)

sns.boxplot(data_per.Quantity)



# Duplicates 

duplicates = data.duplicated()
sum(duplicates)

duplicates = data.drop_duplicates(keep = False)






