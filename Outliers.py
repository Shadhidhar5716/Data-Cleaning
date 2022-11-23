import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np


data = pd.read_csv(r'C:/Users/Shashi/Documents/DataSets/Boston.csv')

data.dtypes

data.info()

data.describe()


# 1st B M

data.mean()

data.median()

data.mode()


## 2nd B M

data.var()

data.std()


### 3rd B M

data.skew()


#### 4th B M

data.kurt()



##### Graphical Representation

plt.boxplot(data)

plt.hist(data)

data.dtypes

plt.boxplot(data['crim'])
sns.boxplot(data['crim'])
plt.hist(data['crim'])


plt.boxplot(data['zn'])
sns.boxplot(data['zn'])


plt.boxplot(data['black'])
sns.boxplot(data['black'])


plt.boxplot(data['medv'])
sns.boxplot(data['medv'])

plt.boxplot(data['indus'])
plt.hist(data['indus'])


plt.boxplot(data['chas']) # Outlier
plt.hist(data['chas'])

plt.boxplot(data['nox'])
plt.hist(data['nox'])


plt.boxplot(data['rm']) # Outlier
plt.hist(data['rm'])


plt.boxplot(data['age'])
plt.hist(data['age'])


plt.boxplot(data['dis']) # Outlier
plt.hist(data['dis'])


plt.boxplot(data['rad'])
plt.hist(data['rad'])


plt.boxplot(data['tax'])
plt.hist(data['tax'])


plt.boxplot(data['ptratio']) # Outlier
plt.hist(data['ptratio'])


plt.boxplot(data['lstat']) # Outlier
plt.hist(data['lstat'])


plt.boxplot(data['Unnamed: 0'])
plt.hist(data['Unnamed: 0'])



###### Outliers columns

# crim
# zn
# ptratio
# lstat
# dis
# rm
# chas
# black
# medv


iqr = data['crim'].quantile(0.75) - data['crim'].quantile(0.25)

upper_limit = data['crim'].quantile(0.75) + (iqr * 1.5) 
lower_limit = data['crim'].quantile(0.25) - (iqr * 1.5)


outlier_data = np.where(data['crim'] > upper_limit, True, np.where(data['crim'] < lower_limit, True, False))


###### Data Trimming

data_trimming = data.loc[~(outlier_data),]

data.shape, data_trimming.shape


sns.boxplot(data_trimming.crim)


###### Data Replaced

data['data_replaced'] = pd.DataFrame(np.where(data['crim'] > upper_limit, upper_limit, np.where(data['crim'] < lower_limit, lower_limit, data['crim'])))

sns.boxplot(data.data_replaced)



###### Winsorization

from feature_engine.outliers import Winsorizer


winsor_iqr = Winsorizer(capping_method='iqr',
                          tail = 'both',
                           fold = 1.5,
                            variables=('crim'))

data_i = winsor_iqr.fit_transform(data[['crim']], )

sns.boxplot(data_i.crim)


###### Creating a model with gaussian


winsor_gaussian = Winsorizer(capping_method='gaussian',
                                tail = 'both',
                                 fold = 3,
                                  variables=('crim'))

data_g = winsor_gaussian.fit_transform(data[['crim']],)

sns.boxplot(data_g.crim)


####### Percentile


winsor_percentile = Winsorizer(capping_method='quantiles',
                                  tail = 'both',
                                   fold = 0.05,
                                    variables=('crim'))


data_p = winsor_percentile.fit_transform(data[['crim']],)


sns.boxplot(data_p.crim)


###### Column 2nd "" zn ""

iqr1 = data['zn'].quantile(0.75) - data['zn'].quantile(0.25)

upper_limit1 = data['zn'].quantile(0.75) + (iqr1 * 1.5) 
lower_limit1 = data['zn'].quantile(0.25) - (iqr1 * 1.5)


outlier_data1 = np.where(data['zn'] > upper_limit1, True, np.where(data['zn'] < lower_limit1, True, False))


###### Data Trimming

data_trimming1 = data.loc[~(outlier_data1),]

data.shape, data_trimming1.shape


sns.boxplot(data_trimming1.zn)


###### Data Replaced

data['data_replaced1'] = pd.DataFrame(np.where(data['zn'] > upper_limit1, upper_limit1, np.where(data['zn'] < lower_limit1, lower_limit1, data['zn'])))

sns.boxplot(data.data_replaced1)



###### Winsorization

from feature_engine.outliers import Winsorizer


winsor_iqr1 = Winsorizer(capping_method='iqr',
                          tail = 'both',
                           fold = 1.5,
                            variables=('zn'))

data_i1 = winsor_iqr1.fit_transform(data[['zn']], )

sns.boxplot(data_i1.zn)


###### Creating a model with gaussian

winsor_gaussian1 = Winsorizer(capping_method='gaussian',
                                tail = 'both',
                                 fold = 3,
                                  variables=('zn'))

data_g1 = winsor_gaussian1.fit_transform(data[['zn']],)

sns.boxplot(data_g1.zn)


####### Percentile

winsor_percentile1 = Winsorizer(capping_method='quantiles',
                                  tail = 'both',
                                   fold = 0.05,
                                    variables=('zn'))


data_p1 = winsor_percentile1.fit_transform(data[['zn']],)


sns.boxplot(data_p1.zn)



# 3rd column "" ptratio ""

iqr2 = data['ptratio'].quantile(0.75) - data['ptratio'].quantile(0.25)

upper_limit2 = data['ptratio'].quantile(0.75) + (iqr2 * 1.5) 
lower_limit2 = data['ptratio'].quantile(0.25) - (iqr2 * 1.5)

outlier_data2 = np.where(data['ptratio'] > upper_limit2, True, np.where(data['ptratio'] < lower_limit2, True, False))


###### Data Trimming

data_trimming2 = data.loc[~(outlier_data2),]

data.shape, data_trimming2.shape


sns.boxplot(data_trimming2.ptratio)


###### Data Replaced

data['data_replaced2'] = pd.DataFrame(np.where(data['ptratio'] > upper_limit2, upper_limit2, np.where(data['ptratio'] < lower_limit2, lower_limit2, data['ptratio'])))

sns.boxplot(data.data_replaced2)



###### Winsorization

from feature_engine.outliers import Winsorizer


winsor_iqr2 = Winsorizer(capping_method='iqr',
                          tail = 'both',
                           fold = 1.5,
                            variables=('ptratio'))

data_i2 = winsor_iqr2.fit_transform(data[['ptratio']], )

sns.boxplot(data_i2.ptratio)


###### Creating a model with gaussian

winsor_gaussian2 = Winsorizer(capping_method='gaussian',
                                tail = 'both',
                                 fold = 3,
                                  variables=('ptratio'))

data_g2 = winsor_gaussian2.fit_transform(data[['ptratio']],)

sns.boxplot(data_g2.ptratio)


####### Percentile

winsor_percentile2 = Winsorizer(capping_method='quantiles',
                                  tail = 'both',
                                   fold = 0.05,
                                    variables=('ptratio'))


data_p2 = winsor_percentile2.fit_transform(data[['ptratio']],)

sns.boxplot(data_p2.ptratio)



# 4th Cloumn "" lstat ""

iqr3 = data['lstat'].quantile(0.75) - data['lstat'].quantile(0.25)

upper_limit3 = data['lstat'].quantile(0.75) + (iqr3 * 1.5) 
lower_limit3 = data['lstat'].quantile(0.25) - (iqr3 * 1.5)

outlier_data3 = np.where(data['lstat'] > upper_limit3, True, np.where(data['lstat'] < lower_limit3, True, False))


###### Data Trimming

data_trimming3 = data.loc[~(outlier_data3),]

data.shape, data_trimming3.shape


sns.boxplot(data_trimming3.lstat)


###### Data Replaced

data['data_replaced3'] = pd.DataFrame(np.where(data['lstat'] > upper_limit3, upper_limit3, np.where(data['lstat'] < lower_limit3, lower_limit3, data['lstat'])))

sns.boxplot(data.data_replaced3)



###### Winsorization

from feature_engine.outliers import Winsorizer


winsor_iqr3 = Winsorizer(capping_method='iqr',
                          tail = 'both',
                           fold = 1.5,
                            variables=('lstat'))

data_i3 = winsor_iqr3.fit_transform(data[['lstat']], )

sns.boxplot(data_i3.lstat)


###### Creating a model with gaussian

winsor_gaussian3 = Winsorizer(capping_method='gaussian',
                                tail = 'both',
                                 fold = 3,
                                  variables=('lstat'))

data_g3 = winsor_gaussian3.fit_transform(data[['lstat']],)

sns.boxplot(data_g3.lstat)


####### Percentile

winsor_percentile3 = Winsorizer(capping_method='quantiles',
                                  tail = 'both',
                                   fold = 0.05,
                                    variables=('lstat'))


data_p3 = winsor_percentile3.fit_transform(data[['lstat']],)

sns.boxplot(data_p3.lstat)


# 5th Column "" dis ""

iqr4 = data['dis'].quantile(0.75) - data['dis'].quantile(0.25)

upper_limit4 = data['dis'].quantile(0.75) + (iqr4 * 1.5) 
lower_limit4 = data['dis'].quantile(0.25) - (iqr4 * 1.5)


outlier_data4 = np.where(data['dis'] > upper_limit4, True, np.where(data['dis'] < lower_limit4, True, False))


###### Data Trimming

data_trimming4 = data.loc[~(outlier_data4),]

data.shape, data_trimming4.shape


sns.boxplot(data_trimming4.lstat)


###### Data Replaced

data['data_replaced4'] = pd.DataFrame(np.where(data['dis'] > upper_limit4, upper_limit4, np.where(data['dis'] < lower_limit4, lower_limit4, data['dis'])))

sns.boxplot(data.data_replaced4)



###### Winsorization

from feature_engine.outliers import Winsorizer


winsor_iqr4 = Winsorizer(capping_method='iqr',
                          tail = 'both',
                           fold = 1.5,
                            variables=('dis'))

data_i4 = winsor_iqr4.fit_transform(data[['dis']], )

sns.boxplot(data_i4.dis)


###### Creating a model with gaussian

winsor_gaussian4 = Winsorizer(capping_method='gaussian',
                                tail = 'both',
                                 fold = 3,
                                  variables=('dis'))

data_g4 = winsor_gaussian4.fit_transform(data[['dis']],)

sns.boxplot(data_g4.dis)


####### Percentile

winsor_percentile4 = Winsorizer(capping_method='quantiles',
                                  tail = 'both',
                                   fold = 0.05,
                                    variables=('dis'))


data_p4 = winsor_percentile4.fit_transform(data[['dis']],)

sns.boxplot(data_p4.dis)



# 6th Column "" rm ""

iqr5 = data['rm'].quantile(0.75) - data['rm'].quantile(0.25)

upper_limit5 = data['dis'].quantile(0.75) + (iqr5 * 1.5) 
lower_limit5 = data['dis'].quantile(0.25) - (iqr5 * 1.5)



outlier_data5 = np.where(data['rm'] > upper_limit5, True, np.where(data['rm'] < lower_limit5, True, False))


###### Data Trimming

data_trimming5 = data.loc[~(outlier_data5),]

data.shape, data_trimming5.shape


sns.boxplot(data_trimming5.rm)


###### Data Replaced

data['data_replaced5'] = pd.DataFrame(np.where(data['rm'] > upper_limit5, upper_limit5, np.where(data['rm'] < lower_limit5, lower_limit5, data['rm'])))

sns.boxplot(data.data_replaced5)



###### Winsorization

from feature_engine.outliers import Winsorizer


winsor_iqr5 = Winsorizer(capping_method='iqr',
                          tail = 'both',
                           fold = 1.5,
                            variables=('rm'))

data_i5 = winsor_iqr5.fit_transform(data[['rm']], )

sns.boxplot(data_i5.rm)


###### Creating a model with gaussian

winsor_gaussian5 = Winsorizer(capping_method='gaussian',
                                tail = 'both',
                                 fold = 3,
                                  variables=('rm'))

data_g5 = winsor_gaussian5.fit_transform(data[['rm']],)

sns.boxplot(data_g5.rm)


####### Percentile

winsor_percentile5 = Winsorizer(capping_method='quantiles',
                                  tail = 'both',
                                   fold = 0.05,
                                    variables=('rm'))


data_p5 = winsor_percentile5.fit_transform(data[['rm']],)

sns.boxplot(data_p5.rm)



# 7th column "" chas ""

iqr6 = data['chas'].quantile(0.75) - data['chas'].quantile(0.25)

upper_limit6 = data['chas'].quantile(0.75) + (iqr6 * 1.5) 
lower_limit6 = data['chas'].quantile(0.25) - (iqr6 * 1.5)


outlier_data6 = np.where(data['chas'] > upper_limit6, True, np.where(data['chas'] < lower_limit6, True, False))


###### Data Trimming

data_trimming6 = data.loc[~(outlier_data6),]

data.shape, data_trimming6.shape


sns.boxplot(data_trimming6.chas)


###### Data Replaced

data['data_replaced6'] = pd.DataFrame(np.where(data['chas'] > upper_limit6, upper_limit6, np.where(data['chas'] < lower_limit6, lower_limit6, data['chas'])))

sns.boxplot(data.data_replaced6)



###### Winsorization

from feature_engine.outliers import Winsorizer


winsor_iqr6 = Winsorizer(capping_method='iqr',
                          tail = 'both',
                           fold = 1.5,
                            variables=('chas'))

data_i6 = winsor_iqr6.fit_transform(data[['chas']], )

sns.boxplot(data_i6.chas)


###### Creating a model with gaussian

winsor_gaussian6 = Winsorizer(capping_method='gaussian',
                                tail = 'both',
                                 fold = 3,
                                  variables=('chas'))

data_g6 = winsor_gaussian6.fit_transform(data[['chas']],)

sns.boxplot(data_g6.chas)


####### Percentile

winsor_percentile6 = Winsorizer(capping_method='quantiles',
                                  tail = 'both',
                                   fold = 0.05,
                                    variables=('chas'))


data_p6 = winsor_percentile6.fit_transform(data[['chas']],)

sns.boxplot(data_p6.chas)



# 8th column "" black "" 

iqr7 = data['black'].quantile(0.75) - data['black'].quantile(0.25)

upper_limit6 = data['black'].quantile(0.75) + (iqr7 * 1.5) 
lower_limit6 = data['black'].quantile(0.25) - (iqr7 * 1.5)


outlier_data7 = np.where(data['black'] > upper_limit, True, np.where(data['black'] < lower_limit, True, False))


###### Data Trimming

data_trimming7 = data.loc[~(outlier_data7),]

data.shape, data_trimming7.shape


sns.boxplot(data_trimming7.black)


###### Data Replaced

data['data_replaced7'] = pd.DataFrame(np.where(data['black'] > upper_limit, upper_limit, np.where(data['black'] < lower_limit, lower_limit, data['black'])))

sns.boxplot(data.data_replaced7)



###### Winsorization

from feature_engine.outliers import Winsorizer


winsor_iqr7 = Winsorizer(capping_method='iqr',
                          tail = 'both',
                           fold = 1.5,
                            variables=('black'))

data_i7 = winsor_iqr7.fit_transform(data[['black']], )

sns.boxplot(data_i7.black)


###### Creating a model with gaussian

winsor_gaussian7 = Winsorizer(capping_method='gaussian',
                                tail = 'both',
                                 fold = 3,
                                  variables=('black'))

data_g7 = winsor_gaussian7.fit_transform(data[['black']],)

sns.boxplot(data_g7.black)


####### Percentile

winsor_percentile7 = Winsorizer(capping_method='quantiles',
                                  tail = 'both',
                                   fold = 0.05,
                                    variables=('black'))


data_p7 = winsor_percentile7.fit_transform(data[['black']],)

sns.boxplot(data_p7.black)


# 9th column "" medv ""

iqr8 = data['medv'].quantile(0.75) - data['medv'].quantile(0.25)

upper_limit8 = data['medv'].quantile(0.75) + (iqr8 * 1.5) 
lower_limit8 = data['medv'].quantile(0.25) - (iqr8 * 1.5)

outlier_data8 = np.where(data['medv'] > upper_limit8, True, np.where(data['medv'] < lower_limit8, True, False))


###### Data Trimming

data_trimming8 = data.loc[~(outlier_data8),]

data.shape, data_trimming8.shape


sns.boxplot(data_trimming8.medv)


###### Data Replaced

data['data_replaced8'] = pd.DataFrame(np.where(data['medv'] > upper_limit8, upper_limit8, np.where(data['medv'] < lower_limit8, lower_limit8, data['medv'])))

sns.boxplot(data.data_replaced8)



###### Winsorization

from feature_engine.outliers import Winsorizer


winsor_iqr8 = Winsorizer(capping_method='iqr',
                          tail = 'both',
                           fold = 1.5,
                            variables=('medv'))

data_i8 = winsor_iqr8.fit_transform(data[['medv']], )

sns.boxplot(data_i8.medv)


###### Creating a model with gaussian

winsor_gaussian8 = Winsorizer(capping_method='gaussian',
                                tail = 'both',
                                 fold = 3,
                                  variables=('medv'))

data_g8 = winsor_gaussian8.fit_transform(data[['medv']],)

sns.boxplot(data_g8.medv)


####### Percentile

winsor_percentile8 = Winsorizer(capping_method='quantiles',
                                  tail = 'both',
                                   fold = 0.05,
                                    variables=('medv'))


data_p8 = winsor_percentile8.fit_transform(data[['medv']],)

sns.boxplot(data_p8.medv)

















































 