import pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



data = pandas.read_csv(r'C:/Users/Shashi/Documents/DataSets/Seeds_data.csv')


data.dtypes

data.shape

data.info()

data.describe()

data.columns

duplicate = data.duplicated()
duplicate.sum()


# EDA 1st B.M

data.mean()

data.Area.mean()
data.Perimeter.mean()
data.Compactness.mean()
data.length
data.Width
data.Assymetry_coeff
data.len_ker_grove
data.Type

data.median()

data.Perimeter

data.mode()


# EDA 2nd B.M

data.var()

data.std()


# EDA 3rd B.M

data.skew()


# EDA 4th B.M

data.kurt()

data.dtypes

# Graphical Representation

plt.boxplot(data)

plt.hist(data)


#outliers columns

# Compactness
# Assymetry_coeff


iqr = data['Assymetry_coeff'].quantile(0.75) - data['Assymetry_coeff'].quantile(0.25)

upper_limit = data['Assymetry_coeff'].quantile(0.75) + (iqr * 1.5)
lower_limit = data['Assymetry_coeff'].quantile(0.25) - (iqr * 1.5)


outliers_data = np.where(data['Assymetry_coeff'] > upper_limit, True, np.where(data['Assymetry_coeff'] < lower_limit, True, False))

# Trimming the data

data_trimming = data.loc[~(outliers_data),]

data.shape, data_trimming.shape


# G R

plt.boxplot(data_trimming.Assymetry_coeff)


# Replace the missing values

data['data_replace'] = pandas.DataFrame(np.where(data['Assymetry_coeff'] > upper_limit, upper_limit, np.where(data['Assymetry_coeff'] < lower_limit, lower_limit, data['Assymetry_coeff'])))
sns.boxplot(data.data_replace)

# Winsorization


from feature_engine.outliers import Winsorizer

winsor_iqr = Winsorizer(capping_method='iqr',
                           tail='both',
                            fold=1.5,
                             variables=('Assymetry_coeff'))

data_iqr = winsor_iqr.fit_transform(data[['Assymetry_coeff']],) 


sns.boxplot(data_iqr.Assymetry_coeff)


# Modeling with gaussian

winsor_gaussian = Winsorizer(capping_method='gaussian',
                           tail='both',
                            fold= 3,
                             variables=('Assymetry_coeff'))

data_g = winsor_gaussian.fit_transform(data[['Assymetry_coeff']],)


sns.boxplot(data_g.Assymetry_coeff)

# model with percentile

winsor_percentile = Winsorizer(capping_method='quantiles',
                           tail='both',
                            fold= 0.05,
                             variables=('Assymetry_coeff'))

data_p = winsor_percentile.fit_transform(data[['Assymetry_coeff']],)

sns.boxplot(data_p.Assymetry_coeff)


# Zero Variance

data.var() == 0


# Discretization

# Binarization

data['Area_new'] = pandas.cut(data['Area'],
                                bins=[min(data.Area), data.Area.mean(), max(data.Area)],
                                  labels=['Low', 'High'])


data.Area_new.value_counts()
data.Area_new.count()


data['Area_new'] = pandas.cut(data['Area'],
                                bins=[min(data.Area), data.Area.mean(), max(data.Area)],
                                  include_lowest = True,
                                   labels=['Low', 'High'])


data.Area_new.count()


data.dtypes

# checking Missing values

data.isna().sum()



# Dummy Variables

data.isna().sum()



# Transformation

import scipy.stats as stats
import pylab
data.dtypes
stats.probplot(data['Area'], dist='norm', plot=pylab)


#stats.probplot(data['Perimeter'], dist='norm', plot=pylab)


stats.probplot(data['Compactness'], dist='norm', plot=pylab)

stats.probplot(data['length'], dist='norm', plot=pylab)


stats.probplot(data['Width'], dist='norm', plot=pylab)

# creating a model

fitted_data, fitted_lambda = stats.boxcox(data['Compactness'])

fig, ax = plt.subplots(1, 2)

sns.distplot(data['Compactness'], hist=False, kde=True,
             kde_kws={'shade' : True, 'linewidth' : 2},
             label = 'Non-Normal', color = 'green', ax = ax[0])


sns.distplot(fitted_data, hist=False, kde=True,
             kde_kws={'shade' : True, 'linewidth' : 2},
             label = 'Normal', color = 'green', ax = ax[1])

plt.legend(loc = 'upper right')

fig.set_figheight(5)
fig.set_figwidth(10)


print(f"Lamda value used for Transformation : {fitted_lambda}")


# YeoJhonson Transformation

prob = stats.probplot(data['Compactness'], dist = stats.norm, plot=pylab)

from feature_engine import transformation

tf = transformation.YeoJohnsonTransformer(variables=('Compactness'))

data_tf = tf.fit_transform(data)

# Transformation

prob = stats.probplot(data_tf['Compactness'], dist = stats.norm, plot = pylab)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df = scaler.fit_transform(data,)
dataset = pandas.DataFrame(df)
res = dataset.describe()


from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()

df = minmax.fit_transform(data,)

dataset = pandas.DataFrame(df)

MinMax_res = dataset.describe()















