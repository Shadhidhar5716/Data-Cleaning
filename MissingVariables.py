import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


data = pd.read_csv(f"C:/Users/Shashi/Documents/DataSets/claimants.csv")

data.dtypes

data.info()

data.describe()

data.mean()

data.median()

data.mode()

data.CASENUM.mean()
data.CASENUM.median()
data.CASENUM.mode()

data.ATTORNEY.mean()
data.ATTORNEY.median()
data.ATTORNEY.mode()

data.CLMSEX.mean()
data.CLMSEX.median()
data.CLMSEX.mode()

data.CLMINSUR.mean()
data.CLMINSUR.median()
data.CLMINSUR.mode()

data.SEATBELT.mean()
data.SEATBELT.median()
data.SEATBELT.mode()

data.CLMAGE.mean()
data.CLMAGE.median()
data.CLMAGE.mode()

data.LOSS.mean()
data.LOSS.median()
data.LOSS.mode()



# Graphical Representation

plt.boxplot(data['CASENUM']) # Outlier
plt.hist(data['CASENUM'])

plt.boxplot(data['ATTORNEY'])
plt.hist(data['ATTORNEY'])

plt.boxplot(data['CLMSEX'])
plt.hist(data['CLMSEX'])

plt.boxplot(data['CLMINSUR'])
plt.hist(data['CLMINSUR'])

plt.boxplot(data['SEATBELT'])
plt.hist(data['SEATBELT'])

plt.boxplot(data['CLMAGE'])
plt.hist(data['CLMAGE'])

plt.boxplot(data['LOSS']) # Outlier
plt.hist(data['LOSS'])


# Outliers

iqr = data['CASENUM'].quantile(0.75) - data['CASENUM'].quantile(0.25) 

upper_limit = data['CASENUM'].quantile(0.75) + (iqr * 1.5)
lower_limit = data['CASENUM'].quantile(0.25) - (iqr * 1.5)


outlier_data = np.where(data['CASENUM'] > upper_limit, True, np.where(data['CASENUM'] < lower_limit, True, False))


# Data_trimming

data_trimmed = data.loc[~(outlier_data)]

data.shape, data_trimmed.shape


# Replacing


data['data_replaced'] = np.where(data['CASENUM'] > upper_limit, upper_limit, np.where(data['CASENUM'] < lower_limit, lower_limit, data['CASENUM']))
sns.boxplot(data['data_replaced'])



# Winsorization

from feature_engine.outliers import Winsorizer

winsor_iqr = Winsorizer(capping_method ='iqr',
                         tail = 'both',
                          fold = 1.5,
                           variables=('CASENUM'))

data_i = winsor_iqr.fit_transform(data[['CASENUM']])

sns.boxplot(data_i.CASENUM)


winsor_gaussian = Winsorizer(capping_method = 'gaussian',
                              tail = 'both',
                               fold = 3,
                                variables=('CASENUM'))

data_g = winsor_gaussian.fit_transform(data[['CASENUM']], )

sns.boxplot(data_g['CASENUM'])


winsor_percentile = Winsorizer(capping_method = 'quantiles',
                                 tail = 'both', 
                                  fold = 0.05,
                                   variables=('CASENUM'))

data_p = winsor_percentile.fit_transform(data[['CASENUM']],)

sns.boxplot(data_p['CASENUM'])



# LOSS column

iqr1 = data['LOSS'].quantile(0.75) - data['LOSS'].quantile(0.25) 

upper_limit1 = data['LOSS'].quantile(0.75) + (iqr1 * 1.5)
lower_limit1 = data['LOSS'].quantile(0.25) - (iqr1 * 1.5)


outlier_data1 = np.where(data['LOSS'] > upper_limit1, True, np.where(data['LOSS'] < lower_limit1, True, False))


# Data_trimming

data_trimmed1 = data.loc[~(outlier_data1)]

data.shape, data_trimmed1.shape


# Replacing


data['data_replaced1'] = np.where(data['LOSS'] > upper_limit1, upper_limit1, np.where(data['LOSS'] < lower_limit1, lower_limit1, data['LOSS']))
sns.boxplot(data['data_replaced1'])



# Winsorization

from feature_engine.outliers import Winsorizer

winsor_iqr1 = Winsorizer(capping_method ='iqr',
                         tail = 'both',
                          fold = 1.5,
                           variables=('LOSS'))

data_i1 = winsor_iqr1.fit_transform(data[['LOSS']])

sns.boxplot(data_i1['LOSS'])


winsor_gaussian1 = Winsorizer(capping_method = 'gaussian',
                              tail = 'both',
                               fold = 3,
                                variables=('LOSS'))

data_g1 = winsor_gaussian1.fit_transform(data[['LOSS']], )

sns.boxplot(data_g1['LOSS'])


winsor_percentile1 = Winsorizer(capping_method = 'quantiles',
                                 tail = 'both', 
                                  fold = 0.05,
                                   variables=('LOSS'))

data_p1 = winsor_percentile1.fit_transform(data[['LOSS']],)

sns.boxplot(data_p1['LOSS'])



#  Zero Variance

data.var() == 0


# Discretization 

# Binarization

data['CASENUM_new'] = pd.cut(data['CASENUM'], 
                              bins = [min(data['CASENUM']), data['CASENUM'].mean(), max(data['CASENUM'])],
                               labels = ["Low", "High"])

data['CASENUM_new'].count()


data['CASENUM_new'] = pd.cut(data['CASENUM'], 
                              bins = [min(data['CASENUM']), data['CASENUM'].mean(), max(data['CASENUM'])],
                               include_lowest=True,
                               labels = ["Low", "High"])

data['CASENUM_new'].value_counts()
data['CASENUM_new'].count()



data['ATTORNEY_new'] = pd.cut(data['ATTORNEY'], 
                              bins = [min(data['ATTORNEY']), data['ATTORNEY'].mean(), max(data['ATTORNEY'])],
                               include_lowest=True,
                               labels = ["Low", "High"])

data['ATTORNEY_new'].value_counts()
data['ATTORNEY_new'].count()



data['CLMSEX_new'] = pd.cut(data['CLMSEX'],
                             bins=[min(data['CLMSEX']), data['CLMSEX'].mean(), max(data['CLMSEX'])],
                             include_lowest=True,
                             labels = ["Low", "High"])

#data['CLMSEX_new'].value_counts()
#data['CLMSEX_new'].count()



data['CLMINSUR_new'] = pd.cut(data['CLMINSUR'],
                             bins=[min(data['CLMINSUR']), data['CLMINSUR'].mean(), max(data['CLMINSUR'])],
                             include_lowest=True,
                             labels = ["Low", "High"])

#data['CLMINSUR_new'].value_counts()
#data['CLMINSUR_new'].count()



# Missing Values

data.isna().sum()


from sklearn.impute import SimpleImputer

mean_impute = SimpleImputer(missing_values=np.nan, strategy= 'mean',)
data['CLMSEX'] = pd.DataFrame(mean_impute.fit_transform(data[['CLMSEX']],))
data['CLMSEX'].isna().sum()


median_impute = SimpleImputer(missing_values=np.nan, strategy= 'median',)
data['CLMINSUR'] = pd.DataFrame(median_impute.fit_transform(data[['CLMINSUR']],))
data['CLMINSUR'].isna().sum()


mode_impute = SimpleImputer(missing_values=np.nan, strategy= 'mostfrequency',)
data['SEATBELT'] = pd.DataFrame(mean_impute.fit_transform(data[['SEATBELT']],))
data['SEATBELT'].isna().sum()


from feature_engine.imputation import RandomSampleImputer

random_impute = RandomSampleImputer(['CLMAGE'])
data_random = pd.DataFrame(random_impute.fit_transform(data[['CLMAGE']],))
data_random.isna().sum()



























