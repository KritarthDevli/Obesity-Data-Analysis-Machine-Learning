import pandas as pd
import matplotlib.pyplot as plt
from category_encoders.ordinal import OrdinalEncoder
df=pd.read_csv("obesitydataset_v1 (1).csv")

print(df.head())
print(df.info())
print(df.describe())

df['NObeyesdad'].value_counts().plot(kind='bar')

df['Gender'].hist(bins=40)

df['Age'].hist(bins=40)

df['Height'].hist(bins=40)

df['Weight'].hist(bins=40)

df['BMI'] = df['Weight']/(df['Height']*df['Height'])

plt.rc('figure', figsize=(15,10))
df.hist('BMI', by='NObeyesdad')
plt.rc('figure', figsize=(5,3))

print(df)

maplist = [{'col': 'Gender',
 'mapping': {'Male': 0, 'Female': 1}}]
oenc = OrdinalEncoder(mapping=maplist)
df = oenc.fit_transform(df)
df.head()

maplist = [{'col': 'family_history_with_overweight',
 'mapping': {'yes': 1, 'no': 0}},
 {'col': 'highcal_intake',
 'mapping': {'yes': 1, 'no': 0}},
 {'col': 'SMOKE',
 'mapping': {'yes': 1, 'no': 0}},
 {'col': 'track_cal_intake',
 'mapping': {'yes': 1, 'no': 0}},
 {'col': 'snacking',
 'mapping': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}},
 {'col': 'alcohol_intake',
 'mapping': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}}
 ]
oenc = OrdinalEncoder(mapping=maplist)
data_df = oenc.fit_transform(data_df)
data_df.head()

print(df.head())

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, dtype='int64')
one_hot_encoded = encoder.fit_transform(df[['transport_mode']])
one_hot_df = pd.DataFrame(one_hot_encoded,
 columns=encoder.get_feature_names_out(['transport_mode']))
df = pd.concat([df, one_hot_df], axis=1)
df = df.drop(['transport_mode'], axis=1)
df.head()

df.to_excel("obesitydataset_v1_pre.xlsx", index=False)

