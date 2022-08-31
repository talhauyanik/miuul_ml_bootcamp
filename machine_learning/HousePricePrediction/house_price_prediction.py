import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate,cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("HousePrice/train.csv")


#####################################
# Veriye Genel Bakış
#####################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)



#####################################
# Numerik ve Kategorik Değişken Analizi
#####################################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].dtype == "O" and dataframe[col].nunique() > car_th]

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].dtype != "O" and dataframe[col].nunique() < cat_th]

    num_cols = num_cols + cat_but_car
    num_cols = [col for col in num_cols if col not in num_but_cat]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return num_cols, cat_cols, cat_but_car


numcols, cat_cols, cat_but_car = grab_col_names(df, cat_th=11)

cat_cols
numcols
df["Neighborhood"].value_counts()




#####################################
# Aykırı Değerlerin İncelenmesi
#####################################
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


outlier_thresholds(df, "GarageArea")
out_index = grab_outliers(df, "GarageArea", index=True)


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


num_cols = [col for col in numcols if col not in ["Neighborhood", "Id"]]

outlier_list = []
for col in num_cols:
    if check_outlier(df, col):
        print(col, "True")
        outlier_list.append(col)
    else:
        print(col, "False")

check_outlier(df, "YearRemodAdd")

sns.boxplot(data=df, x="SalePrice")



#####################################
# Eksik Gözlem Analizi
#####################################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


na_cols = missing_values_table(df, na_name=True)
na_cols = [col for col in na_cols if "Electrical" not in col]



#####################################
# Eksik Değerlerin Doldurulması
#####################################
df.drop(columns=na_cols, inplace=True)
df.drop(df.loc[df["Electrical"].isnull()].index, inplace=True)
df.isnull().sum().max()



#####################################
# Kategorik Değişkenler için Encoding
#####################################
df = pd.get_dummies(df, drop_first=True)



#####################################
# Model ve Tahmin
#####################################

y = df["SalePrice"]
X = df.drop("SalePrice", axis=1)


def base_reg_models(X, y, scoring="neg_mean_squared_error"):
    print("Base Regression Models....")
    regressors = [('LR', LinearRegression()),
                  ("DT", DecisionTreeRegressor()),
                  ("RF", RandomForestRegressor(verbose=False)),
                  ('GB', GradientBoostingRegressor(verbose=False)),
                  ('ADA', AdaBoostRegressor()),
                  ('XGB', XGBRegressor()),
                  ('LGBM', LGBMRegressor()),
                  ('CB', CatBoostRegressor(verbose=False))
                  ]

    for name, regressor in regressors:

        cv_result = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=3, scoring=scoring)))

        print(f"{scoring}: {round(cv_result, 4)} ({name}) ")


base_reg_models(X, y)


# Base Regression Models....
# neg_mean_squared_error: 40808.238 (LR)
# neg_mean_squared_error: 45166.9159 (DT)
# neg_mean_squared_error: 30233.1513 (RF)
# neg_mean_squared_error: 26781.5047 (GB)
# neg_mean_squared_error: 35539.6148 (ADA)
# neg_mean_squared_error: 30158.8191 (XGB)
# neg_mean_squared_error: 28984.4983 (LGBM)
# neg_mean_squared_error: 25836.3413 (CB)

