import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

pd.set_option("display.width",500)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows",None)
pd.set_option("display.float_format",lambda x: "%.3f" % x)

df = pd.read_csv("datasets/hitters.csv")



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

num_cols,cat_cols,catbutcar = grab_col_names(df)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

num_summary(df,"AtBat",plot=True)

num_cols = [col for col in df.columns if df[col].dtypes != "O"]

for col in num_cols:
    num_summary(df,col,plot=True)



def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_cols = [col for col in df.columns if df[col].dtypes == "O"]

for col in cat_cols:
    cat_summary(df,col,plot=True)



#####################################
# Hedef Değişken Analizi
#####################################
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)


def target_summary_with_num(dataframe, target, numerical_col):
    sns.scatterplot(x= dataframe[target],y= dataframe[numerical_col])
    plt.show(block=True)


for col in num_cols:
    target_summary_with_num(df, "Salary", col)



#####################################
# Aykırı Değerlerin İncelenmesi
#####################################
def outlier_thresholds(dataframe, col_name,q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 -quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit,up_limit

for col in num_cols:

    low,up = outlier_thresholds(df,col,q1=0.25, q3=0.75)

    l = df[df[col] < low][col].count()
    u = df[df[col] > up][col].count()
    lu = round((l+u) / len(df)*100,3)
    luc = l+u
    print(col,"\nratio:",lu ,"\ncount:",luc,"\n")


for col in num_cols:
    sns.boxplot(x=df[col])
    plt.show(block=True)


#####################################
# Aykırı Değerlerin Değiştirilmesi
#####################################
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


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

missing_values_table(df)


#####################################
# Kategorik Değişkenler için Encoding
#####################################
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in cat_cols:
    label_encoder(df,col)


#####################################
# Eksik Değerlerin Doldurulması
#####################################
imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df),columns=df.columns)

sns.histplot(data=df,x="Salary")


#####################################
# Korelasyon Analizi
#####################################
corr = df.corr()
sns.set(font_scale=1)
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(df.corr(), cmap=cmap,annot=True, square=True,mask=mask)
plt.show()


#####################################
# Model ve Tahmin
#####################################
y = df["Salary"]
X = df.drop("Salary",axis=1)

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
# neg_mean_squared_error: 313.6077 (LR)
# neg_mean_squared_error: 346.1561 (DT)
# neg_mean_squared_error: 265.6919 (RF)
# neg_mean_squared_error: 272.3365 (GB)
# neg_mean_squared_error: 295.3559 (ADA)
# neg_mean_squared_error: 290.9605 (XGB)
# neg_mean_squared_error: 278.7652 (LGBM)
# neg_mean_squared_error: 260.5026 (CB)