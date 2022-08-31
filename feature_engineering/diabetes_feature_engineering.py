import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)

df = pd.read_csv("datasets/diabetes.csv")


##############################################################
# Adım 1: Genel resmi inceleyiniz.
##############################################################
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


##############################################################
# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
##############################################################
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


num_cols, cat_cols, cat_but_car = grab_col_names(df)


##############################################################
# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
##############################################################
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


cat_summary(df, "Outcome")

num_summary(df, "Glucose")

##############################################################
# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)
##############################################################
not_target = [col for col in df.columns if col not in "Outcome"]

df.groupby("Outcome")[not_target].mean()


##############################################################
# Adım 5: Aykırı gözlem analizi yapınız.
##############################################################
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


for col in df.columns:
    low, up = outlier_thresholds(df, col)

    low_count = df[df[col] < low][col].count()
    up_count = df[df[col] > up][col].count()

    low_up_ratio = round((low_count + up_count) / len(df) * 100, 3)
    low_up_count = low_count + up_count

    print(col, "\noutlier ratio:", low_up_ratio, "\noutlier count:", low_up_count, "\n")


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in df.columns:

    result = check_outlier(df, col)
    if result:
        print(col, result)
        sns.boxplot(x=df[col])
        plt.show(block=True)


##############################################################
# Adım 6: Eksik gözlem analizi yapınız.
##############################################################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

##############################################################
# Adım 6: Korelasyon analizi yapınız.
##############################################################
df.corr()
sns.set(font_scale=1)
sns.heatmap(df.corr(), cmap="RdBu", annot=True)
plt.show()

##############################################################
# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz,
# Insulin vb. değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin
# glikoz veya insulin değeri 0 olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak
# atama yapıp sonrasında eksik değerlere işlemleri uygulayabilirsiniz.
##############################################################
zero_col = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df.isin([0]).sum()

for col in zero_col:
    df.loc[df[col] == 0, col] = np.nan

for col in zero_col:
    df.loc[df[col].isnull(), col] = df[col].median()

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
scores = pd.DataFrame(np.sort(df_scores))

scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

th = np.sort(df_scores)[3]
df[df_scores < th]
df[df_scores < th].index

df = df.drop(axis=0, labels=df[df_scores < th].index)

##############################################################
# Adım 2: Yeni değişkenler oluşturunuz.
##############################################################

df["NEW_Age"] = pd.cut(x=df["Age"], bins=[20, 30, 45, 100],
                       labels=["YoungAdults", "MiddleAgedAdults", "OldAdults"])

df["NEW_BMI"] = pd.cut(x=df["BMI"], bins=[0, 18.5, 25, 30, 100],
                       labels=["Underweight", "Optimal", "Overweight", "Obese"])

df["NEW_Insulin"] = pd.cut(x=df["Insulin"], bins=[1, 166, 1000],
                           labels=["Normal", "Abnormal"])

df["NEW_Glucose"] = pd.cut(x=df["Glucose"], bins=[0, 140, 199, 500],
                           labels=["Normal", "Prediabetes", "Diabetes"])

df["NEW_BloodPressure"] = pd.cut(x=df["BloodPressure"], bins=[0, 80, 89, 150],
                                  labels=["Normal", "Stage1Hypertension", "Stage2Hypertension"])


df["NEW_Insulin*BMI"] = df["Insulin"] * df["BMI"]



##############################################################
# Adım 3: Encoding işlemlerini gerçekleştiriniz.
##############################################################

num_cols,cat_cols,cat_but_car = grab_col_names(df)
cat_cols

encode_col = [col for col in cat_cols if col != "Outcome"]

df = pd.get_dummies(df, columns=encode_col, drop_first=True)




##############################################################
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
##############################################################

cols = [col for col in df.columns if col != "Outcome"]
for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])


##############################################################
# Adım 5: Model oluşturunuz.
##############################################################

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=22)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=50).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)