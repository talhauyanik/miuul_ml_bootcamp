import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("datasets/telco.csv")



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

cat_but_car

df.drop(columns="customerID", inplace=True)
num_cols, cat_cols, cat_but_car = grab_col_names(df)
df.loc[df["TotalCharges"] == " ", "TotalCharges"] = np.nan

df[df["tenure"] == 0]

df["TotalCharges"] = df["TotalCharges"].astype(dtype=float)

df.info()



##############################################################
# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız
##############################################################
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col)



##############################################################
# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)
##############################################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Churn", col)

df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)



##############################################################
# Adım 5: Aykırı gözlem analizi yapınız
##############################################################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        print(col_name, "True")
    else:
        print(col_name, "False")


for col in num_cols:
    check_outlier(df, col)

for col in num_cols:
    sns.boxplot(df[col])
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


missing_values_table(df, na_name=True)



##############################################################
# Adım 7: Korelasyon analizi yapınız.
##############################################################

df.info()
corr = df.corr()
palette = sns.diverging_palette(20, 220, n=256)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, cmap=palette, mask=mask)



##############################################################
# Görev 2: Feature Engineering
# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız
##############################################################

df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

df.isnull().sum()



##############################################################
# Adım 2: Yeni değişkenler oluşturunuz.
##############################################################

df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"

# Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)

# Herhangi bir destek, yedek veya koruma almayan kişiler
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (
            x["TechSupport"] != "Yes") else 0, axis=1)

# Aylık sözleşmesi bulunan ve genç olan müşteriler
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0,
                                       axis=1)

# Kişinin toplam aldığı servis sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                               'OnlineBackup', 'DeviceProtection', 'TechSupport',
                               'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)

# Herhangi bir streaming hizmeti alan kişiler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(
    lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Kişi otomatik ödeme yapıyor mu?
df["PaymentMethod"].unique()
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(
    lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)

# ortalama aylık ödeme
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 0.1)

# Güncel Fiyatın ortalama fiyata göre artışı
df["NEW_Increase"] = df["NEW_AVG_Charges"] / (df["MonthlyCharges"] + 1)

# Servis başına ücret
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)



##############################################################
# Adım 3: Encoding işlemlerini gerçekleştiriniz.
##############################################################

num_cols, cat_cols, cat_but_car = grab_col_names(df)


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols)



##############################################################
# Adım 4: Numerik değişkenler için standartlaştırma yapınız
##############################################################

rs = RobustScaler()

df[num_cols] = rs.fit_transform(df[num_cols])

df[num_cols].head()



##############################################################
# Görev 3 : Modelleme
# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.
##############################################################
y = df["Churn_1"]
X = df.drop(columns=["Churn_1"])

def base_class_models(X, y, scoring="accuracy"):
    print("Base Classifier Models....")
    classifiers = [('Logistic Regression', LogisticRegression()),
                  ('KNN', KNeighborsClassifier()),
                  ("Decision Tree", DecisionTreeClassifier()),
                  ("Random Forest", RandomForestClassifier(verbose=False)),
                  ('Gradient Boosting', GradientBoostingClassifier(verbose=False)),
                  ('ADABoost', AdaBoostClassifier()),
                  ('XGBoost', XGBClassifier()),
                  ('LGBM', LGBMClassifier()),
                  ('CatBoost', CatBoostClassifier(verbose=False))
                  ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=5, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


base_class_models(X, y)



##############################################################
# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz hiparparametreler ile modeli
# tekrar kurunuz.
##############################################################

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "max_depth": [5, 8],
                   "n_estimators": [300, 500]}

gradientboost_params = {"learning_rate": [0.1, 0.01],
                   "max_depth": [5, 7],
                   "n_estimators": [200, 500]}

adaboost_params = {"learning_rate": [0.1, 0.01],
                   "n_estimators": [200, 500]}

catboost_params = {"learning_rate": [0.01, 0.1],
                   "max_depth": [5, 7],
                   "n_estimators": [300, 500]}


classifiers = [('LGBM', LGBMClassifier(), lightgbm_params),
               ("Gradient Boosting",  GradientBoostingClassifier(verbose=False), gradientboost_params),
               ("ADABoost", AdaBoostClassifier(), adaboost_params),
               ('CatBoost', CatBoostClassifier(verbose=False), catboost_params)]


def hyperparameter_optimization(X, y, cv=3, scoring="accuracy"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y)

########### LGBM ##########
#accuracy (Before): 0.792
#accuracy (After): 0.8035
#LGBM best params: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 300}

########## Gradient Boosting ##########
#accuracy (Before): 0.8012
#accuracy (After): 0.8029
#Gradient Boosting best params: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 200}

########## ADABoost ##########
#accuracy (Before): 0.7995
#accuracy (After): 0.8022
#ADABoost best params: {'learning_rate': 0.1, 'n_estimators': 200}

########## CatBoost ##########
#accuracy (Before): 0.8011
#accuracy (After): 0.8043
#CatBoost best params: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 500}