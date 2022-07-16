import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
df = sns.load_dataset("titanic")



# Görev 2: Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
df["sex"].value_counts()



# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
for col in df.columns:
    print(col, df[col].nunique())



# Görev 4: pclass değişkeninin unique değerlerinin sayısını bulunuz
df["pclass"].nunique()



# Görev 5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz
colist = ["pclass", "parch"]
for col in colist:
    print(col, df[col].nunique())



# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.
str(df["embarked"].dtype)
df["embarked"] = df["embarked"].astype("category")
str(df["embarked"].dtype)



# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
df[df["embarked"] == "C"]



# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
df[df["embarked"] != "S"]



# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz
df[(df["age"] < 30) & (df["sex"] == "female")]



# Görev 10: Fare'i 500'den büyük veya yaşı 70’den büyük yolcuların bilgilerini gösteriniz.
df[(df["fare"] > 500) | (df["age"] > 70)]



# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
df.isna().sum()



# Görev 12: who değişkenini dataframe’den çıkarınız
df.drop(columns="who", inplace=True)



# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
df["deck"].fillna(df["deck"].mode()[0], inplace=True)



# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz.
df["age"].fillna(df["age"].median(), inplace=True)



# Görev 15: survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
df.groupby("survived").agg({"sex": "count", "pclass": ["sum", "count", "mean"]})



# Görev 16: 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 verecek bir fonksiyon yazın. Yazdığınız
# fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda
# yapılarını kullanınız)
df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)



# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız
df = sns.load_dataset("tips")



# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerinin sum, min, max ve mean
# değerlerini bulunuz.
df.groupby("time").agg({"total_bill": ["sum", "min", "max"]})



# Görev 19: Day ve time’a göre total_bill değerlerinin sum, min, max ve mean değerlerini bulunuz.
df.groupby(["day", "time"]).agg({"total_bill": ["sum", "min", "max"]})



# Görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre sum, min, max ve mean
# değerlerini bulunuz.
subdf = df[(df["time"] == "Lunch") & (df["sex"] == "Female")]

subdf.groupby("day").agg({"total_bill": ["sum", "min", "max", "mean"],
                          "tip": ["sum", "min", "max", "mean"]})



# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)
df.loc[(df["size"] < 3) & (df["total_bill"] > 10), ["total_bill"]].mean()



# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği totalbill ve tip in
# toplamını versin.
def fx(x, y):
    return x+y

df["total_bill_tip_sum"] = np.vectorize(fx)(df["tip"],df["total_bill"])
df



# Görev 23: Total_bill değişkeninin kadın ve erkek için ayrı ayrı ortalamasını bulunuz. Bulduğunuz ortalamaların
# altında olanlara 0, üstünde ve eşit olanlara 1 verildiği yeni bir total_bill_flag değişkeni oluşturunuz. Kadınlar
# için Female olanlarının ortalamaları, erkekler için ise Male olanların ortalamaları dikkate alınacktır. Parametre
# olarak cinsiyet ve total_bill alan bir fonksiyon yazarak başlayınız. (If-else koşulları içerecek)
f_mean = df.loc[df["sex"] == "Female"]["total_bill"].mean()
m_mean = df.loc[df["sex"] == "Male"]["total_bill"].mean()

df["total_bill_flag"] = df.apply(lambda x: 1 if (x["sex"] == "Female" and x.total_bill > f_mean)
                                else (1 if x["sex"] == "Male" and x.total_bill > m_mean else 0),axis=1)



# Görev 24: total_bill_flag değişkenini kullanarak cinsiyetlere göre ortalamanın altında ve üstünde olanların
# sayısını gözlemleyiniz.
df.groupby(["sex","total_bill_flag"])["total_bill_flag"].count()



# Görev 25: Veriyi total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir
# dataframe'e atayınız.
first_30 = df.sort_values("total_bill_tip_sum",ascending=False)[0:30]












