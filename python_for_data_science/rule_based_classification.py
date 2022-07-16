import pandas as pd
import numpy as np

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df = pd.read_csv("persona.csv")

df.info()
df.head()

for col in df.columns:
    print(col, df[col].unique())



# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
df["SOURCE"].value_counts()



# Soru 3: Kaç unique PRICE vardır?
df["PRICE"].nunique()



# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts()



# Soru 5: Hangi ülkeden kaçar tane satış olmuş?
df["COUNTRY"].value_counts()



# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby("COUNTRY")["PRICE"].sum()



# Soru 7: SOURCE türlerine göre satış sayıları nedir?
df["SOURCE"].value_counts()



# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
df.groupby("COUNTRY")["PRICE"].mean()



# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby("SOURCE")["PRICE"].mean()



# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(["COUNTRY", "SOURCE"])["PRICE"].mean()



# Görev 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])["PRICE"].mean()



# Görev 3: Çıktıyı PRICE’a göre sıralayınız.
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])["PRICE"].mean().sort_values(ascending=False)



# Görev 4: Indekste yer alan isimleri değişken ismine çeviriniz
agg_df = agg_df.reset_index()



# Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], [0, 18, 23, 30, 40, 70], labels=["0_18", "19_23", "24_30", "31_40", "41_70"])
agg_df.info()



# Görev 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız
agg_df["AGE_CAT"] = agg_df["AGE_CAT"].astype(str)

agg_df["customer_level_based"] = [
    agg_df["COUNTRY"][index].upper() + "_" + agg_df["SOURCE"][index].upper() + "_" + agg_df["SEX"][
        index].upper() + "_" + agg_df["AGE_CAT"][index] for index in range(0, len(agg_df))]

agg_df = pd.DataFrame(agg_df.groupby("customer_level_based")["PRICE"].mean())
agg_df.reset_index(inplace=True)



# Görev 7: Yeni müşterileri (personaları) segmentlere ayırınız.
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])

agg_df.groupby("SEGMENT").agg({"PRICE" : ["mean", "max", "sum"]})



# Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.
new_user = "TUR_ANDROID_FEMALE_31_40"
new_user2 = "FRA_IOS_FEMALE_31_40"

agg_df[agg_df["customer_level_based"] == new_user]
agg_df[agg_df["customer_level_based"] == new_user2]

