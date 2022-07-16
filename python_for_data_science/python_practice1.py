# Görev 1:  Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz,
# kelime kelime ayırınız.

text = "The goal is to turn data into information, and information into inside."
text.upper().replace(".", " ").replace(",", " ").split()



# Görev 2:  Verilen listeye aşağıdaki adımları uygulayınız.
# Adım1: Verilen listenin eleman sayısına bakınız.
# Adım2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
# Adım3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
# Adım4: Sekizinci indeksteki elemanı siliniz.
# Adım5: Yeni bir eleman ekleyiniz.
# Adım6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]
len(lst)

lst[0]
lst[10]

newlst = lst[0:4]

lst.pop(8)

lst.append("1")

lst.insert(8, "N")



# Görev 3:  Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
# Adım1: Key değerlerine erişiniz.
# Adım2: Value'lara erişiniz.
# Adım3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
# Adım4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
# Adım5: Antonio'yu dictionary'den siliniz.

dict = {"Christian": ["America", 18],
        "Daisy": ["England", 12],
        "Antonio": ["Spain", 22],
        "Dante": ["Italy", 25]}

dict.keys()
dict.values()
dict["Daisy"] = ["England", 13]
dict["Ahmet"] = ["Turkey", 24]

dict.pop("Antonio")



# Görev 4:Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atayan ve bu
# listeleri return eden fonksiyon yazınız.

l = [2, 13, 18, 93, 22]

def func(x):
    even_list = []
    odd_list = []
    for number in x:
        if number % 2 == 0:
            even_list.append(number)
        else:
            odd_list.append(number)

    return odd_list, even_list


odd_list, even_list = func(l)



# Görev 5:  List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini büyük harfe
# çeviriniz ve başına NUM ekleyiniz.

import seaborn as sns

df = sns.load_dataset("car_crashes")

["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]



# Görev 6:  ListComprehension yapısı kullanarak car_crashes verisinde isminde "no" barındırmayan değişkenlerin
# isimlerinin sonuna "FLAG" yazınız.

[col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]



# Görev 7:  ListComprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin
# isimlerini seçiniz ve yeni bir dataframe oluşturunuz.

og_list = ["abbrev", "no_previous"]

new_cols = [col for col in df.columns if col not in og_list]

new_df = df[new_cols]

new_df
