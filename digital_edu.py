
import matplotlib.pyplot as plt
import pandas as pd 


df = pd.read_csv("ra.csv")
df["education_form"].fillna("Full-time", inplace= True)
df[list(pd.get_dummies(df["education_form"]).columns)] = pd.get_dummies(df["education_form"])
df.drop(["education_form"], axis = 1, inplace = True)

established = 0
not_installed = 0
established_followers = 0
not_installed_followers = 0
def photo_count(row):
    global established, not_installed, established_followers, not_installed_followers
    if row['has_photo'] == 1:
        established += 1
        if row['followers_count'] == 1:
            established_followers += 1
    elif row['has_photo'] == 0:
        not_installed += 1
        if row['followers_count'] == 1:
            not_installed_followers += 1
df.apply(photo_count, axis = 1)
established_procent = established_followers/(established+not_installed)
not_installed_procent = not_installed_followers/(established+not_installed)

photos = pd.Series(index = ['Есть фото', 'нету фото'], data = [established_procent, not_installed_procent])
photos.plot(kind = 'pie')
plt.show()

df.drop(["bdate","has_mobile","has_photo","city","graduation","followers_count","last_seen","occupation_type","occupation_name","life_main","people_main","career_start","career_end","education_status","langs"], axis = 1, inplace = True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

x = df.drop("result", axis = 1)
y = df["result"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(y_test)
print(y_pred)
print("Правильные исходы:",  round(accuracy_score(y_test, y_pred) * 100, 2))