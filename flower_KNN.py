import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


iris = load_iris()
#print(iris.feature_names)
#print(iris.target_names)

df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['target']=iris.target
df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')

#model 
X=df.drop(['target'],axis='columns')
y=df['target']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

knn_model=KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train,y_train)

#print(knn_model.score(X_test,y_test)) =1.0

y_predicted = knn_model.predict(X_test)
cm=confusion_matrix(y_test,y_predicted)
print(cm)





