import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

df=pd.read_csv('salaries.csv')
X=df.drop('salary_more_then_100k',axis='columns')
y=df['salary_more_then_100k']

#transform categorical features into numerical one
company=LabelEncoder()
job=LabelEncoder()
degree=LabelEncoder()

X['company_n']=company.fit_transform(X['company'])
X['job_n']=job.fit_transform(X['job'])
X['degree_n']=degree.fit_transform(X['degree'])

X_n =X.drop(['company','job','degree'],axis='columns')

model=tree.DecisionTreeClassifier()
model.fit(X_n,y)
print(model.predict([[2,0,1]]))

