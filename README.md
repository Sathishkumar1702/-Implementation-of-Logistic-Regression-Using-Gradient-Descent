# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. start the program.
2. import libraries.
3. Read the dataset of social network ads.
4. from sklearn import train_test_split.
5. preprocessing import standardscaler.
6. sklearn import logisticregression.
7. use a classifier for fit(x_train,y_train).
8. from sklearn import confusion matrix,accuracy.
9. recall sensitivity metrics score(y_test,y_pred)
10. import listed colormap and plot it.
11. plot scatter and plot title, Age, estimated salary.
12. stop the program.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sathish kumar M
RegisterNumber:  212220220042
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('/content/sample_data/Social_Network_Ads (1).csv')
X=dataset.iloc[:, [2,3]].values
Y=dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_X

X_Train=sc_X.fit_transform(X_Train)
X_Test=sc_X.transform(X_Test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state = 0)
classifier.fit(X_Train, Y_Train)

Y_Pred=classifier.predict(X_Test)
Y_Pred

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_Test, Y_Pred)
cm

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_Test, Y_Pred)
accuracy

recall_sensitivity=metrics.recall_score(Y_Test,Y_Pred,pos_label=1)
recall_specificity=metrics.recall_score(Y_Test,Y_Pred,pos_label=0)
recall_sensitivity, recall_specificity

from matplotlib.colors import ListedColormap
X_set,Y_set=X_Train,Y_Train
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('black','yellow')))

plt.xlim(X1.min(),X2.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(Y_set)):
  plt.scatter(X_set[Y_set==j,0],X_set[Y_set==j,1],c=ListedColormap(('black','yellow'))(i),label=j)
  plt.title('Logistic Regression(Training set)')
  plt.xlabel('Age')
  plt.ylabel('Estimated Salary')
  plt.legend()
  plt.show()
```

## Output:
![logistic regression using gradient descent](/OP%201.png)
![logistic regression using gradient descent](/OP%202.png)
![logistic regression using gradient descent](/OP%203.png)
![logistic regression using gradient descent](/OP%204.png)
![logistic regression using gradient descent](/OP%205.png)
![logistic regression using gradient descent](/OP%206.png)
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

