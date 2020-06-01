
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


data = pd.read_csv("titanic.csv").dropna()

data = pd.get_dummies(data,drop_first=True)

y = data.survived
x = data[["pclass","age","sibsp","parch","fare","adult_male"]]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

lr = LogisticRegression()
lr.fit(y=y_train,X=X_train)

y_pred = lr.predict(X_test)


print(metrics.classification_report(y_test, y_pred, labels=[0, 1]))