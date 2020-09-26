import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

data_set_path= "Iris-dataset.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data_set = pd.read_csv(data_set_path,names=names)

array = data_set.values
X = array[:,0:4] #all rows-data, columns 0,1,2,3
y = array[:,4] #all rows of data, column 4-last column-result column
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

model = KNeighborsClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
