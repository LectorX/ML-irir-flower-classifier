from array import array
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
NAMES = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data_set = read_csv(URL, names=NAMES)

#set data ti array for processing
array = data_set.values
x = array[:, 0:4]
y = array[:, 4]

#creating testing and validation arrays
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=1)

#creating models
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

#evaluating models
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, cv=kfold, y=y_train, X=x_train, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean()} ({cv_results.std()})')

#compare with graphics
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm comparison')
pyplot.show()

#final
model = SVC(gamma='auto')
model.fit(x_train, y_train)
predictions = model.predict(x_validation)

print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))
