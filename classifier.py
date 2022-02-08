from array import array
from statistics import mode
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
import pickle

URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
NAMES = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data_set = read_csv(URL, names=NAMES)

#set data ti array for processing
array = data_set.values
x = array[:, 0:4]
y = array[:, 4]

#saving the final model
model = SVC(gamma='auto')
model.fit(x, y)

with open('finalized_model.sav', 'wb') as file:
    pickle.dump(model, file)


#how to use
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=1)
with open('finalized_model.sav', 'rb') as file:
    loaded_model = pickle.load(file)
    result = loaded_model.score(x_validation, y_validation)
#show the perfoance of the model
print(result)
#make a prediction
xnew = [[3, 5, 6.8, 4]]
print(loaded_model.predict(xnew))