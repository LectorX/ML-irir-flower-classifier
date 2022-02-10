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

#show the data structure (table columns, 150 rows)
print(data_set.shape)
input('PRESS ANY KEY TO CONTINUE')

#show the first 20 rows of data set
print(data_set.head(20))
input('PRESS ANY KEY TO CONTINUE')

#show stadictics of the data
print(data_set.describe())
input('PRESS ANY KEY TO CONTINUE')

#show the count of data for each 'class'
#being 'class' one of the data columns
print(data_set.groupby('class').size())