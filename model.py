import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from neupy import algorithms
import numpy as np
import statistics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


dataset = pd.read_csv("student-por.csv", sep=";")
pd.pandas.set_option('display.max_columns', None)
sc = {
    'GP': 1,
    'MS': 2,
}
parent = {
    'mother': 1,
    'father': 2,
    'other': 3,
}
reas = {
    'home': 1,
    'reputation': 2,
    'course': 3,
    'other': 4,
}
mjob = {
    'teacher': 1,
    'health': 2,
    'services': 3,
    'at_home': 4,
    'other': 5,

}
fjob = {
    'teacher': 1,
    'health': 2,
    'services': 3,
    'at_home': 4,
    'other': 5,

}
change = {
    'yes': 1,
    'no': 0,
}

dataset['address'].replace(to_replace="U", value=1, inplace=True)
dataset['address'].replace(to_replace="R", value=2, inplace=True)
dataset['famsize'].replace(to_replace="LE3", value=1, inplace=True)
dataset['famsize'].replace(to_replace="GT3", value=2, inplace=True)
dataset['Pstatus'].replace(to_replace="T", value=1, inplace=True)
dataset['Pstatus'].replace(to_replace="A", value=2, inplace=True)
dataset['romantic'] = dataset['romantic'].map(change)
dataset['internet'] = dataset['internet'].map(change)
dataset['famsup'] = dataset['famsup'].map(change)
dataset['schoolsup'] = dataset['schoolsup'].map(change)
dataset['sex'].replace(to_replace="M", value=1, inplace=True)
dataset['sex'].replace(to_replace="F", value=2, inplace=True)
dataset['Mjob'] = dataset['Mjob'].map(mjob)
dataset['Fjob'] = dataset['Fjob'].map(fjob)
dataset['activities'] = dataset['activities'].map(change)
dataset['paid'] = dataset['paid'].map(change)
dataset['nursery'] = dataset['nursery'].map(change)
dataset['higher'] = dataset['higher'].map(change)
dataset['reason'] = dataset['reason'].map(reas)
dataset['guardian'] = dataset['guardian'].map(parent)
dataset['school'] = dataset['school'].map(sc)
grade = []
for i in dataset['G3'].values:
    if i in range(0, 10):
        grade.append(4)
    elif i in range(10, 12):
        grade.append(3)
    elif i in range(12, 14):
        grade.append(2)
    elif i in range(14, 16):
        grade.append(1)
    else:
        grade.append(0)

d1 = dataset
print(d1)
se = pd.Series(grade)
d1['Grade'] = se.values
dataset.drop(dataset[dataset.G1 == 0].index, inplace=True)
dataset.drop(dataset[dataset.G3 == 0].index, inplace=True)
d1 = dataset
d1['All_Sup'] = d1['famsup'] & d1['schoolsup']

def max_parenteducation(d1):
    return max(d1['Medu'], d1['Fedu'])


d1['maxparent_edu'] = d1.apply(lambda row: max_parenteducation(row), axis=1)
# d1['PairEdu'] = d1[['Fedu', 'Medu']].mean(axis=1)
d1['more_high'] = d1['higher'] & (d1['schoolsup'] | d1['paid'])
d1['All_alc'] = d1['Walc'] + d1['Dalc']
d1['Dalc_per_week'] = d1['Dalc'] / d1['All_alc']
d1.drop(['Dalc'], axis=1, inplace=True)
d1.drop(['Walc'], axis=1, inplace=True)
d1['studytime_ratio'] = d1['studytime'] / (d1[['studytime', 'traveltime', 'freetime']].sum(axis=1))
d1.drop(['studytime'], axis=1, inplace=True)
d1.drop(['Fedu'], axis=1, inplace=True)
d1.drop(['Medu'], axis=1, inplace=True)
X = d1.iloc[:, [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32,
                33, 34]]
Y = d1.iloc[:, [28]]
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.30, random_state=42)
pnn = algorithms.PNN(std=5, verbose=False)
pnn.train(xTrain, yTrain)
y_training = pnn.predict(xTrain)
y_prediction = pnn.predict(xTest)
print('Prediction accuracy of train data : ')
print('{:.2%}\n'.format(metrics.accuracy_score(yTrain, y_training)))
print('Prediction accuracy of test data : ')
print('{:.2%}\n'.format(metrics.accuracy_score(yTest, y_prediction)))