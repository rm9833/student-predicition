import streamlit as st
import plotly_express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from neupy import algorithms
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
import itertools
import time
import copy
global df


def app():
    header = st.beta_container()
    dataset = st.beta_container()
    datapreprocessing = st.beta_container()
    features = st.beta_container()
    graphs = st.beta_container()
    model_training = st.beta_container()

    @st.cache
    def feat(data2):
        x = data2.describe()
        return x

    @st.cache
    def pre_data(filename1):

        data2 = filename1.copy()
        return data2

    @st.cache
    def grapy(data4):
        df = data4
        numeric_cols = list(df.select_dtypes(['float64', 'int64']).columns)
        text_data = df.select_dtypes(['object'])
        text_cols = text_data.columns
        return df, numeric_cols, text_cols

    with header:
        st.markdown(
            '<h2 style="background-color:Grey; border-radius:5px; padding:5px 15px ; text-align:center ; font-family:arial;color:white"><i>Analysis</i></h2>',
            unsafe_allow_html=True)
        st.subheader("- Let's understand and analyize it.")

    with dataset:
        with st.beta_expander("Dataset-"):
            st.header("Online Dataset")
            dataset = pd.read_csv("student-por.csv", sep=";")

            if st.button('View Data'):
               latest_iteration = st.empty()
               for i in range(100):
                   latest_iteration.info(f' {i + 1} %')
                   time.sleep(0.05)
               time.sleep(0.2)
               latest_iteration.empty()
               st.info("student-por.csv")
               st.write(dataset.head(649))
               x_val = dataset.shape[0]
               y_val = dataset.shape[1]
               st.write("Data-shape :", x_val, "Features :", y_val)
    with datapreprocessing:
        with st.beta_expander("Pre-Processed Data-"):
            st.header("Data after Pre-processing:")
            st.write(dataset.head(649))
            dd = dataset.copy()
            data = copy.deepcopy(pre_data(dd))
            st.write('Number of participants: ', len(data))
            st.write('Is there any missing value? ', data.isnull().values.any())
            st.write('How many missing values? ', data.isnull().values.sum())
            data.dropna(inplace=True)
            st.write('Number of participants after eliminating missing values: ', len(data))
            ppd = st.checkbox(label="Preprocess-data")
            if ppd:
                dataset = data
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

                Data1 = dataset
                se = pd.Series(grade)
                Data1['Grade'] = se.values
                dataset.drop(dataset[dataset.G1 == 0].index, inplace=True)
                dataset.drop(dataset[dataset.G3 == 0].index, inplace=True)
                d1 = dataset
                d1['All_Sup'] = d1['famsup'] & d1['schoolsup']

                def max_parenteducation(d1):
                    return (max(d1['Medu'], d1['Fedu']))

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
                X = d1.iloc[:,
                    [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29,
                     30, 31, 32, 33, 34]]
                Y = d1.iloc[:, [28]]
                time.sleep(0.01)
                dp = st.success("Data-Preprocessed")
                time.sleep(1)
                dp.empty()
                if st.button("Final Data"):
                    d1 = dataset.copy()
                    d1.drop(d1.columns[28:], axis=1, inplace=True)
                    st.write(d1.head(150))
                    x_valnew = d1.shape[0]
                    y_valnew = d1.shape[1]
                    st.write("Data-shape :", x_valnew, "Features :", y_valnew)
    with features:
        with st.beta_expander("Features-"):
            st.header("Features Description:")
            y3 = copy.deepcopy(dataset)
            y3.drop(y3.columns[28:], axis=1, inplace=True)
            y = copy.deepcopy(feat(y3))
            st.write(y)

    with graphs:
        with st.beta_expander("Graphical Visualization-"):
            st.header("Graphical representation:")
            df, numeric_cols, text_cols = grapy(y3)

            col3, col4 = st.beta_columns((1, 3))

            with col3:
                chart_select = st.selectbox(label="Select the chart-type", options=[
                    'Scatter-plots', 'Histogram', 'Distplot', 'Box-plot', 'Violin-plot', 'Heat-map',
                    ])
                if chart_select == 'Scatter-plots':
                    st.subheader("Scatter-plot Settings:")
                    x_values = st.selectbox('X-axis', options=numeric_cols)
                    y_values = st.selectbox('Y-axis', options=numeric_cols)
                    with col4:
                        plot = px.scatter(data_frame=df, x=x_values, y=y_values)
                        st.plotly_chart(plot)
                if chart_select == 'Histogram':
                    st.subheader("Histogram Settings:")
                    x_values = st.selectbox('value', options=numeric_cols)
                    x_val = np.array(df[x_values])
                    fig, ax = plt.subplots(figsize=(15, 9))
                    sns.set_style("dark")
                    sns.set_style("darkgrid")
                    sns.histplot(data=x_val, kde=True)
                    with col4:
                        st.pyplot(fig)
                if chart_select == 'Distplot':
                    st.subheader("Distplot Settings:")
                    x_values = st.selectbox('value', options=numeric_cols)
                    x_val = np.array(df[x_values])
                    fig, ax = plt.subplots(figsize=(15, 9))
                    sns.set_style("dark")
                    sns.set_style("darkgrid")
                    sns.distplot(x_val)
                    with col4:
                        st.pyplot(fig)
                if chart_select == 'Box-plot':
                    st.subheader("Box-plot Settings:")
                    x_values = st.selectbox('X-axis', options=numeric_cols)
                    y_values = st.selectbox('Y-axis', options=numeric_cols)
                    with col4:
                        plot = px.box(data_frame=df, x=x_values, y=y_values)
                        st.plotly_chart(plot)
                if chart_select == 'Violin-plot':
                    st.subheader("Violin-plot Settings:")
                    x_values = st.selectbox('X-axis', options=numeric_cols)
                    y_values = st.selectbox('Y-axis', options=numeric_cols)
                    with col4:
                        plot = px.violin(data_frame=df, x=x_values, y=y_values, points='all', box=True)
                        st.plotly_chart(plot)
                if chart_select == 'Heat-map':
                    st.subheader('Heat-map')

                    data_val = y3
                    fig, ax = plt.subplots(figsize=(25, 10))
                    sns.set_style("darkgrid")
                    sns.set_style("dark")
                    sns.set_theme(style='darkgrid', palette='deep')
                    sns.heatmap(data_val.corr(), ax=ax, annot=True, annot_kws={"size": 9}, fmt='.1f', linewidths=.5,
                                cbar=True, xticklabels=1, yticklabels=1,
                                cbar_kws={"orientation": "vertical"}, cmap='BuPu')
                    with col4:
                        st.pyplot(fig)

    with model_training:
        #col4, col5 = st._columns((3, 4))
        with st.beta_expander("Model Training-"):
            st.header("Accuracy of Model:")
            classifier_name = st.selectbox("Select Classifier :", ("Logistic Regression", "Support Vector Machine","Naive Bayes","KNeighborsClassifier"))
            if classifier_name == "Logistic Regression":
                if st.button("Score of Logistic Regression"):
                    time.sleep(0.1)
                    xy = st.balloons()
                    from sklearn import linear_model
                    ln= linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
                    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.30, random_state=42)
                    ln.fit(xTrain, yTrain)
                    y_training = ln.predict(xTrain)
                    y_prediction = ln.predict(xTest)
                    st.write('Prediction accuracy of train data : ')
                    st.write('{:.2%}\n'.format(metrics.accuracy_score(yTrain, y_training)))
                    st.write('Prediction accuracy of test data : ')
                    st.write('{:.2%}\n'.format(metrics.accuracy_score(yTest, y_prediction)))
                    from sklearn.metrics import cohen_kappa_score
                    cohen_score = cohen_kappa_score(yTest, y_prediction)
                    st.write('Cohen-Score of test data:')
                    st.write(cohen_score)
                    time.sleep(1)
                    xy.empty()
            if classifier_name == "Support Vector Machine":
                if st.button("Score of Support Vector Machine"):
                    time.sleep(0.1)
                    xy = st.balloons()
                    from sklearn.svm import SVC
                    svc = SVC(kernel='poly', random_state=0)
                    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.30, random_state=42)
                    svc.fit(xTrain, yTrain)
                    y_training = svc.predict(xTrain)
                    y_prediction = svc.predict(xTest)
                    st.write('Prediction accuracy of train data : ')
                    st.write('{:.2%}\n'.format(metrics.accuracy_score(yTrain, y_training)))
                    st.write('Prediction accuracy of test data : ')
                    st.write('{:.2%}\n'.format(metrics.accuracy_score(yTest, y_prediction)))
                    from sklearn.metrics import cohen_kappa_score
                    cohen_score = cohen_kappa_score(yTest, y_prediction)
                    st.write('Cohen-Score of test data:')
                    st.write(cohen_score)
                    time.sleep(1)
                    xy.empty()

            if classifier_name == "Naive Bayes":
                if st.button("Score of Naive Bayes"):
                    time.sleep(0.1)
                    xy = st.balloons()
                    from sklearn.naive_bayes import GaussianNB
                    gnb = GaussianNB()
                    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.30, random_state=42)
                    gnb.fit(xTrain, yTrain)
                    y_training = gnb.predict(xTrain)
                    y_prediction = gnb.predict(xTest)
                    st.write('Prediction accuracy of train data : ')
                    st.write('{:.2%}\n'.format(metrics.accuracy_score(yTrain, y_training)))
                    st.write('Prediction accuracy of test data : ')
                    st.write('{:.2%}\n'.format(metrics.accuracy_score(yTest, y_prediction)))
                    from sklearn.metrics import cohen_kappa_score
                    cohen_score = cohen_kappa_score(yTest, y_prediction)
                    st.write('Cohen-Score of test data:')
                    st.write(cohen_score)
                    time.sleep(1)
                    xy.empty()

            if classifier_name == "KNeighborsClassifier":
                if st.button("Score of KNeighborsClassifier"):
                    time.sleep(0.1)
                    xy = st.balloons()
                    from sklearn.neighbors import KNeighborsClassifier
                    knn = KNeighborsClassifier(n_neighbors=8)
                    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.30, random_state=42)
                    knn.fit(xTrain, yTrain)
                    y_training = knn.predict(xTrain)
                    y_prediction = knn.predict(xTest)
                    st.write('Prediction accuracy of train data : ')
                    st.write('{:.2%}\n'.format(metrics.accuracy_score(yTrain, y_training)))
                    st.write('Prediction accuracy of test data : ')
                    st.write('{:.2%}\n'.format(metrics.accuracy_score(yTest, y_prediction)))
                    from sklearn.metrics import cohen_kappa_score
                    cohen_score = cohen_kappa_score(yTest, y_prediction)
                    st.write('Cohen-Score of test data:')
                    st.write(cohen_score)
                    time.sleep(1)
                    xy.empty()