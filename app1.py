import streamlit as st
from PIL import Image
img = Image.open('images/icon.jpg')
img2 = Image.open('images/Linear-regression.png')
img3 = Image.open('images/SVM.png')
img4 = Image.open('images/KNN.png')
img5 = Image.open('images/flowchart.png')

def app():
    st.markdown('<h2 style="background-color:Grey; border-radius:5px; padding:5px 15px ; text-align:center ; font-family:arial;color:white">Student-Performance-Lab</h2>', unsafe_allow_html=True)
    st.markdown('<h4 style="border: inset 1px white; border-radius:4px; padding:2px 15px">Student Performance-Lab : <i>Analyzer</i></h4>', unsafe_allow_html=True)
    col1, col2 = st.beta_columns((1, 2))
    with col1:
        st.text("___" * 100)
        st.image(img, width=152)
    with col2:
        st.text("___" * 100)
        st.info("**_Student Performance-Lab_** : _Analyzer_ to analyze Student performance using _Machine Learning_ and _Artificial Intelligence_.")
        st.info("Let's dive into the world of Data !!")
    st.text("___" * 100)
    col3 = st.beta_columns(1)
    with st.beta_expander("Logistic Regression"):
        st.write('''<p>Logistic regression is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables. Logistic regression predicts the output of a categorical 
        dependent variable. Therefore the outcome must be a categorical or discrete value. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, <b>it gives the probabilistic values which lie between 0 and 1.</b></p>
        <p>Logistic Regression is much similar to the Linear Regression except that how they are used. Linear Regression is used for solving Regression problems, whereas <b>Logistic regression is used for solving the classification problems.</b> 
        In Logistic regression, instead of fitting a regression line, we fit an "S" shaped logistic function, which predicts two maximum values (0 or 1).
        The curve from the logistic function indicates the likelihood of something such as whether the cells are cancerous or not, a mouse is obese or not based on its weight, etc.
        </p>
        <p>Logistic Regression is a significant machine learning algorithm because it has the ability to provide probabilities and classify new data using continuous and discrete datasets.
        Logistic Regression can be used to classify the observations using different types of data and can easily determine the most effective variables used for the classification.
       </p>''',
                 unsafe_allow_html=True)
        st.image(img2, use_column_width=True)

    with st.beta_expander("Support Vector Machine (SVM)"):
        st.write('''<p>Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. 
        However, primarily, it is used for Classification problems in Machine Learning.</p>
        <p>The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. 
        This best decision boundary is called a hyperplane.
        </p>
        <p>SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors, and hence algorithm is termed as Support Vector Machine. 
        Consider the below diagram in which there are two different categories that are classified using a decision boundary or hyperplane:
        </p>''', unsafe_allow_html=True)
        st.image(img3,use_column_width=True)
        st.write('''<pre>
        <h1>Types of SVM</h1>
        <p><b>Linear SVM</b>: Linear SVM is used for linearly separable data, which means if a dataset can be classified
        into two classes by using a single straight line, then such data is termed as linearly separable 
        data, and classifier is used called as Linear SVM classifier.</p>
        <p><b>Non-Linear SVM</b>: Non-Linear SVM is used for non-linearly separated data, which means if a dataset cannot
        be classified by using a straight line, then such data is termed as non-linear data and classifier
        used is called as Non-linear SVM classifier.</p>
        </pre>''', unsafe_allow_html=True)

    with st.beta_expander("Navie Bayes"):
        st.write('''<p>Naïve Bayes algorithm is a supervised learning algorithm, which is based on <b>Bayes theorem</b> and used for solving classification problems.</p>
        <p>It is mainly used in text classification that includes a high-dimensional training dataset.</p>
        <p>Naïve Bayes Classifier is one of the simple and most effective Classification algorithms which helps in building the fast machine learning models that can make quick predictions.</p>
        <p><b>It is a probabilistic classifier, which means it predicts on the basis of the probability of an object</b>.</p>
        <p>Some popular examples of Naïve Bayes Algorithm are <b>spam filtration, Sentimental analysis, and classifying articles.</b>
        </p>''', unsafe_allow_html=True)
        st.write('''<b><h1>Bayes' Theorem:</h1></b>''', unsafe_allow_html=True)
        st.write('''<p>Bayes' theorem is also known as <b>Bayes' Rule</b> or <b>Bayes' law</b>, which is used to determine the probability of a hypothesis with prior knowledge. It depends on the conditional probability.</p>
        <p>The formula for Bayes' theorem is given as:</p>
        <h1><center><sub>P(A|B)=(P(B|A)*P(A))/P(B)</sub></center></h1>
        <p><b>Where,</b></p>
        <p><b>P(A|B) is Posterior probability</b>: Probability of hypothesis A on the observed event B.</p>
        <p><b>P(B|A) is Likelihood probability</b>: Probability of the evidence given that the probability of a hypothesis is true.</p>
        <p><b>P(A) is Prior Probability</b>: Probability of hypothesis before observing the evidence.</p>
        <p><b>P(B) is Marginal Probability</b>: Probability of Evidence.</p>
        ''', unsafe_allow_html=True)

    with st.beta_expander("KNeighborsClassifier"):
        st.write('''<p>K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on Supervised Learning technique.</p>
        <p>K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.</p>
        <p>K-NN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily classified into a well suite category by using K- NN algorithm.</p>
        <p>K-NN algorithm can be used for Regression as well as for Classification but mostly it is used for the Classification problems.</p>
        <p>K-NN is a <b>non-parametric algorithm</b>, which means it does not make any assumption on underlying data.</p>
        <p>It is also called a <b>lazy learner algorithm</b> because it does not learn from the training set immediately instead it stores the dataset and at the time of classification, it performs an action on the dataset.</p>
        <p>KNN algorithm at the training phase just stores the dataset and when it gets new data, then it classifies that data into a category that is much similar to the new data.</p>
        ''',unsafe_allow_html=True)
        st.image(img4, use_column_width=True)

    with st.beta_expander("Methodology"):
        st.write('''<p>The methodology is the core component of any research-related work. The methods used to gain the results are shown in the methodology. 
        Here, the whole research implementation is done using python. There are different steps involved to get the entire research work done which is as follows:</p>''',unsafe_allow_html=True)
        st.image(img5, use_column_width=True)
        st.write('''<b><h4>1. Acquire Student Dataset</h4></b>''', unsafe_allow_html=True)
        st.write('''<p>The UCI machine learning repository is a collection of databases, data generators which are used by machine learning community for analysis purpose. 
        The student performance dataset is acquired from the UCI repository website. The student performance dataset can be downloaded in zip file format just by clicking on the link available. 
        The student zip file consists of two subject CSV files (student-por.csv and student-mat.csv). The Portuguese file has no missing values, 33 attributes, and classification, regression-related tasks. 
        Also, the dataset has multivariate characteristics. Here, data-preprocessing is done for checking inconsistent behaviors or trends.</p>''', unsafe_allow_html=True)
        st.write('''<b><h4>2. Data preprocessing</h4></b>''', unsafe_allow_html=True)
        st.write('''<p>After, Data acquisition the next step is to clean and preprocess the data. 
        The Dataset available has object type features that need to be converted into numerical type. 
        Thus, using python dictionary and mapping functions the transformation is being done. 
        Also, a new column Grade and some new features have been created using two or more columns. 
        The target value is a five-level classification consisting of 0 i.e. excellent or 'A' to 4 i.e. fail or 'F'. 
        The preprocessed dataset is further split into training and testing datasets. 
        This is achieved by passing feature value, target value, test size to the train-test split method of the scikit-learn package. 
        After splitting of data, the training data is sent to the following neural network design i.e. Logistic regression, SVM, Navie Bayes and KNeighborClassifier for training the artificial neural networks then test data is used to predict the accuracy of the trained network model.</p>''', unsafe_allow_html=True)
        st.write('''<b><h4>3. Design of Logistic regression, Support Vector Machine(SVM), Navie Bayes and KNeighborsClassifier</h4></b>''', unsafe_allow_html=True)
        st.write('''<p>The design of logistic regression and support vector machine in python environment is achieved through neupy package which requires the standard deviation value as the most important parameter. 
        Along with it, the network comprises 30 inputs neuron, pattern layer, summation layer, and decision layer for five-level classification whereas the design of Navie Bayes and KNeighborClassifier neural network in python environment is achieved through neupy package which requires the number of input features, the number of classes i.e. the classification result output neuron,  learning rate. 
        The network comprises 30 input features i.e. input neurons, hidden layer, and the output layer for five-level classification. 
        Once the design for logistic regression, support vector machine, Navie Bayes and KNeighborClassifier is ready it is trained with the training data for accurate classification and then testing data is used for the trained neural network.</p>''', unsafe_allow_html=True)
        st.write('''<b><h4>4. Testing and Classified Output</h4></b>''', unsafe_allow_html=True)
        st.write('''<p>After the training of the designed neural network, the testing of logistic regression, support vector machine, Navie Bayes and KNeighborClassifier is performed using testing data. 
        Based on testing data,  the accuracy of the classifier is determined.</p>''', unsafe_allow_html=True)