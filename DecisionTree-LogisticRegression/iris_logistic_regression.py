import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

def iris_logistic_reg(path):
    """
    1.Logistic Regression on Iris dataset with features petal length, labels 1 means virginica,
      0 means not virginica species
    2.Hyperparameter path is should be set up by users to access iris data set,
      predicted is the petal length you want to predict (both its possible species and probability)
    """

    df = pd.read_csv(path, engine='python')
    df = df[df.Species!='setosa'] # discard setosa species in this file
    df = df[['Petal.Length', 'Species']]

    X = df[['Petal.Length']].to_numpy()
    y = (df['Species']=='virginica').to_numpy().astype(int) # either 0 or 1 (is virginica)

    model = linear_model.LogisticRegression(C=1000) # Suggested
    model.fit(X,y)
    accuracy = model.score(X,y)
    print(f'The accuracy on the training data is {accuracy}.')

    # report estimated probability of X[X[:,0]==5.0] & y = 1
    prob = model.predict_proba(X[X[:,0]==5])[:,1][0] # because the four probabilities are identical, pick the first one
    print(f'The estimated probability of virginica with petal length=5 is {prob:.2}.')
    print(f'The estimated probability of non-virginica with petal length=5 is {1-prob:.2}.')

    pred_species = np.array([[5]]) # petal_length = 5
    pred = model.predict(pred_species) # array of probabilities

    def label(value):
        if value==1:
            return 'virginica'
        if value==0:
            return 'versicolor'

    print(f'The predicted species with petal length 5 is {label(pred[0])}')
    # print(X[0:3,0:3]) test the dimension N*D

    # Plotting
    w = model.coef_[0][0]
    b = model.intercept_[0]
    xplot = np.linspace(2,8)
    yplot = 1/(1 + np.exp(-(w * xplot + b)))

    plt.scatter(x=X[:,0],y=y, label=r'data $(x_i, y_i)$')
    plt.plot(xplot, yplot, '-g', label=r'logistic curve $\hat{P}(y=1|x)$')

    # Calculate Sample Proportion
    x_values, x_counts = np.unique(X, return_counts=True) # array
    n_x_values = x_values.shape[0]
    success_proportion_per_x_value = np.zeros(n_x_values)
    for i in np.arange(n_x_values):
        success_proportion_per_x_value[i] = np.sum(y[X[:, 0] == x_values[i]]) / x_counts[i]

    # a legend and title and other labels necessary to make the plot easy to read
    plt.plot(x_values, success_proportion_per_x_value, '.r', label='sample proportions')
    plt.legend(loc='upper left')
    plt.title('Logistic Regression on Iris Dataset (Feature:Petal Length. Species: Virginica and Versicolor.)')
    plt.show()

if __name__ == '__main__':
    iris_logistic_reg('iris.csv')