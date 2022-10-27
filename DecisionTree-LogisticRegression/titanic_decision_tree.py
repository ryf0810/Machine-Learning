import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree # for tree.plot_tree()
from sklearn.tree import export_text # for export_text()


def decision_tree(path):
    """
    1.This decision tree classifies if the passenger was survived based on three features (Pclass, Sex, Age)
    2.Retain only the Survived, Pclass, Sex, and Age columns for this decision tree
    """
    df = pd.read_csv(path, engine='python')
    df = df[['Survived', 'Pclass', 'Sex', 'Age']]

    # Display the first seven rows (passengers).
    print('First Seven Rows:')
    print(df.iloc[0:7,:])

    # Display your data frame's shape before and after dropping rows.
    print(f'\nDataFrame Shape before Dropping NaN Values: {df.shape}')
    df = df.dropna() # drop rows at any column has an NaN value
    print(f'DataFrame Shape before Dropping NaN Values: {df.shape}')

    # Add a column called 'Female', containts 1/0 representing T/F
    df['Female'] = (df.Sex=='female').astype('int')

    # Train a decision tree with max_depth=None to decided whether a passenger Survived from the other three columns.
    # Report its accuracy (with 3 decimal places) on training data along with the tree's depth (which is available in clf.tree_.max_depth).
    feature_names = ['Pclass', 'Age', 'Female']
    class_names = ['female', 'male']

    X = df[feature_names].to_numpy()
    y = df['Survived'].to_numpy()

    clf = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=0)
    clf.fit(X,y)
    # Report its accuracy (with 3 decimal places) on training data along with the tree's depth (which is available in clf.tree_.max_depth)
    print(f'\nDecision tree with depth {clf.tree_.max_depth} on training data has accuracy: {clf.score(X,y):.3}')


    # Train another tree with max_depth=2, report its accuracy(3 decimal places), display the tree with corresponding names
    clf_2 = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=0)
    clf_2.fit(X,y)

    print(f'\nDecision tree with depth {clf_2.tree_.max_depth} on training data has accuracy: {clf_2.score(X,y):.3}')
    tree.plot_tree(clf_2, feature_names=feature_names, class_names=class_names)
    plt.show()

if __name__ == '__main__':
    decision_tree('kaggle_titanic_train.csv')