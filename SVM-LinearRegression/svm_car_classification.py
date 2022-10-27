# to classify cars as having automatic or manual transmissions.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, linear_model

# Dataset is given in the repository
def svm_car_classification(filename):
    df = pd.read_csv(filename, engine='python')
    df = df.rename(columns={'Unnamed: 0': 'brand'})

    X = np.array([df.mpg, df.wt]).T
    y = np.array(df.am)

    # fit & accuracy
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y)
    accuracy = clf.score(X, y)

    # predict
    X_new = np.array([[20],[4]])
    X_new = np.reshape(X_new, newshape=(1,-1)) #(-1,1) row->col, (1,-1) col->row
    predicted_label = clf.predict(X_new)
    # print(f'clf.coef_: {clf.coef_}\nclf.intercept_: {clf.intercept_}')
    print(f'The decision boundary is {clf.coef_[0][1]:.3} * weight + {clf.coef_[0][0]:.3} * mileage + 32.0 = 0.')
    print(f'The training accuracy is {accuracy}.')
    print(f'We predict that a car weighing 4 thousand pounds that gets 20 mpg has transmission type {predicted_label[0]} (where 0=automatic, 1=manual).')

    # Plots & Add legends for two groups points
    labels = ['automatic', 'manual']
    colors = ['red', 'blue']
    grouped = df.groupby(df.am)
    for idx, groupedDF in grouped:
        plt.scatter(groupedDF.wt, groupedDF.mpg, label=labels[idx], c=colors[idx])
    plt.xlim(0,6)
    plt.ylim(0,35,5)
    plt.title('SVM to guess transmission from car weight and mileage')
    plt.xlabel('weight (1000s of pounds)')
    plt.ylabel('gas mileage (miles per gallon)')

    x1 = X[:,1]
    x2 = (-clf.coef_[0][1]*x1-clf.intercept_[0])/clf.coef_[0][0]
    plt.plot(x1, x2, '-', label=r'decision boundary $\mathbf{wx} + b = 0$')
    plt.plot(x1, x2+(1/clf.coef_[0][0]), ':', color='orange', label=r'+1 support $\mathbf{wx} + b = 1$')
    plt.plot(x1, x2-(1/clf.coef_[0][0]), ':', color='green', label=r'-1 support $\mathbf{wx} + b = -1$')
    plt.legend(loc='lower left')
    plt.show()
    # plt.hold(True)
    
if __name__ == '__main__':
    svm_car_classification('mtcars30.csv')
