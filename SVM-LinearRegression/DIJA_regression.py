import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

def simple_linear_regression(filename, predict_val):
    df = pd.read_csv(filename, engine='python')
    y = df.AvgVol
    x = df['MarketCap'].to_numpy()
    X = x.copy() # prevent to edit the original data
    X.shape = (X.shape[0], 1) # X.shape[0]: nrows ---> n by 1 matrix

    model = linear_model.LinearRegression()
    model.fit(X, y)

    print(f'The model is Volume = {model.coef_[0]:.3} * (Market Capitalization) + {model.intercept_:.3}')
    print(f'R^2 is {model.score(X,y):.3}')
    
    # 0.25 * (10**12) as example, and we later on find the predicted y-value of this point
    market_cap = np.array([[predict_val]]) # 1x1
    # print(market_cap.shape)
    predicted_cap = model.predict(market_cap)
    print(f'We predict a Volume of {predicted_cap[0]:.3} for a company with Market Capitalization {predict_val:.3} (see red dot).')

    # Plots
    y_hat = model.predict(X)
    plt.scatter(X, y, color='k')
    plt.title('DJIA: Volume vs. Capitalization')
    plt.ylabel('Average Daily Trading Volume')
    plt.xlabel('Market Capitalization')
    plt.plot(X, y_hat, color='k')
    plt.scatter(market_cap, predicted_cap, color='r')
    plt.show()

def multiple_linear_regression(filename):
    df = pd.read_csv(filename, engine='python')
    X = df[['MarketCap', 'Price']].to_numpy() # 30x2
    y = df['AvgVol'].to_numpy()

    model = linear_model.LinearRegression()
    model.fit(X, y)

    print(f'The model is Volume = {model.coef_[0]:.3} * (Market Capitalization) + {model.coef_[1]:.3} * Price + {model.intercept_:.3}.')
    print(f'R^2 is {model.score(X,y):.3}') 

if __name__ == '__main__':
    print('Running simple linear regression where we consider Volume as Y and Market Capitalization as X')
    simple_linear_regression('DJIA.csv', 0.85 * (10**12))
    print('')
    print('Running multiple linear regression where we consider Volume as Y and Market Capitalization as X1 and Price as X2')
    multiple_linear_regression('DJIA.csv')
