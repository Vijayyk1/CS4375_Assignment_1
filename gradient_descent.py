import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

#Getting the Dataset from my GitHub
dataset = 'https://raw.githubusercontent.com/Vijayyk1/CS4375_Assignment_1/main/car.data'
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
data = pd.read_csv(dataset, names=columns)

#Preprocessing Data
data.isnull().sum()
data = data.drop_duplicates()

#Converting categorical data to numerical data
label_encoders = {}
for column in data.columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

#Setting class variable as the target variable
X = data.drop('class', axis=1)
y = data['class']

#Standardizing data through standard scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Spliting dataset into training and test sets 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Parameters required for Gradient Descent
alpha = 1.001   #learning rate
m = y.size   # no. of samples
np.random.seed(10)
theta = np.random.rand(X.shape[1])  # theta is the weights vector, initializing theta with some random values

# Gradient Descent function
def gradient_descent(x, y, m, theta, alpha):
    cost_list = []   #to record all cost values to this list
    theta_list = []  #to record all theta_0 and theta_1 values to this list
    prediction_list = []
    run = True
    cost_list.append(1e10)    #we append some large value to the cost list
    i=0
    while run:
        prediction = np.dot(x, theta)   #predicted y values theta_0*x0+theta_1*x1
        prediction_list.append(prediction)
        error = prediction - y
        cost = 1/(2*m) * np.dot(error.T, error)   #  (1/2m)*sum[(error)^2]
        cost_list.append(cost)
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))   # alpha * (1/m) * sum[error*x]
        theta_list.append(theta)
        if cost_list[i]-cost_list[i+1] < 1e-4:   #checking if the change in cost function is less than 10^(-4)
            run = False

        i+=1
    cost_list.pop(0)   # Remove the large number we added in the begining
    return prediction_list, cost_list, theta_list

prediction_list, cost_list, theta_list = gradient_descent(X_train, y_train, m, theta, alpha)
theta = theta_list[-1]

#Evaluating Data
y_test_predict = np.dot(X_test, theta)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2 = r2_score(y_test, y_test_predict)
#Output for log and used to copy paste onto my report for analysis of various trials
print(f'Learning rate: {alpha}\n')
print(f'Number of iterations: {m}\n')
print(f'MSE on test set: {rmse}\n')  
print(f'R2 on test set: {r2}\n')
print(f'Weight Coefficients: {theta}')

sns.histplot(data = data, x = "safety")
plt.title('Evaluation of Cars')
plt.xlabel('Safety Rating')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()