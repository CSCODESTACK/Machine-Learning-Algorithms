
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#importing data
customers = pd.read_csv("Ecommerce Customers")

customers.head()
customers.describe()
customers.info()

#Exploratory Data Analysis
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')

# More time on site, more money spent.
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)
sns.pairplot(customers)
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)

#Training and Testing Data
y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Training the Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

#Predicting Test Data
predictions = lm.predict( X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

#Evaluating the Model
# calculate these metrics by hand!
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

sns.distplot((y_test-predictions),bins=50);