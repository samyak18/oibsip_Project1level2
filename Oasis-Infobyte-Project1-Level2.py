#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
np.random.seed(125)
n = 200
square_footage = np.random.normal(loc=2000, scale=500, size=n)
bedrooms = np.random.choice(range(1, 8), size=n)
bathrooms = np.random.choice(range(1, 4), size=n)
price = 200000 + 50 * square_footage + 40000 * bedrooms + 60000 * bathrooms + np.random.normal(loc=0, scale=30000, size=n)

data = pd.DataFrame({'square_footage': square_footage,
                     'bedrooms': bedrooms,
                     'bathrooms': bathrooms,
                     'price': price})
print(data.head())
X = data[['square_footage', 'bedrooms', 'bathrooms']]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
predictions = model.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('Actual prices')
plt.ylabel('Predicted prices')
plt.title('Predicting house Prices with linear regression')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='Black')  
plt.show()


# In[ ]:




