# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Importing Modules
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```
### Authenticate & Create DataFrame using Data in Sheets
```
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('dp').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})


```
### Assign X and Y values
```
X=df[['Input']].values
y=df[['Output']].values
```
### Normalize the value & split the data
```
Scaler=MinMaxScaler()
Scaler.fit(X)
X=Scaler.fit_transforms(x)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=33)

```
### Create a Neural Network & Train it
```
ai_brain=Sequential([
    Dense(9,activation='relu'),
    Dense(11,activation='relu'),
    
    Dense(1)
])
ai_brain.compile(
    optimizer = 'rmsprop',
    loss = 'mse'
)

ai_brain.fit(X_train1,y_train,epochs = 3000)
```
### Plot the Loss
```
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
```
### Evaluate the model
```
ai_brain.evaluate(X_test,y_test)
```
### Predict the Value
```
X_n1 = [[101]]

X_n1_1 = Scaler.transform(X_n1)

ai_brain.predict(X_n1_1)

```

## Dataset Information
![](1.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![](2.png)

### Test Data Root Mean Squared Error

![](3.png)

### New Sample Data Prediction

![](4.png)

## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully
