# Developing-a-Neural-Network-Regression-Model
Developed a Neural Network Regression Model using PyTorch to predict student exam marks based on study hours. Implemented data preprocessing, model training, evaluation, and visualization using deep learning techniques.

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The objective of this project is to develop a Neural Network Regression Model using PyTorch to predict student exam marks based on the number of study hours. The model uses **Study_hours** as the input feature and **Exam_marks** as the target variable. By learning the relationship between study hours and exam performance from the given dataset, the model aims to generate predictions for students based on their study time.

The project involves data preprocessing, feature scaling, splitting the dataset into training and testing sets, and designing a neural network with multiple hidden layers. The model is trained using the Adam optimizer and Mean Squared Error (MSE) loss function. Its performance is evaluated using MSE, RMSE, MAE, and R² score. Finally, the trained model is used to predict the exam marks of a student studying for 9 hours, demonstrating the application of deep learning in regression-based prediction.

## Neural Network Model:
<img width="1024" height="759" alt="NN" src="https://github.com/user-attachments/assets/dc62de6a-15fb-49e4-a108-cbc7d4a4bb6b" />


## DESIGN STEPS:

STEP 1:
Create your dataset in a Google sheet with one numeric input and one numeric output.

STEP 2:
Split the dataset into training and testing

STEP 3:
Create MinMaxScalar objects ,fit the model and transform the data.

STEP 4:
Build the Neural Network Model and compile the model.

STEP 5:
Train the model with the training data.

STEP 6:
Plot the performance plot

STEP 7:
Evaluate the model with the testing data.

STEP 8:
Use the trained model to predict for a new input value .

## PROGRAM:
```
Name: MANIKANDAN K
Register Number: 212224230150
```

```python
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("dataset-1.csv")
print("MANIKANDAN K")
print("212224230150")
print(df)
```
## OUTPUT:
<img width="298" height="303" alt="image" src="https://github.com/user-attachments/assets/c8dd9667-a753-490d-bcce-65084927264d" />



```python
X = df[["Study_ours"]].values
y = df[["Exam_marks"]].values

print("Input:")
print(X)

print("Output:")
print(y)
```


## OUTPUT:
<img width="567" height="510" alt="image" src="https://github.com/user-attachments/assets/c631c9b1-1d19-416a-a0da-cc7ca92fffef" />


```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=33
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
```

## OUTPUT:
<img width="215" height="51" alt="image" src="https://github.com/user-attachments/assets/f9d0e749-eda8-402c-9fa1-e12b545e87bb" />


```python
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)
print(X_test)
```


## OUTPUT:
<img width="220" height="270" alt="image" src="https://github.com/user-attachments/assets/83bd88cb-2847-4119-9a73-9755e1b6472d" />


```python
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print(X_train.shape)
print(y_train.shape)
```

## OUTPUT:
<img width="231" height="66" alt="image" src="https://github.com/user-attachments/assets/a658007c-7c27-4daf-98c2-28c3041617ce" />




```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)
torch.manual_seed(33)

model = NeuralNet()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(model)
```



## OUTPUT:
<img width="444" height="173" alt="image" src="https://github.com/user-attachments/assets/31a34ed0-3bdb-4fd5-ad27-eec30b049ebd" />




```python
epochs = 1000
loss_list = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
model.eval()

with torch.no_grad():
    predictions = model(X_test).numpy()

print("\nModel Evaluation")
print("MSE:", mean_squared_error(y_test.numpy(), predictions))
print("RMSE:", mean_squared_error(y_test.numpy(), predictions) ** 0.5)
print("MAE:", mean_absolute_error(y_test.numpy(), predictions))
print("R2 Score:", r2_score(y_test.numpy(), predictions))
plt.figure(figsize=(8, 4))
plt.plot(loss_list)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Student Marks Prediction - Training Loss")
plt.show()
```



## OUTPUT:
<img width="668" height="628" alt="image" src="https://github.com/user-attachments/assets/d20225cd-2227-438e-b36a-f9c613719dc1" />



## NEW SAMPLE DATA PREDICTION:
```python
new_student = [[9]]

new_student_scaled = scaler.transform(new_student)

new_student_tensor = torch.tensor(
    new_student_scaled,
    dtype=torch.float32
)

with torch.no_grad():
    predicted_marks = model(new_student_tensor).item()

print(f"Predicted Exam Marks for 9 Study Hours: {predicted_marks:.2f}")
```


## OUTPUT:
<img width="610" height="297" alt="image" src="https://github.com/user-attachments/assets/4d6eb953-0618-4a85-b700-bd70adc31fb0" />




## RESULT:
Thus, a neural network regression model was successfully developed and trained using PyTorch.

