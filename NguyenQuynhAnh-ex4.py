import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the data
data = pd.read_csv('framingham.csv')
data = data.dropna(how="any", axis=0)

# Prepare the data
y = data['TenYearCHD'].to_numpy()
X = data.drop('TenYearCHD', axis=1).to_numpy()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=37)

# Logistic Regression Model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Calculate Accuracy, Precision, and Recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy using Logistic Regression: {accuracy:.4f}")
print(f"Precision using Logistic Regression: {precision:.4f}")
print(f"Recall using Logistic Regression: {recall:.4f}")

# Perceptron
def h(w, x):
    return np.sign(np.dot(w.T, x))

def has_converged(X, y, w):
    return np.array_equal(h(w, X.T), y)

def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[0]
    mis_points = []  # set of miss position points

    while True:
        # mix data
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[mix_id[i]].reshape(-1, 1)
            yi = y[mix_id[i]]
            if h(w[-1], xi) != yi:
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi * xi
                w.append(w_new)
        if has_converged(X, y, w[-1]):
            break
    return w, mis_points

# Initialize weights
d = X_train.shape[1]
w_init = np.random.randn(d, 1)

# Train the Perceptron model
w, mis_points = perceptron(X_train, y_train, w_init)

# Predict on the testing set
y_pred_perceptron = h(w[-1], X_test.T)

# Calculate Accuracy, Precision, and Recall for Perceptron
accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)
precision_perceptron = precision_score(y_test, y_pred_perceptron)
recall_perceptron = recall_score(y_test, y_pred_perceptron)

print(f"\nAccuracy using Perceptron: {accuracy_perceptron:.4f}")
print(f"Precision using Perceptron: {precision_perceptron:.4f}")
print(f"Recall using Perceptron: {recall_perceptron:.4f}")

#Mô hình hồi quy logistic hoạt động tốt hơn đáng kể so với mô hình Perceptron trên tập dữ liệu này.
# Nó có độ chính xác, precision, and recall thu hồi cao hơn,