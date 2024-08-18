### Module 2: Classification and Training Models

---

## 2.1 Classification: MNIST, Training Binary Classifier

**Objective:** Understand the basics of classification using the MNIST dataset, including building and evaluating a binary classifier.

### Topics:

1. **Binary Classifiers:**
   - **Definition:** A binary classifier is a type of model that categorizes data into one of two classes.
   - **Example:** Classifying MNIST digits as either "0" or "not 0."
   - **Implementation:** Using logistic regression, support vector machines, or neural networks to build a binary classifier.

   **Code Example:**
   ```python
   from sklearn.datasets import fetch_openml
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score

   # Load MNIST data
   X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
   y = (y == '0')  # Binary classification: 0 vs. not 0

   # Split data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Train binary classifier
   model = LogisticRegression(max_iter=1000)
   model.fit(X_train, y_train)

   # Predict and evaluate
   y_pred = model.predict(X_test)
   print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
   ```

2. **Decision Boundaries:**
   - **Concept:** The decision boundary is the surface that separates the different classes in a classification problem.
   - **Visualization:** Plotting the decision boundary helps in understanding how the model is dividing the feature space.
   - **Example:** Visualizing the decision boundary for a logistic regression model on a 2D dataset.

3. **Confusion Matrix:**
   - **Definition:** A confusion matrix is a table used to evaluate the performance of a classification algorithm, showing the true positives, false positives, true negatives, and false negatives.
   - **Importance:** It provides insight into not just the overall accuracy but also the types of errors being made.
   - **Example:**
     ```python
     from sklearn.metrics import confusion_matrix
     import seaborn as sns

     # Compute confusion matrix
     cm = confusion_matrix(y_test, y_pred)
     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
     ```

**Exercise:**
- Train a binary classifier on the MNIST dataset to classify digits as "5" or "not 5."
- Visualize the decision boundary on a smaller subset of the data.
- Calculate and interpret the confusion matrix.

---

## 2.2 Performance Measures, Multiclass Classification

**Objective:** Learn how to evaluate classification models using various performance metrics and understand the difference between multiclass and multilabel classification.

### Topics:

1. **Precision, Recall, F1 Score:**
   - **Precision:** The ratio of true positives to the sum of true positives and false positives.
     - **Formula:** `Precision = TP / (TP + FP)`
   - **Recall (Sensitivity):** The ratio of true positives to the sum of true positives and false negatives.
     - **Formula:** `Recall = TP / (TP + FN)`
   - **F1 Score:** The harmonic mean of precision and recall, balancing both concerns.
     - **Formula:** `F1 = 2 * (Precision * Recall) / (Precision + Recall)`

   **Example:**
   ```python
   from sklearn.metrics import precision_score, recall_score, f1_score

   print(f'Precision: {precision_score(y_test, y_pred)}')
   print(f'Recall: {recall_score(y_test, y_pred)}')
   print(f'F1 Score: {f1_score(y_test, y_pred)}')
   ```

2. **ROC Curve:**
   - **Definition:** The ROC (Receiver Operating Characteristic) curve plots the true positive rate against the false positive rate at various threshold settings.
   - **AUC (Area Under the Curve):** Measures the entire two-dimensional area underneath the ROC curve, with a higher value indicating a better model.
   - **Example:**
     ```python
     from sklearn.metrics import roc_curve, roc_auc_score
     import matplotlib.pyplot as plt

     y_scores = model.decision_function(X_test)
     fpr, tpr, thresholds = roc_curve(y_test, y_scores)

     plt.plot(fpr, tpr)
     plt.title('ROC Curve')
     plt.xlabel('False Positive Rate')
     plt.ylabel('True Positive Rate')
     plt.show()

     print(f'ROC AUC Score: {roc_auc_score(y_test, y_scores)}')
     ```

3. **Multiclass vs. Multilabel Classification:**
   - **Multiclass Classification:** Involves classification into more than two categories (e.g., classifying MNIST digits from 0 to 9).
   - **Multilabel Classification:** Each instance can be assigned multiple labels (e.g., predicting multiple diseases from a medical record).

   **Example:**
   ```python
   # Multiclass classification using a Logistic Regression model
   model = LogisticRegression(multi_class='multinomial', max_iter=1000)
   model.fit(X_train, y_train)  # Where y_train contains multiple classes (e.g., 0 to 9)

   # Multilabel classification (requires appropriate dataset)
   ```

**Exercise:**
- Calculate the precision, recall, and F1 score for a multiclass classifier.
- Plot the ROC curve for a binary classifier and calculate the AUC score.
- Train a model for multiclass classification on the MNIST dataset.

---

## 2.3 Training Models: Linear Regression, Gradient Descent

**Objective:** Dive into linear regression and optimization techniques, focusing on gradient descent.

### Topics:

1. **Linear Regression Model:**
   - **Definition:** A linear regression model predicts a target value by fitting a linear relationship between the features and the target.
   - **Formula:** `y = wx + b`, where `w` is the weight (slope) and `b` is the bias (intercept).
   - **Example:**
     ```python
     from sklearn.linear_model import LinearRegression

     # Load sample dataset
     from sklearn.datasets import make_regression
     X, y = make_regression(n_samples=100, n_features=1, noise=10)

     # Train linear regression model
     model = LinearRegression()
     model.fit(X, y)

     # Predict and plot
     import matplotlib.pyplot as plt
     plt.scatter(X, y)
     plt.plot(X, model.predict(X), color='red')
     plt.title('Linear Regression Fit')
     plt.show()
     ```

2. **Gradient Descent Algorithm:**
   - **Definition:** An optimization algorithm used to minimize the cost function by iteratively adjusting the model parameters.
   - **Concepts:**
     - **Learning Rate:** Controls how big a step is taken during each iteration. Too small a learning rate results in slow convergence; too large a learning rate can overshoot the optimal solution.
     - **Convergence:** The point where further iterations result in negligible change in the cost function.
   - **Mathematical Representation:**
     - Update rule: `θ = θ - η * ∇J(θ)`, where `θ` is the parameter vector, `η` is the learning rate, and `∇J(θ)` is the gradient of the cost function.

   **Code Example:**
   ```python
   import numpy as np

   # Gradient Descent function
   def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
       m = len(y)
       theta = np.random.randn(2, 1)
       X_b = np.c_[np.ones((m, 1)), X]  # Add bias term

       for iteration in range(n_iterations):
           gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
           theta -= learning_rate * gradients

       return theta

   # Generate synthetic data
   X = 2 * np.random.rand(100, 1)
   y = 4 + 3 * X + np.random.randn(100, 1)

   # Train using Gradient Descent
   theta = gradient_descent(X, y)
   print(f'Theta values: {theta}')
   ```

3. **Learning Rate and Convergence:**
   - **Learning Rate:** If too high, gradient descent may fail to converge; if too low, it may converge too slowly.
   - **Convergence:** Monitoring the cost function over iterations helps ensure the model is moving towards an optimal solution.

**Exercise:**
- Implement gradient descent from scratch and compare it with the results of `LinearRegression` from Scikit-learn.
- Experiment with different learning rates and observe the impact on convergence.

---

## 2.4 Regularized Linear Models: Ridge & Lasso Regression

**Objective:** Explore regularization techniques like Ridge and Lasso regression to improve model generalization.

### Topics:

1. **Ridge Regression (L2 Regularization):**
   - **Definition:** Ridge regression adds a penalty equal to the square of the magnitude of coefficients to the cost function to shrink the coefficients.
   - **Formula:** `J(θ) = RSS + α * (Σθ_j^2)`, where `RSS` is the residual sum of squares and `α` is the regularization strength.
   - **Impact:** Helps to reduce overfitting by keeping the coefficients small.
   - **Example:**
     ```python
     from sklearn.linear

_model import Ridge

     # Train a Ridge regression model
     model = Ridge(alpha=1.0)
     model.fit(X, y)

     print(f'Ridge Coefficients: {model.coef_}')
     ```

2. **Lasso Regression (L1 Regularization):**
   - **Definition:** Lasso regression adds a penalty equal to the absolute value of the coefficients to the cost function, potentially driving some coefficients to zero, resulting in feature selection.
   - **Formula:** `J(θ) = RSS + α * (Σ|θ_j|)`.
   - **Impact:** Helps in feature selection by shrinking some coefficients to zero.
   - **Example:**
     ```python
     from sklearn.linear_model import Lasso

     # Train a Lasso regression model
     model = Lasso(alpha=0.1)
     model.fit(X, y)

     print(f'Lasso Coefficients: {model.coef_}')
     ```

3. **Elastic Net:**
   - **Definition:** Elastic Net combines both L1 and L2 regularization, balancing the benefits of Ridge and Lasso regression.
   - **Formula:** `J(θ) = RSS + α * (ρ * Σ|θ_j| + (1-ρ) * Σθ_j^2)`, where `ρ` controls the balance between L1 and L2 penalties.
   - **Impact:** Useful when there are multiple features that are correlated with each other.

   **Example:**
   ```python
   from sklearn.linear_model import ElasticNet

   # Train an ElasticNet regression model
   model = ElasticNet(alpha=0.1, l1_ratio=0.5)
   model.fit(X, y)

   print(f'ElasticNet Coefficients: {model.coef_}')
   ```

**Exercise:**
- Compare the performance of Ridge, Lasso, and Elastic Net regression on a dataset.
- Experiment with different values of `alpha` and observe their impact on the coefficients.
