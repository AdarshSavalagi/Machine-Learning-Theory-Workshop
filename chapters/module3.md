## Module 3: Dimensionality Reduction and Support Vector Machines

### 3.1 Dimensionality Reduction: The Curse of Dimensionality

**Objective:** Learn about the challenges posed by high-dimensional data and explore techniques for reducing data dimensions effectively.

#### Topics:

1. **The Curse of Dimensionality:**
   - **Concept:** As the number of dimensions (features) increases, the volume of the space increases exponentially, making the data sparse and harder to analyze. This phenomenon is known as the "Curse of Dimensionality."
   - **Challenges:**
     - **Distance Metrics:** In high-dimensional space, the difference between the nearest and farthest data points becomes negligible, making distance-based algorithms less effective.
     - **Overfitting:** High-dimensional data often leads to models that are overly complex and prone to overfitting.
   - **Example:** In a high-dimensional dataset, a nearest-neighbor classifier may perform poorly because all points tend to be nearly equidistant.

2. **Feature Selection vs. Feature Extraction:**
   - **Feature Selection:**
     - **Definition:** Selecting a subset of the original features based on some criteria (e.g., correlation with the target variable, mutual information).
     - **Techniques:** 
       - **Filter Methods:** Use statistical measures to select features (e.g., Chi-square, ANOVA).
       - **Wrapper Methods:** Use a predictive model to evaluate feature subsets (e.g., Recursive Feature Elimination, Forward Selection).
       - **Embedded Methods:** Feature selection occurs naturally during model training (e.g., Lasso Regression).
   - **Feature Extraction:**
     - **Definition:** Creating new features by transforming the original features (e.g., using PCA to combine features into principal components).
     - **Techniques:**
       - **Principal Component Analysis (PCA):** Reduces dimensionality by projecting data onto a lower-dimensional space.
       - **Linear Discriminant Analysis (LDA):** Projects data onto a lower-dimensional space with maximum class separability.

**Exercise:**
- Apply feature selection techniques on a dataset and observe how the model's performance changes.
- Use PCA to reduce the dimensionality of a high-dimensional dataset and visualize the result.

---

### 3.2 PCA, Linear Discriminant Analysis (LDA)

**Objective:** Understand and apply Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) for dimensionality reduction.

#### Topics:

1. **Principal Component Analysis (PCA):**
   - **Definition:** PCA is a technique for reducing the dimensionality of a dataset by transforming the original features into a set of linearly uncorrelated components called principal components.
   - **Steps:**
     1. **Standardize the Data:** Center the data around the mean.
     2. **Compute the Covariance Matrix:** Determine the relationships between the features.
     3. **Eigen Decomposition:** Calculate the eigenvalues and eigenvectors of the covariance matrix.
     4. **Project the Data:** Transform the original data into the new feature space defined by the principal components.
   - **Choosing the Number of Components:** Typically, the number of components is chosen to explain a desired amount of variance (e.g., 95%).
   - **Example:**
     ```python
     from sklearn.decomposition import PCA
     import matplotlib.pyplot as plt

     # Load dataset (e.g., Iris)
     from sklearn.datasets import load_iris
     data = load_iris()
     X, y = data.data, data.target

     # Apply PCA
     pca = PCA(n_components=2)
     X_pca = pca.fit_transform(X)

     # Visualize the PCA result
     plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
     plt.xlabel('Principal Component 1')
     plt.ylabel('Principal Component 2')
     plt.title('PCA on Iris Dataset')
     plt.show()
     ```

2. **Linear Discriminant Analysis (LDA):**
   - **Definition:** LDA is a technique for dimensionality reduction that aims to project the data onto a lower-dimensional space with maximum class separability.
   - **Difference from PCA:** While PCA focuses on maximizing variance, LDA focuses on maximizing the separation between different classes.
   - **Steps:**
     1. **Compute the Within-Class and Between-Class Scatter Matrices:** Measure how much the data points vary within and between classes.
     2. **Eigen Decomposition:** Calculate the eigenvalues and eigenvectors to determine the new axes that maximize class separability.
     3. **Project the Data:** Transform the original data into the new feature space defined by the LDA components.
   - **Example:**
     ```python
     from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
     import matplotlib.pyplot as plt

     # Apply LDA
     lda = LDA(n_components=2)
     X_lda = lda.fit_transform(X, y)

     # Visualize the LDA result
     plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)
     plt.xlabel('LDA Component 1')
     plt.ylabel('LDA Component 2')
     plt.title('LDA on Iris Dataset')
     plt.show()
     ```

3. **Choosing the Number of Components:**
   - **PCA:** Choose components that explain a sufficient amount of variance (e.g., 95% of the total variance).
   - **LDA:** Limited by the number of classes minus one (e.g., if there are three classes, LDA can reduce to at most two components).

**Exercise:**
- Apply PCA and LDA on a high-dimensional dataset and compare the results in terms of visualization and classification performance.
- Experiment with different numbers of components and analyze their impact.

---

### 3.3 Support Vector Machines (SVM)

**Objective:** Master Support Vector Machines (SVMs) for classification and regression tasks, including linear and nonlinear decision boundaries.

#### Topics:

1. **Linear SVM Classification:**
   - **Definition:** SVM is a supervised learning algorithm that finds the hyperplane which best separates the classes in the feature space.
   - **Concepts:**
     - **Hyperplane:** The decision boundary that separates different classes.
     - **Support Vectors:** The data points that are closest to the hyperplane and influence its position.
     - **Margin:** The distance between the hyperplane and the nearest support vectors. SVM aims to maximize this margin.
   - **Example:**
     ```python
     from sklearn.svm import SVC
     import matplotlib.pyplot as plt
     import numpy as np

     # Load sample dataset (e.g., linearly separable)
     X = np.array([[1, 2], [2, 3], [3, 3], [5, 5], [1, 0.5], [7, 8]])
     y = np.array([0, 0, 0, 1, 1, 1])

     # Train a linear SVM
     svm = SVC(kernel='linear')
     svm.fit(X, y)

     # Plot decision boundary
     w = svm.coef_[0]
     b = svm.intercept_[0]
     x0 = np.linspace(0, 8, 100)
     decision_boundary = -w[0]/w[1] * x0 - b/w[1]

     plt.scatter(X[:, 0], X[:, 1], c=y)
     plt.plot(x0, decision_boundary, "k-")
     plt.title('Linear SVM Decision Boundary')
     plt.show()
     ```

2. **Nonlinear SVMs:**
   - **Definition:** Nonlinear SVMs can classify data that are not linearly separable by using kernel functions to map the data into a higher-dimensional space.
   - **Kernel Trick:** The kernel trick allows SVMs to operate in the original feature space while implicitly calculating the separation in a higher-dimensional space.
   - **Common Kernels:**
     - **Polynomial Kernel:** Captures interactions between features by adding polynomial terms.
     - **Radial Basis Function (RBF) Kernel:** Maps data into a higher-dimensional space where it becomes linearly separable.
   - **Example:**
     ```python
     # Nonlinear SVM with RBF kernel
     svm = SVC(kernel='rbf', gamma='scale')
     svm.fit(X, y)

     # Plot decision boundary
     plt.scatter(X[:, 0], X[:, 1], c=y)
     plt.title('Nonlinear SVM with RBF Kernel')
     plt.show()
     ```

3. **Kernel Trick:**
   - **Definition:** The kernel trick enables SVMs to efficiently compute the separation in a higher-dimensional space without explicitly transforming the data.
   - **Mathematical Representation:** Instead of mapping `x` to a higher dimension, SVMs use a kernel function `K(x, y)` that computes the dot product in that space.
   - **Advantages:** Allows SVMs to handle very complex and nonlinear decision boundaries without the computational cost of working in the higher-dimensional space directly.

**Exercise:**
- Train a linear SVM on a linearly separable dataset and visualize the decision boundary.
- Train a nonlinear SVM with an RBF kernel on a non-linearly separable dataset and visualize the decision boundary.
- Experiment with different kernel functions and parameters (e.g., `gamma` for RBF) to see how they affect the model's performance.

