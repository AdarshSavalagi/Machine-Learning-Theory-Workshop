Here's the content for **Module 4: Decision Trees**:

---

## Module 4: Decision Trees

### 4.1 Decision Trees: Univariate Trees

**Objective:** Explore decision trees and understand how they work for both classification and regression tasks.

#### Topics:

1. **How Decision Trees Work:**
   - **Concept:** Decision trees are non-parametric models that split the data into subsets based on feature values, creating a tree-like structure of decisions.
   - **Structure:**
     - **Root Node:** The top node that represents the entire dataset.
     - **Internal Nodes:** Nodes that represent a feature test, branching into child nodes.
     - **Leaf Nodes:** Terminal nodes that represent the final output, either a class label or a regression value.
   - **Splitting Criteria:** Decision trees use criteria like Gini impurity or Information Gain to determine the best split at each node.

2. **Gini Impurity and Information Gain:**
   - **Gini Impurity:**
     - **Definition:** A measure of how often a randomly chosen element from the set would be incorrectly labeled if it were randomly labeled according to the distribution of labels in the subset.
     - **Formula:** \( \text{Gini}(D) = 1 - \sum_{i=1}^{n} p_i^2 \)
     - **Usage:** Used in the CART algorithm for classification trees.
   - **Information Gain:**
     - **Definition:** The reduction in entropy after a dataset is split on an attribute.
     - **Formula:** \( \text{Information Gain}(D, A) = \text{Entropy}(D) - \sum_{v \in \text{Values}(A)} \frac{|D_v|}{|D|} \times \text{Entropy}(D_v) \)
     - **Usage:** Used in the ID3 algorithm to decide the best feature to split on.

**Exercise:**
- Build a decision tree using a dataset like the Iris dataset and visualize its structure.
- Experiment with different splitting criteria (Gini vs. Entropy) and observe their impact on the tree structure.

---

### 4.2 Pruning, Rule Extraction, Making Predictions

**Objective:** Learn techniques to optimize decision trees, including pruning and rule extraction.

#### Topics:

1. **Pruning Techniques:**
   - **Definition:** Pruning is the process of removing branches from the tree that have little importance or are likely to overfit the model.
   - **Types:**
     - **Pre-Pruning (Early Stopping):** Halting the tree growth early based on certain conditions (e.g., minimum number of samples per node).
     - **Post-Pruning:** Removing nodes from a fully grown tree to reduce complexity.
   - **Advantages:** Pruning helps in improving the generalization of the tree and prevents overfitting.
   - **Example:**
     ```python
     from sklearn.tree import DecisionTreeClassifier

     tree = DecisionTreeClassifier(max_depth=3)  # Example of pre-pruning by limiting tree depth
     ```

2. **Rule Extraction from Trees:**
   - **Definition:** Extracting the decision rules from the tree in the form of if-else conditions.
   - **Advantages:** Simplifies the model, making it interpretable and easy to understand.
   - **Example:**
     ```python
     from sklearn.tree import export_text

     tree = DecisionTreeClassifier()
     tree.fit(X, y)
     rules = export_text(tree, feature_names=['Feature1', 'Feature2'])
     print(rules)
     ```

3. **Making Predictions with Trees:**
   - **Process:** To make a prediction, start at the root node, and move down the tree, following the branches according to the feature values, until you reach a leaf node.
   - **Example:**
     ```python
     prediction = tree.predict([[5.1, 3.5, 1.4, 0.2]])  # Predict class for a given sample
     ```

**Exercise:**
- Train a decision tree and extract the decision rules. Apply pruning and compare the performance of pruned and unpruned trees.
- Make predictions on a test dataset and analyze the accuracy.

---

### 4.3 Estimating Class Probabilities, Computational Complexity, CART Algorithm

**Objective:** Understand the probabilistic interpretation of decision trees and their computational complexity.

#### Topics:

1. **Estimating Class Probabilities:**
   - **Concept:** Decision trees can provide not just class predictions but also probabilities of each class.
   - **How It Works:** The probability is calculated as the proportion of samples of a particular class in the leaf node.
   - **Example:**
     ```python
     probabilities = tree.predict_proba([[5.1, 3.5, 1.4, 0.2]])  # Get class probabilities for a given sample
     ```

2. **Computational Complexity:**
   - **Training Complexity:** The complexity of training a decision tree is \(O(n \cdot m \cdot \log n)\), where \(n\) is the number of samples and \(m\) is the number of features.
   - **Prediction Complexity:** The complexity of making a prediction is \(O(\log n)\) due to the depth of the tree.
   - **Scalability:** Decision trees can handle large datasets but may become slow with very large and high-dimensional data.

 **CART Algorithm:**
   - **Definition:** CART (Classification and Regression Trees) is a popular algorithm for building decision trees. It uses Gini impurity for classification and mean squared error for regression.

## 1. Initialization
- Start with the entire training set \( D \) containing \( m \) training instances.
- Initialize a decision tree \( T \) with a single root node representing the entire training set.

## 2. Splitting Criteria
- For each node, the CART algorithm selects the feature \( k \) and threshold \( t_k \) that minimizes the cost function \( J(k, t_k) \).
- The cost function is defined as:
  \[
  J(k, t_k) = \frac{m_{\text{left}}}{m} G_{\text{left}} + \frac{m_{\text{right}}}{m} G_{\text{right}}
  \]
  where:
  - \( m_{\text{left}} \) and \( m_{\text{right}} \) are the number of instances in the left and right subsets, respectively.
  - \( G_{\text{left}} \) and \( G_{\text{right}} \) are measures of impurity for the left and right subsets, respectively.
  - \( m \) is the total number of instances at the current node.

## 3. Impurity Measures
- Common impurity measures \( G \) used for classification include:
  - **Gini Impurity**:
    \[
    G_i = 1 - \sum_{i=1}^{C} p_i^2
    \]
    where \( p_i \) is the proportion of instances of class \( i \) in the subset and \( C \) is the number of classes.
  - **Entropy**:
    \[
    G_i = -\sum_{i=1}^{C} p_i \log_2(p_i)
    \]
- For regression, **variance** is often used as the impurity measure.

## 4. Recursive Splitting
- Split the dataset into two subsets \( D_{\text{left}} \) and \( D_{\text{right}} \) based on the feature \( k \) and threshold \( t_k \) that minimize the cost function.
- Continue recursively applying the same logic to split each subset.

## 5. Stopping Criteria
The recursion stops when any of the following conditions are met:
- The maximum tree depth \( \text{max\_depth} \) is reached.
- The number of instances in a node is less than \( \text{min\_samples\_split} \).
- The number of instances in a leaf is less than \( \text{min\_samples\_leaf} \).
- The weight fraction of instances in a leaf node is less than \( \text{min\_weight\_fraction\_leaf} \).
- The maximum number of leaf nodes \( \text{max\_leaf\_nodes} \) is reached.

## 6. Prediction
- Once the tree is fully grown, predictions for a new instance are made by traversing the tree from the root node to a leaf node, following the splits determined during training.



   - **Example:**
     ```python
     tree = DecisionTreeClassifier(criterion='gini')  # Using Gini impurity in CART
     ```

**Exercise:**
- Train a decision tree on a dataset and estimate class probabilities for different samples.
- Analyze the computational complexity of training and predicting with decision trees.

---

### 4.4 Gini Impurity, Regularization Hyperparameters

**Objective:** Explore the use of Gini impurity and regularization hyperparameters in decision trees.

#### Topics:

1. **Gini Impurity vs. Entropy:**
   - **Gini Impurity:** Measures the frequency at which any element of the dataset would be misclassified when randomly labeled according to the distribution of labels in the subset.
   - **Entropy:** Measures the amount of information needed to represent the distribution of labels.
   - **Comparison:**
     - **Gini is faster:** Gini impurity is computationally less expensive.
     - **Entropy is more informative:** Entropy can provide a more detailed measure of uncertainty.

2. **Regularization Hyperparameters:**
   - **Objective:** Regularization prevents the model from becoming too complex, thus reducing overfitting.
   - **Common Hyperparameters:**
     - **Max Depth:** Limits the maximum depth of the tree.
     - **Min Samples Split:** The minimum number of samples required to split an internal node.
     - **Min Samples Leaf:** The minimum number of samples required to be at a leaf node.
     - **Max Features:** The number of features to consider when looking for the best split.
   - **Example:**
     ```python
     tree = DecisionTreeClassifier(max_depth=3, min_samples_split=5)  # Applying regularization
     ```

3. **Multivariate Trees:**
   - **Definition:** Trees that consider multiple features at a time when making splits, rather than just one feature.
   - **Advantages:** Can capture interactions between features.
   - **Challenges:** More computationally expensive and harder to interpret.

**Exercise:**
- Compare decision trees trained with Gini impurity vs. entropy as the splitting criterion.
- Tune regularization hyperparameters using cross-validation to find the optimal settings for a decision tree model.
- Explore the performance of multivariate trees on a complex dataset.

---

This content provides a comprehensive understanding of decision trees, from basic concepts to advanced techniques like pruning and regularization. It also covers practical implementation examples and exercises to reinforce learning.