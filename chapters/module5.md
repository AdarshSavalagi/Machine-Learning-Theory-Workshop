## Module 5: Ensemble Learning and Unsupervised Learning

### 5.1 Ensemble Learning and Random Forests

**Objective:** Explore ensemble methods to boost model performance and understand the principles behind Random Forests.

#### Topics:

1. **Voting Classifiers:**
   - **Concept:** Combine multiple models (classifiers) to improve overall predictions.
   - **Types:**
     - **Hard Voting:** Majority rule voting among classifiers.
     - **Soft Voting:** Averaging predicted probabilities for each class.
   - **Example:**
     ```python
     from sklearn.ensemble import VotingClassifier
     from sklearn.linear_model import LogisticRegression
     from sklearn.svm import SVC
     from sklearn.tree import DecisionTreeClassifier

     clf1 = LogisticRegression()
     clf2 = SVC(probability=True)
     clf3 = DecisionTreeClassifier()

     voting_clf = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2), ('dt', clf3)], voting='soft')
     voting_clf.fit(X_train, y_train)
     ```

2. **Bagging and Pasting:**
   - **Bagging (Bootstrap Aggregating):**
     - **Concept:** Train several models independently using different subsets of the training data sampled with replacement.
     - **Advantage:** Reduces variance and helps prevent overfitting.
   - **Pasting:**
     - **Concept:** Similar to bagging but without replacement (each subset is unique).
   - **Bagging Classifier Example:**
     ```python
     from sklearn.ensemble import BaggingClassifier
     from sklearn.tree import DecisionTreeClassifier

     bagging_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, bootstrap=True)
     bagging_clf.fit(X_train, y_train)
     ```

3. **Random Forests:**
   - **Concept:** An ensemble of decision trees, where each tree is trained on a random subset of the data and features.
   - **Advantages:**
     - **Robustness:** Handles large datasets and reduces overfitting.
     - **Feature Importance:** Provides insights into which features are most important.
   - **Implementation:**
     ```python
     from sklearn.ensemble import RandomForestClassifier

     rf_clf = RandomForestClassifier(n_estimators=100)
     rf_clf.fit(X_train, y_train)
     importance = rf_clf.feature_importances_
     ```

**Exercise:**
- Train a Random Forest on a dataset and evaluate its performance compared to individual decision trees.
- Experiment with voting classifiers by combining different models and assess the impact on accuracy.
- Explore the effect of bagging and pasting on model performance.

---

### 5.2 Unsupervised Learning Techniques: Clustering

**Objective:** Understand the fundamentals of unsupervised learning, focusing on various clustering techniques.

#### Topics:

1. **K-Means Clustering:**
   - **Concept:** Partition the dataset into \(k\) distinct, non-overlapping clusters.
   - **Algorithm:**
     1. Initialize \(k\) centroids randomly.
     2. Assign each data point to the nearest centroid.
     3. Recompute centroids based on the mean of assigned points.
     4. Repeat until convergence.
   - **Example:**
     ```python
     from sklearn.cluster import KMeans

     kmeans = KMeans(n_clusters=3)
     kmeans.fit(X)
     labels = kmeans.labels_
     centroids = kmeans.cluster_centers_
     ```

2. **Spectral Clustering:**
   - **Concept:** Clustering based on the eigenvalues of a similarity matrix derived from the data.
   - **Advantages:** Effective for non-convex clusters and data that is not well-separated in Euclidean space.
   - **Example:**
     ```python
     from sklearn.cluster import SpectralClustering

     spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors')
     labels = spectral.fit_predict(X)
     ```

3. **Hierarchical Clustering:**
   - **Concept:** Build a hierarchy of clusters either by agglomerating or dividing the dataset.
   - **Types:**
     - **Agglomerative (Bottom-Up):** Start with individual points and merge clusters iteratively.
     - **Divisive (Top-Down):** Start with one cluster and recursively split it into smaller clusters.
   - **Dendrogram:** A tree-like diagram that records the sequences of merges or splits.
   - **Example:**
     ```python
     from scipy.cluster.hierarchy import dendrogram, linkage

     Z = linkage(X, 'ward')
     dendrogram(Z)
     ```

**Exercise:**
- Implement K-Means clustering on a dataset and visualize the clusters.
- Perform Spectral Clustering and compare the results with K-Means.
- Create a dendrogram for hierarchical clustering and analyze the cluster hierarchy.
