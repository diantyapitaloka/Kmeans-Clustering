## ⛄🌀⭐ Kmeans Clustering ⭐🌀⛄
- Before running the model, you need to decide how many groups (clusters) actually exist in your mall data. The most common method is the Elbow Method. You calculate the Within-Cluster Sum of Squares (WCSS) for different values of $k$ and plot them. The point where the rate of decrease sharply slows down (forming an "elbow") indicates the ideal number of clusters.
- Sensitivity to Outliers: A single "Whale" (an extreme high-spender far beyond the norm) can pull a centroid away from the actual center of a cluster. These outliers can distort your segments. It is often better to detect and handle these anomalies before running the algorithm.
- If your mall data starts including 20+ variables (age, visit frequency, dwell time, parking usage, etc.), the concept of "distance" starts to break down. In high-dimensional space, all points begin to look equally far apart. To keep your clusters meaningful, it’s common to use PCA (Principal Component Analysis) to reduce the number of features before running K-Means.
- The algorithm works through an iterative process: Assignment (assigning customers to the nearest centroid) and Update (moving the centroid to the mean of its assigned points). The model "converges" when the centroids stop moving significantly. If your mall data is highly volatile, you might need to increase the max_iter parameter to ensure the model actually finishes its journey.
- K-Means relies on Euclidean distance, which is extremely sensitive to the scale of your data. If your "Annual Income" is in the thousands but your "Spending Score" is 1–100, the algorithm will treat income as the dominant factor simply because the numbers are larger. You must apply Standardization (Z-score normalization) so that $1,000 in income doesn't outweigh a 10-point shift in spending behavior.
- Strategic Segment Profiling: Once the algorithm converges, you can visualize the results on a scatter plot to identify distinct behavioral archetypes—such as High Earners/High Spenders or Low Earners/Low Spenders—allowing the mall management to create highly personalized marketing campaigns for each unique group.
- K-Means++ Initialization: Standard K-Means used to pick initial "centroids" randomly, which often led to poor convergence. K-Means++ is the modern standard; it selects the first centroid randomly but picks subsequent ones based on their distance from existing centroids, ensuring they are spread out from the start.
- While the Elbow Method is great for WCSS, the Silhouette Score is the "sanity check" for cluster quality. It measures how similar an object is to its own cluster compared to other clusters. A high Silhouette Score means your mall segments are well-defined and distinct, whereas a low score suggests your clusters are overlapping and messy.
- Impact of Categorical Data: Standard K-Means is designed for continuous numerical data (like $\$$ or scores). If your mall data includes categorical info like "Gender" or "Membership Type," you shouldn't use Euclidean distance. Instead, you would look into K-Prototypes or convert categories into numerical representations carefully to avoid biasing the distance.
- The Problem of Hard Assignment: K-Means is a "Hard Clustering" algorithm—every customers must belong to exactly one group. In reality, a customer might sit right on the border between "Budget Conscious" and "Average Spender." For those cases, "Soft Clustering" methods (like Gaussian Mixture Models) are used, but K-Means remains the preferred choice for clear-cut business action.
- Creating an unsupervised learning model using the K-Means Clustering technique. Data on visitors from a mall.
- Non-Global Optima: K-Means is a "greedy" algorithm. This means it can sometimes get stuck in a local minimum rather than finding the best overall (global) solution. Running the algorithm multiple times with different initializations (the n_init parameter in many libraries) helps ensure you find the most stable configuration.
- The "Curse of Dimensionality": As you add more features (Age, Income, Visit Frequency, Tenures, etc.), the "distance" between any two points starts to become uniform. In high-dimensional space, everything looks far away from everything else, making clusters less meaningful. Stick to the most impactful features.
- The Assumption of Spherical Clusters: K-Means naturally tries to create "round" clusters. If your customer segments are elongated, elliptical, or irregularly shaped, K-Means might struggle. It essentially assumes that the variance of each cluster is equal and the features are independent.
- Feature Scaling is Mandatory: Since K-Means relies on Euclidean distance, features with larger scales will dominate the calculation. If "Annual Income" is in the thousands and "Spending Score" is 1-100, the model will essentially ignore the score. Always apply StandardScaler or MinMaxScaler before fitting.
- The Silhouette Coefficient: While the Elbow Method measures compactness (WCSS), the Silhouette Score measures how well-separated the clusters are. A score near $+1$ means the customer is far from neighboring clusters, while a score near $0$ suggests they are on the boundary.
- The Silhouette Coefficient: While the Elbow Method is the go-to, the Silhouette Score is a more refined metric. It measures how similar a point is to its own cluster compared to other clusters. A score near $+1$ indicates that the customer is well-matched to their group and poorly matched to neighboring ones.
- Handling High-Dimensional Data: While mall data often focuses on "Income" and "Spending Score," adding more features (like age, frequency of visits, or dwell time) can lead to the "Curse of Dimensionality." In high-dimensional spaces, the distance between any two points becomes nearly constant, making it harder for K-Means to find meaningful clusters.
- Optimal Cluster Selection: Before running the model, you must determine the ideal number of groups ($k$) using the Elbow Method. By plotting the Within-Cluster Sum of Squares (WCSS) against various values of $k$, you identify the "elbow" point where adding more clusters no longer significantly improves the model's tightness.
- Centroid Initialization and Iteration: The algorithm begins by placing $k$ random centroids in the data space and assigning each customer to the nearest one. It then iteratively recalculates the center of each group and reassigns points until the clusters become stable and the total variance is minimized.
- The Convergence Criterion: The iterative process doesn't go on forever. It stops when the centroids no longer change position significantly or when a pre-defined maximum number of iterations is reached, signaling that the model has found the most stable configuration.
- Minimizing the Objective Function: Mathematically, the goal of the algorithm is to minimize the Inertia, also known as the Within-Cluster Sum of Squares (WCSS). 
- Sensitivity to Outliers: K-Means is particularly sensitive to outliers because they can significantly pull the centroids away from the true center of the main cluster. Pre-processing the mall data to handle extreme "big spenders" or "rare earners" is crucial for balanced groups.
- The K-Means++ Initialization: To avoid the "random initialization trap"—where poor starting points lead to suboptimal clusters—most modern implementations use K-Means++. This smart initialization spreads the initial centroids far apart from each other to ensure a more reliable result.
- Feature Scaling Necessity: Since K-Means relies on Euclidean distance, features with larger scales can dominate the result. You must apply Standardization or Normalization to ensure that "Annual Income" and "Spending Score" contribute equally to the distance calculations.
- Strategic Segment Profiling: Once the algorithm converges, you can visualize the results on a scatter plot to identify distinct behavioral archetypes. This allows mall management to distinguish between groups like High Earners/High Spenders and Low Earners/Low Spenders, facilitating highly personalized marketing campaigns.
- Distance-Based Logic: The model relies on calculating the Euclidean distance between data points to ensure that customers within a group are as similar as possible. Because this math is sensitive to scale, it is crucial to normalize your data (like income and age) so that one feature doesn't disproportionately influence the groupings.
- Actionable Business Insights: Beyond simple grouping, K-means provides the "coordinates" of your average customer in each segment to help predict future behavior. These insights enable precision targeting, such as offering luxury rewards to top-tier spenders while sending discount coupons to price-sensitive clusters.
- K-Means naturally tries to create spherical (circular) clusters of similar sizes. If your mall data actually has elongated or "U-shaped" distributions (e.g., a group of shoppers that bridges two different spending tiers), K-Means will struggle. It will try to "cut" those shapes into circles, which might misrepresent the actual flow of your customer base.
```
import pandas as pd
```

## ⛄🌀⭐ Convert to Data Frame ⭐🌀⛄
Convert csv file to dataframe.
```
df = pd.read_csv('Mall_Customers.csv')
```

## ⛄🌀⭐ Show Head Function ⭐🌀⛄
Show the first 3 lines.
```
df.head(3)
```

## ⛄🌀⭐ Final Appearance ⭐🌀⛄
The appearance of the first 3 rows of the dataframe above is as follows.
![image](https://github.com/diantyapitaloka/Kmeans-Clustering/assets/147487436/510ce9fb-d8fb-43de-bc25-d90e7d1adfad)

## ⛄🌀⭐ Preporcessing Data ⭐🌀⛄
Then we will do a little preprocessing, namely changing the column names to make them more uniform. Then the gender column is a categorical column, so we will convert the data into numerical data.

Change column name.
```
df = df.rename(columns={'Gender': 'gender', 'Age': 'age',
'Annual Income (k$)': 'annual_income',
'Spending Score (1-100)': 'spending_score'})
```
 
Convert categorical data to numeric data.
```
df['gender'].replace(['Female', 'Male'], [0,1], inplace=True)
```
 
Display data that has been preprocessed.
```
df.head(3)
```

After preprocessing by changing the column names to make them more uniform, the results are as below.
![image](https://github.com/diantyapitaloka/Kmeans-Clustering/assets/147487436/7b5791e0-9fc0-45be-b529-0e8a051fcb34)

## ⛄🌀⭐ Import KMeans ⭐🌀⛄
In the next stage we will import K-Means. At this stage we will also remove the Customer ID and gender columns because they are less relevant for the clustering process. Next, we will determine the optimal K value using the Elbow method. The K-means library from SKLearn provides a function to calculate the inertia of K-Means with a certain number of K. Here we will create a list containing the inertia of K values between 1 and 11.

```
from sklearn.cluster import KMeans
```

## ⛄🌀⭐ Remove Useless Data ⭐🌀⛄
Remove customer ID and gender columns.
```
X = df.drop(['CustomerID', 'gender'], axis=1)
```

## ⛄🌀⭐ Creating a List ⭐🌀⛄
Create a list containing inertia.
```
clusters = []
for i in range(1,11):
km = KMeans(n_clusters=i).fit(X)
clusters.append(km.inertia_)
```

## ⛄🌀⭐ Inertia Plot ⭐🌀⛄
Run the code below to create an inertia plot for each K value. According to the plot below, we can see that the elbow is at a K value equal to 5, where the decrease in inertia is no longer significant after the K value equals 5. Don't forget to import the library needed to create a plot.

```
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```

## ⛄🌀⭐ Create an Inertia Plot ⭐🌀⛄
Create an inertia plot
```
fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title('Search Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')
```

The results of the code above display an inertia plot as follows.
![image](https://github.com/diantyapitaloka/Kmeans-Clustering/assets/147487436/b3e48bfb-9f25-49a7-bd02-21f8f597a839)

## ⛄🌀⭐ Elbow Method ⭐🌀⛄
Finally, we can retrain K-Means with the number of K obtained from the Elbow method. Then we can plot the K-Means clustering results by running the code below.

Create a KMeans object
```
km5 = KMeans(n_clusters=5).fit(X)
```
 
Add a label column to the dataset
```
X['Labels'] = km5.labels_
```

## ⛄🌀⭐ Create KMeans Plot ⭐🌀⛄
Create a KMeans plot with 5 clusters
```
plt.figure(figsize=(8,4))
sns.scatterplot(x=X['annual_income'], y=X['spending_score']
hue=X['Labels'],
palette=sns.color_palette('hls', 5))
plt.title('KMeans with 5 Clusters')
plt.show()
```

## ⛄🌀⭐ Output ⭐🌀⛄
So if the code above is executed, the KMeans display with 5 clusters will look like the one below.
The Output is :
![image](https://github.com/diantyapitaloka/Kmeans-Clustering/assets/147487436/6bf9a724-dc93-47d8-aecb-f3219f455ef1)


## ⛄🌀⭐ License ⭐🌀⛄
- Copyright by Diantya Pitaloka

  
