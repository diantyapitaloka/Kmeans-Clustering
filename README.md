# Kmeans Clustering
Creating an unsupervised learning model using the K-Means Clustering technique. Data on visitors from a mall.
- import pandas as pd

Convert csv file to dataframe.
- df = pd.read_csv('Mall_Customers.csv')
 
Show the first 3 lines.
- df.head(3)

The appearance of the first 3 rows of the dataframe above is as follows.
![image](https://github.com/diantyapitaloka/Kmeans-Clustering/assets/147487436/510ce9fb-d8fb-43de-bc25-d90e7d1adfad)

Then we will do a little preprocessing, namely changing the column names to make them more uniform. Then the gender column is a categorical column, so we will convert the data into numerical data.

Change column name.
- df = df.rename(columns={'Gender': 'gender', 'Age': 'age',
- 'Annual Income (k$)': 'annual_income',
- 'Spending Score (1-100)': 'spending_score'})
 
Convert categorical data to numeric data.
- df['gender'].replace(['Female', 'Male'], [0,1], inplace=True)
 
Display data that has been preprocessed.
- df.head(3)

After preprocessing by changing the column names to make them more uniform, the results are as below.
![image](https://github.com/diantyapitaloka/Kmeans-Clustering/assets/147487436/7b5791e0-9fc0-45be-b529-0e8a051fcb34)

In the next stage we will import K-Means. At this stage we will also remove the Customer ID and gender columns because they are less relevant for the clustering process. Next, we will determine the optimal K value using the Elbow method. The K-means library from SKLearn provides a function to calculate the inertia of K-Means with a certain number of K. Here we will create a list containing the inertia of K values between 1 and 11.

- from sklearn.cluster import KMeans
 
Remove customer ID and gender columns.
- X = df.drop(['CustomerID', 'gender'], axis=1)
 
Create a list containing inertia.
- clusters = []
- for i in range(1,11):
- km = KMeans(n_clusters=i).fit(X)
- clusters.append(km.inertia_)

Run the code below to create an inertia plot for each K value. According to the plot below, we can see that the elbow is at a K value equal to 5, where the decrease in inertia is no longer significant after the K value equals 5. Don't forget to import the library needed to create a plot.

- import matplotlib.pyplot as plt
- %matplotlib inline
- import seaborn as sns
 
Create an inertia plot
- fig, ax = plt.subplots(figsize=(8, 4))
- sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
- ax.set_title('Search Elbow')
- ax.set_xlabel('Clusters')
- ax.set_ylabel('Inertia')

The results of the code above display an inertia plot as follows.
![image](https://github.com/diantyapitaloka/Kmeans-Clustering/assets/147487436/b3e48bfb-9f25-49a7-bd02-21f8f597a839)

Finally, we can retrain K-Means with the number of K obtained from the Elbow method. Then we can plot the K-Means clustering results by running the code below.

Create a KMeans object
- km5 = KMeans(n_clusters=5).fit(X)
 
Add a label column to the dataset
- X['Labels'] = km5.labels_
 
Create a KMeans plot with 5 clusters
- plt.figure(figsize=(8,4))
- sns.scatterplot(x=X['annual_income'], y=X['spending_score']
- hue=X['Labels'],
- palette=sns.color_palette('hls', 5))
- plt.title('KMeans with 5 Clusters')
- plt.show()

So if the code above is executed, the KMeans display with 5 clusters will look like the one below.
![image](https://github.com/diantyapitaloka/Kmeans-Clustering/assets/147487436/e3e010f9-3c96-470f-b000-e327bfa06444)


