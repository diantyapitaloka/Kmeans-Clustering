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


