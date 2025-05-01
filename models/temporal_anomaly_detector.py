
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from numpy.linalg import norm
import matplotlib.pyplot as plt

df = pd.read_csv("B:\BOSTON UNI\Acad\TDS\datasets\PWV_processed.csv")

trash_keywords = [
    'improper storage trash',
    'improper storage of trash',
    'overfilling of barrel/dumpster'
]
df = df[~df['description'].isin(trash_keywords)]

df['status_dttm'] = pd.to_datetime(df['status_dttm'], errors='coerce')
df['year_month'] = df['status_dttm'].dt.to_period('M').astype(str)


grouped = df.groupby(['violation_zip', 'year_month', 'description']).size().unstack(fill_value=0)

valid_zips = df['violation_zip'].value_counts()[lambda x: x > 8000].index
grouped = grouped[grouped.index.get_level_values(0).isin(valid_zips)]


scaler = StandardScaler()
X = scaler.fit_transform(grouped)


kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)
grouped['cluster'] = labels


centroids = kmeans.cluster_centers_
grouped['anomaly_score'] = [
    norm(x - centroids[cluster])
    for x, cluster in zip(X, labels)
]

grouped = grouped.reset_index().rename_axis(None, axis=1)

top_anomalies = grouped.sort_values("anomaly_score", ascending=False).head(20)

print("\nTop Anomalies:")
print(top_anomalies[['violation_zip', 'year_month', 'anomaly_score']])
