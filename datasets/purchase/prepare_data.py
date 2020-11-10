import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.sparse import load_npz


data = np.concatenate([load_npz('data1.npz').toarray(), load_npz('data2.npz').toarray()]).astype(int)

num_class = 2

if not os.path.exists(f'{num_class}_kmeans.npy'):
    kmeans = KMeans(n_clusters=num_class, random_state=0).fit(data)
    label = kmeans.labels_
    np.save(f'{num_class}_kmeans.npy', label)
else:
    label = np.load(f'{num_class}_kmeans.npy')

if not os.path.exists(f'purchase{num_class}_train.npy'):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    np.save(f'purchase{num_class}_train.npy', {'X': X_train, 'y': y_train})
    np.save(f'purchase{num_class}_test.npy', {'X': X_test, 'y': y_test})
