import time
import cifar10
from sklearn import neighbors

# load
time_start = time.time()
path = 'data/cifar-10-batches-py'
label_name, train_data, train_label, test_data, test_label = cifar10.load(path)

# p=1: manhattan_distance(l1)
# p=2: euclidean_distance(l2)
p = 2
k = 20
knn = neighbors.KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree', p=p, n_jobs=-1)

# train
knn.fit(train_data, train_label)

# score
score = knn.score(test_data, test_label)

# show
time_end = time.time()
time_used = time_end - time_start

print('k: ', k)
print(['', 'manhattan_distance', 'euclidean_distance'][p])
print('total time: ', time_used)
print('Accuracy: ', score)