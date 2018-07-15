import numpy as np

class KMeans:
    def __init__(self, k, max_iter = 30):
        self.k = k
        self.max_iter = max_iter
        
    def fit(self, data):
        """
        :param data: numpy array of shape (..., dims)
        """
        self.dim = data.shape[-1]
        self._initialize_means(data)
        
        for i in range(self.max_iter): 
            
            self.clusters = {}

            for d in data:
                distances = np.array([np.linalg.norm(d-c) for c in self.centers])
                cluster_index = np.argmin(distances)
                self.clusters.setdefault(cluster_index,[])
                self.clusters[cluster_index].append(d)
            
            self.centers = []
            
            for cluster in self.clusters:
                center = np.mean(np.array(self.clusters[cluster]),axis=0)
                self.centers.append(center)
              
        return self.clusters
            
    def _initialize_means(self, data):
        indices = np.random.choice(data.shape[0], self.k, replace=False)
        self.centers = np.array([data[i] for i in indices])
        return self.centers

    def predict(self, data):
        """
        :param data: numpy array of shape ( ..., dims)
        :return: labels of each datapoint and it's mean
                 0 <= labels[i] <= k - 1
        """
        labels = []
        means = []
        
        for d in data:
            distances = np.array([np.linalg.norm(d-c) for c in self.centers])
            cluster_index = np.argmin(distances)
            labels.append(cluster_index)
            means.append(self.centers[cluster_index])
        
        return labels, means


class KMeansPlusPlus(KMeans):
    def _initialize_means(self, data):
        indices = list(range(data.shape[0]))
        index = np.random.randint(data.shape[0])
        indices.remove(index)
        centers = []
        centers.append(data[index])
         
        for k in range(self.k-1):
            distances = []
            for i in indices:
                distances_d = np.array([np.linalg.norm(data[i]-c) for c in centers])
                distances.append(np.min(distances_d))
            sum_dist = np.sum(distances)
            probs = distances / sum_dist
            
            index = np.random.choice(indices, 1, replace=False, p=probs)
            
            centers.append(data[index])
            indices.remove(index)
            
        self.centers = centers
        return self.centers
