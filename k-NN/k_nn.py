import numpy as np

def preprocessing(data):
    new_data = []
    for index,d in enumerate(data):
        for c in d:
            new_data.append([c[0],c[1],index])
    return np.array(new_data)

class K_NN:
    def __init__(self, k):
        """
        :param k: number of nearest neighbours
        """
        self.k = k

    def fit(self, data):
        """
        :param data: 3D array, where data[i, j] is i-th classes j-th point (vector: D dimenstions)
        """
        
        self.data = preprocessing(data)
        
        return self.data
    
    def predict(self, data):
        """
        :param data: 2D array of floats N points each D dimensions
        :return: array of integers
        """
<<<<<<< HEAD
        predictions = []
        if data.shape == (data.shape[0],):
            dist = []
            for c in self.data:
                dist.append([np.linalg.norm(data - c[:2]), c[2]])
            dist.sort()
            dist = np.array(dist)
            pr = dist[:int(self.k)][:, 1]
            unique_elements, counts_elements = np.unique(pr, return_counts=True)

            return unique_elements[np.argmax(counts_elements)]
        else:
            for d in data:
                dist = []
                for c in self.data:
                    dist.append([np.linalg.norm(d-c[:2]), c[2]])
                dist.sort()
                dist = np.array(dist)
                pr = dist[:int(self.k)][:, 1]
                unique_elements, counts_elements = np.unique(pr, return_counts=True)

                predictions.append(unique_elements[np.argmax(counts_elements)])
            
            return np.array(predictions)
        
=======
        data = np.array(data)
        shp = data.shape
        if len(data.shape) == 1:
            data = data.reshape([1] + list(data.shape))
        # TODO: predict
        prediction = np.array([0])
        return prediction.reshape(shp[:-1])
>>>>>>> e8a0bc6ebf161be007748b10a7a530831fd9c8b6
