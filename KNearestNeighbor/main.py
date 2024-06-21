import numpy as np

def euclidean_distance(x1,x2):
    distance = np.sqrt((np.sum(x1-x2)**2))
    return distance

class NearestNeighbor:
    def __init__(self):
        pass
    
    def fit(self,X,y):
        self.X_tr = X
        self.y_tr = y
        
    def predict(self,X):
        predictions = [self._predict(x) for x in X]
        return predictions
        
      
        
    def _predict(self,x):
        pass

