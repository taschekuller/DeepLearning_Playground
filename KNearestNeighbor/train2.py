from main import KNN
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os

def load_cifar10_batch(file_path):
    print(f"Loading file: {file_path}")  # Debug print
    with open(file_path, 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
    data = batch['data']
    labels = batch['labels']
    return np.array(data), np.array(labels)

def load_cifar10_data(data_dir):
    X_train = []
    y_train = []
    
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f'data_batch_{i}')
        data_batch, labels_batch = load_cifar10_batch(batch_path)
        X_train.append(data_batch)
        y_train.append(labels_batch)
    
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    
    test_batch_path = os.path.join(data_dir, 'test_batch')
    X_test, y_test = load_cifar10_batch(test_batch_path)
    
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.astype(np.float64))
    X_test = scaler.transform(X_test.astype(np.float64))
    return X_train, X_test

def main():
    # Print the current working directory for debugging
    print("Current working directory:", os.getcwd())
    
    # Use an absolute path to the inputs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'inputs')
    
    X_train, y_train, X_test, y_test = load_cifar10_data(data_dir)
    
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    
    X_train, X_test = preprocess_data(X_train, X_test)
    
    knn = KNN(k=3)  # Initialize KNN with k=5
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    print("y_pred shape:", y_pred.shape)
    print("y_test shape:", y_test.shape)
    
    # Check for any discrepancies in y_pred
    print("y_pred first 10 predictions:", y_pred[:10])
    print("y_test first 10 true labels:", y_test[:10])
    
    accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
    print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()
