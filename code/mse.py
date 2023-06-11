import numpy as np

# Calculate Squared Errors
def mse(x_train, y_train, x_test, y_test, p=False):

    x_1 = np.append(x_train, np.ones((x_train.shape[0],1)),1)
    xt = np.transpose(x_1)

    betas = np.matmul(np.matmul(np.linalg.inv(np.matmul(xt, x_1)), xt), y_train)

    x_1_test = np.append(x_test, np.ones((x_test.shape[0],1)),1)
    
    if p:
        y_pred = get_predicted_values(x_1_test, betas, x_train.shape[1])
    else:
        y_pred = np.sum(x_1_test * np.transpose(betas), axis=1)
        
    e = y_test.reshape(-1) - y_pred
    
    return [sum(e*e),len(x_test)]


def get_predicted_values(x_1_test, betas, m):
    '''
    In the case of testing prediction performance of AR models, we want the predicted values to be generated iteratively.
    In other words, the next predicted value results from using the previous predicted value(s), not the previous observed value(s).
    '''
    y_pred = x_1_test[0][:m]
    
    for _ in x_1_test:
        
        x = np.append(y_pred[:m], np.array([1]), 0)
        pred = np.sum(x * betas)
        y_pred = np.insert(y_pred, 0, pred)
    
    return np.flip(y_pred[:-m])
