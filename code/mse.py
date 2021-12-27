import numpy as np

# Calculate Squared Errors
def mse(x_train, y_train, x_test, y_test, p=False):
    #print("within mse:", x_train.shape)

    x_1 = np.append(x_train, np.ones((x_train.shape[0],1)),1)
    xt = np.transpose(x_1)

    #print(xt)
    betas = np.matmul(np.matmul(np.linalg.inv(np.matmul(xt, x_1)), xt), y_train)

    x_1_test = np.append(x_test, np.ones((x_test.shape[0],1)),1)
    e = y_test.reshape(-1) - np.sum(x_1_test * np.transpose(betas), axis=1)
    
    return [sum(e*e),len(x_test)]
