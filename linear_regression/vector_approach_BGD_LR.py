
import numpy as np
import matplotlib.pyplot as plt


# In this example i will reacreate the "classic Batch Gradient Descent" optimization algorithm but i will use the vector approach
# Where theta is a vector that contains w and b. (also i ll follow the approac where i add ones to the X verctor)  


if __name__ == "__main__" :

    # Data Preparation
    m = 300
    mySeed = np.random.seed(44)
    X = np.random.uniform(25, 250, m) # House Size

    noise = np.random.normal(0, 30000, m)
    y= 1500*X + noise # Price in dollars

    X_normalized = (X - min(X))/(max(X) - min(X))
    y_normalized = (y - min(y))/(max(y) - min(y))

    plt.title('Housing Size-Price')
    plt.scatter(X_normalized,y_normalized)
    plt.xlabel("House Size")
    plt.ylabel("price")
    plt.show()


    # Add a column of ones at X 
    m = X.shape[0]
    X_with_ones_normalized = np.c_[np.ones(m), X_normalized]  # shape: (m, n+1)
    #print(X_with_ones_normalized)

    # temp = np.array([[1, 1, 1], [2, 3, 4]]).T
    # y = np.array([21, 32, 40])

    # Model Decision: Linear Regression (y = wX + b)
    w = 0.0
    b = 0.0
    theta = np.array([b,w]) # Because the ones in X are in the first column thats why b goes first -at the top of the column vector- 
    #print(theta)


    # Predict
    def predict(X,theta):
        return np.dot(X,theta)
    

    # Calculate Loss
    def calculate_Loss(y, X, theta):
        # MSE = 1/m * Σ(y_hat - y)^2
        m = len(y)
        y_hat = predict(X, theta)
        loss = (1/m)*np.sum((y_hat - y)**2)
        return loss
    
    
    # Calculate Gradient
    def calculate_gradient(y, X, theta):
        m = len(y)
        y_hat = predict(X, theta)
        X_trabsposed = X.transpose()
        grad_theta = (2/m) * np.dot(X_trabsposed,(y_hat - y))
        return grad_theta
    
    
    # Gradient Descent
    def gradient_descent(y,X,theta,lr=0.001,epochs=100):
        for i in range(epochs):
            grad_theta = calculate_gradient(y,X, theta) # [grad_b, grad_w]
            theta = theta - lr*grad_theta

            if (i % 10 == 0):
                print(f'Epoch {i}, Loss: {calculate_Loss(y, X, theta)}')
        
        return theta
 
    final_theta = gradient_descent(y_normalized, X_with_ones_normalized, theta, lr=0.1, epochs=100)
    print(f"Final weight: {final_theta[1]}, Final bias: {final_theta[0]}")


    plt.scatter(X_normalized, y_normalized, s=60)
    plt.plot(X_normalized, predict(X_with_ones_normalized, final_theta), color='red')
    plt.xlabel("House Size (Normalized)")
    plt.ylabel("Price (Normalized)")
    plt.title("Linear Regression Fit")
    plt.show()
