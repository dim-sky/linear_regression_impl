
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    m = 30
    mySeed = np.random.seed(42)

    X = np.random.uniform(0, 45, m) # Weekly hours of studying
    randomNoise = np.random.normal(0, 5, m)
    y = 2*X + 10 + randomNoise

    plt.title("Study - Score Corellation")
    plt.scatter(X,y)
    plt.xlabel("Weekly Hours of Studying")
    plt.ylabel("Final Grade")
    plt.show()

    X_normalised = (X - min(X)) / (max(X) - min(X))
    y_normalized = (y - min(y)) / (max(y) - min(y))

    ones = np.ones(m)
    X_biase = np.c_[X_normalised, ones]
    print(X_biase)


    # Model: Linear Regression (y = wX +b)
    w = 0.0
    b = 0.0
    theta = np.array([w, b])


    # Prediction
    def predict(X,theta):
        return np.dot(X,theta)
    

    # Loss
    def calculate_loss(y,X,theta):
        m = len(y)
        y_pred = predict(X, theta)
        loss = (1/m)*np.sum((y_pred - y)**2)
        return loss
    
    # Gradient
    def gradient(y,X,theta):
        y_pred = predict(X, theta)
        yPred_minus_y = y_pred - y
        theta_grad =  np.dot(X.transpose(),yPred_minus_y)
        return theta_grad
    

    # Gradient Descent
    def gradient_descent(y,X,theta,lr=0.01, epochs=100):
        temp = 0
        for i in range (epochs):
            index_array = np.arange(0, m)
            for j in range (m):
                y_j = y[j]
                X_j = X[j]
                theta_grad = gradient(y_j,X_j,theta)
                theta = theta - lr*theta_grad
                temp += 1
                print(f"weights updated -- {temp}")
        
            if (i % 20 == 0):
                print(f'epoch[{i}] ---- loss = {calculate_loss(y,X,theta)}')
        return theta
    
    final_theta = gradient_descent(y_normalized, X_biase, theta, lr=0.01, epochs=100)
    print(f"Final weight: {final_theta[1]}, Final bias: {final_theta[0]}")


    plt.title("Study - Score Corellation")
    plt.scatter(X_normalised, y_normalized, s=60)
    plt.plot(X_normalised, predict(X_biase, final_theta), color='red')
    plt.xlabel("Weekly Hours of Studying")
    plt.ylabel("Final Grade")
    plt.show()



    