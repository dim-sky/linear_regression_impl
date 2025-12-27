import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    # Data
    X = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], dtype=float)  # Hours of studying 
    y = np.array([16, 21, 29, 34, 41, 46, 53, 59, 65, 71, 77, 82, 89, 95, 97], dtype=float)  # Final grade

    # Data normalization (to avoid exploding gradient)
    # Min-Max Normalization 
    y = (y - min(y)) / (max(y) - min(y))
    X = (X - min(X)) / (max(X) - min(X))
    print(y)
    
    # X_norm_rounded = np.round(X_norm, 4)  # 4 δεκαδικά

    # Step 0: Visualize the data
    plt.scatter(X, y)
    plt.xlabel("Hours Of Studying")
    plt.ylabel("Final Grade")
    plt.title("Hours Of Studying - Final Grade Corellation")
    plt.show()


    # Step 1: Decide the model we are learning
    # For linear regression its a straight line y = wX + b
    w = 0.0
    b = 0.0


    # Step 2: What do we mean with predict? 
    def predict(X, w, b):
        return w*X + b
    

    # Step 3: Define Loss Function (for lr its the MSE)
    def calculate_loss(y,X,b,w):
        y_hat = predict(X,w,b)
        m = len(y)
        mse = (1/m) * np.sum((y - y_hat)**2)
        return mse


    # Step 4: Compute Gradient
    def calculate_gradient(y,X,w,b):
        m = len(y)
        idx = np.random.randint(0, m)  # τυχαίο δείγμα
        x_i = X[idx]
        y_i = y[idx]
        y_hat = predict(x_i,w,b)
        dw =  -(y_i - y_hat)*x_i
        db =  -(y_i - y_hat)
        return dw, db
    
    # Step 5: Update Paremeters
    def gradient_descent(y,X,w,b, lr=0.01, epochs=1000):
        for i in range(epochs):
            dw, db = calculate_gradient(y,X,w,b)
            w = w - lr*dw
            b = b - lr*db

            if (i % 100 == 0):
                print(f'Epoch {i}, Loss: {calculate_loss(y, X, w, b)}')

        return w, b
        

        # TRAIN MODEL AND VISUALISE 
    w_final, b_final = gradient_descent(X, y, w, b, lr=0.001, epochs=1000)
    print(f"Final weight: {w_final}, Final bias: {b_final}")


    plt.scatter(X, y, s=60)  # your data points
    plt.plot(X, predict(X, w_final, b_final), color='red')  # learned line
    plt.xlabel("Square Meters")
    plt.ylabel("Price")
    plt.show()


    ## Ovservations 

    # I see that with the specific data i need a lot of epochs for a specific lr to actually see the line "fitting the data"
    # Note that if you run the model with small lr (0.001) and a "not enough"(small is relative) number of epochs (1000) the line will be a really bad predictor