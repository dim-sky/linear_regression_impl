import matplotlib.pyplot as plt
import numpy as np

### In this example we will implement a linear regression model that uses (batch) Gradient Descent as the optimization algorithm

if __name__ == "__main__":

    # Data Preparation
    m = 300
    mySeed = np.random.seed(44)
    X = np.random.uniform(25, 250, m) # House Size

    noise = np.random.normal(0, 30000, m)
    y= 1500*X + noise # Price in dollars

    # Normilize the data. (if i dont do that i will experience exploding gradient. Run without it to understand)
    X = (X - min(X)) / (max(X) - min(X))  # normalize
    y = (y - min(y)) / (max(y) - min(y))  # normalize

    # By looking at the plotted data we can see that they are lineary correlated 
    plt.scatter(X,y)
    plt.title("Housing Price According to size of houses")
    plt.xlabel("Square meters")
    plt.ylabel("Price in $")
    plt.show()


    # STEP 1: DECIDE THE MODEL WE ARE LEARNING
    # !!!! We must first define what kind of function we are trying to learn. !!!!
    # For simple linear regression: its a straight line  (y = wX + b)
    
    # We will initialize weights and biases to zero (the simplest strategy)
    w = 0.0
    b = 0.0


    # STEP 2: WE MUST DECIDE WHAT DO WE MEAN WITH PREDICTION
    def predict(X, w, b):
        return w*X + b


    # Step 3: DEFINE THE LOSS FUNCTION
    def calculate_loss(X,y,w,b):
        m = len(y)
        y_pred = predict(X, w, b)
        loss = (1/m) * np.sum((y_pred - y)**2) # MSE 
        return loss
    

    # Step 4: COMPUTE THE GRADIENTS.
    # WHICH GRADIENTS??? ----> The gradient of the loss function with respect to w and with respect to b
    def gradient(y, w, X, b):
        m = len(y)
        y_pred = predict(X,w,b)
        dw = (2/m) * np.sum((y_pred - y) * X)
        db = (2/m) * np.sum(y_pred - y)
        # print(dw)
        return dw, db
    

    # STEP 5: UPDATE PARAMETERS
    def gradient_descent(X,y,w,b, lr=0.01, epochs=1000):
        for i in range(epochs):
            dw, db = gradient(y,w,X,b)
            w = w - lr*dw
            b = b - lr*db

            if (i % 100 == 0):
                print(f'Epoch {i}, Loss: {calculate_loss(X, y, w, b)}')
        return w, b
    

    # TRAIN MODEL AND VISUALISE 
    print(gradient(y, w, X, b))
    w_final, b_final = gradient_descent(X, y, w, b, lr=0.1, epochs=1000)
    print(f"Final weight: {w_final}, Final bias: {b_final}")


    plt.scatter(X, y, s=60)
    plt.plot(X, predict(X, w_final, b_final), color='red')  # learned line
    plt.xlabel("Square Meters")
    plt.ylabel("Price")
    plt.show()

