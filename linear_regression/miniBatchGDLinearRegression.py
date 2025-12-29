
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Data Preparation
    m = 50
    batch_size = 10
    mySeed = np.random.seed(44)
    X = np.random.uniform(30, 200, m) # House Size

    noise = np.random.normal(0, 15000, m)
    y= 1500*X + noise # Price in dollars

    # Normalization
    X = (X - min(X)) / (max(X) - min(X))
    y = (y - min(y)) / (max(y) - min(y))

    plt.title("Size - Cost Correlation")
    plt.scatter(X,y)
    plt.xlabel("House Size")
    plt.ylabel("Cost in $")
    plt.show()


    # Model that we are learning. Linear Regression (y = wX + b)
    w = 0.0
    b = 0.0


    # Predict 
    def predict(X,w,b):
        return w*X + b
        
    # Loss 
    def calculate_loss(y,X,w,b):
        m = len(y)
        y_hat = predict(X,w,b)
        loss = (1/m)*np.sum((y_hat - y)**2) # mse
        return loss
    
    # Gradient
    def gradient(y,X,w,b):
        m = len(X)
        y_hat = predict(X,w,b)
        dw = (2/m)*np.sum((y_hat - y)*X)
        db = (2/m)*np.sum((y_hat - y))
        return dw, db


    def gradient_descent(y,X,w,b,batch_size=10, lr=0.01,epochs=10):
        m = len(y)
        number_of_batches = m / batch_size
        for i in range(epochs):
            X_in_batches = np.array_split(X, number_of_batches)
            y_in_batches = np.array_split(y, number_of_batches)
            # print(X_in_batches)
            # print(y_in_batches)
            for j in range(len(X_in_batches)):
                dw,db = gradient(y_in_batches[j],X_in_batches[j], w, b)
                w = w - lr*dw
                b = b - lr*db

            if (i % 10 == 0):
                print(f'Epoch {i}, Loss: {calculate_loss(y, X, w, b)}')

        return w,b


    w_final, b_final = gradient_descent(y, X, w, b,batch_size=10,  lr=0.001, epochs=1500)
    
    # Make prediction after model is trained
    # normalized_grade = (16 - min(temp_X)) / (max(temp_X) - min(temp_X))  # normalized x
    # normalized_res_pred = predict(normalized_grade, w_final, b_final)  # normalized y_hat
    # res_pred = normalized_res_pred*(max(temp_y) - min(temp_y)) + min(temp_y)  # de-normalize y_hat
    # print('if you study 17 hours expect this grade : ' + str(res_pred))
    # print(f"Final weight: {w_final}, Final bias: {b_final}")


    plt.scatter(X, y, s=60)
    plt.plot(X, predict(X, w_final, b_final), color='red')  # learned line
    plt.xlabel("Square Meters")
    plt.ylabel("Price")
    plt.show()



    

