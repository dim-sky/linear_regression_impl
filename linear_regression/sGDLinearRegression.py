import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    # Data
    X = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], dtype=float)  # Hours of studying 
    y = np.array([16, 21, 29, 34, 41, 46, 53, 59, 65, 71, 77, 82, 89, 95, 97], dtype=float)  # Final grade

    # Data normalization (to avoid exploding gradient)
    # Min-Max Normalization  
    temp_y = y  # i have overriden X and y with the normalized versions so i am using this to make predictions at the end
    temp_X = X  # i have overriden X and y with the normalized versions so i am using this to make predictions at the end
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
    # For linear regression its a straight line y = wX + bres_pred
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
        y_hat = predict(X,w,b)
        dw =  (y_hat - y)*X
        db =  (y_hat - y)
        return dw, db
    
    # Step 5: Update Paremeters
    def gradient_descent(y,X,w,b, lr=0.01, epochs=1000):
        m = len(y)
        for i in range(epochs):
            index_array = np.arange(0, m)
            np.random.shuffle(index_array)

            # print(index_array)
            for j in range(len(index_array)):
                dw, db = calculate_gradient(y[j],X[j],w,b)
                w = w - lr*dw
                b = b - lr*db
                # print(f'iteration[{j}] --- new weight {w} ---- new bias {b}')

            if (i % 10 == 0):
                print(f'Epoch {i}, Loss: {calculate_loss(y, X, w, b)}')

        return w, b
        

        # TRAIN MODEL AND VISUALISE 
    w_final, b_final = gradient_descent(X, y, w, b, lr=0.001, epochs=2000)
    
    # Make prediction after model is trained
    normalized_grade = (16 - min(temp_X)) / (max(temp_X) - min(temp_X))  # normalized x
    normalized_res_pred = predict(normalized_grade, w_final, b_final)  # normalized y_hat
    res_pred = normalized_res_pred*(max(temp_y) - min(temp_y)) + min(temp_y)  # de-normalize y_hat
    print('if you study 17 hours expect this grade : ' + str(res_pred))
    print(f"Final weight: {w_final}, Final bias: {b_final}")


    plt.scatter(X, y, s=60)
    plt.plot(X, predict(X, w_final, b_final), color='red')  # learned line
    plt.xlabel("Hours Of Studying")
    plt.ylabel("Final Grade")
    plt.title("Hours Of Studying - Final Grade Corellation")
    plt.show()

