from logistic_regression import LogisticRegression as lr
import numpy as np
import matplotlib.pyplot as plt



def main():

    np.random.seed(42)  # Set a random seed for reproducibility

    #generate some random data with at least 2 features and 300 samples
    #train a logistic regression model on the data
    X = np.random.rand(300, 2)
    y = np.random.randint(0, 2, 300)

    """ Train the models using different learning rates and plot the loss history for each model. """
    epoches = [100,1000,4000,3000]
    plt.figure(figsize=(10, 6))
    for epoch in epoches:
        model = lr(learning_rate=1, num_iterations=epoch)
        model.fit(X, y)
        plt.plot(model.loss_history, label=f"Epochs: {epoch}")

    plt.title("Loss History")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

main()