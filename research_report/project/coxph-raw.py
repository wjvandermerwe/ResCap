import numpy as np
from sklearn.linear_model import Lasso

class CoxPHModel:
    def __init__(self, lr=0.01, iterations=1000, lasso_penalty=0.1):
        self.lr = lr
        self.iterations = iterations
        self.lasso_penalty = lasso_penalty
        self.beta = None

    def fit(self, X, T, E):
        self.beta = np.zeros(X.shape[1])

        for _ in range(self.iterations):
            linear_predictor = np.dot(X, self.beta)
            exp_lp = np.exp(linear_predictor)
            risk_set = np.cumsum(exp_lp[::-1])[::-1]

            gradient = np.dot(E, X) - np.dot((exp_lp / risk_set), X)
            self.beta += self.lr * gradient

        return self.beta

    def fit_lasso(self, X, T, E):
        self.beta = np.zeros(X.shape[1])

        for _ in range(self.iterations):
            linear_predictor = np.dot(X, self.beta)
            exp_lp = np.exp(linear_predictor)
            risk_set = np.cumsum(exp_lp[::-1])[::-1]

            gradient = np.dot(E, X) - np.dot((exp_lp / risk_set), X)
            lasso_term = self.lasso_penalty * np.sign(self.beta)
            self.beta += self.lr * (gradient - lasso_term)

        return self.beta

    def fit_lasso_sklearn(self, X, T, E):
        linear_predictor = np.dot(X, self.beta)
        exp_lp = np.exp(linear_predictor)
        risk_set = np.cumsum(exp_lp[::-1])[::-1]

        gradient = np.dot(E, X) - np.dot((exp_lp / risk_set), X)

        lasso = Lasso(alpha=self.lasso_penalty)
        lasso.fit(X, gradient)
        self.beta = lasso.coef_

        return self.beta

    def predict(self, X):
        return np.dot(X, self.beta)

# Example data
X = np.array([[0.5, 1.2], [1.3, 0.7], [2.1, 1.9]])
T = np.array([5, 10, 15])  # Survival times
E = np.array([1, 0, 1])    # Event occurred or censored

# Create and fit the CoxPHModel
cox_model = CoxPHModel(lr=0.01, iterations=1000)
beta = cox_model.fit(X, T, E)
print("Estimated coefficients without Lasso:", beta)

# Fit the Lasso-regularized CoxPHModel
cox_lasso_model = CoxPHModel(lr=0.01, iterations=1000, lasso_penalty=0.1)
beta_lasso = cox_lasso_model.fit_lasso(X, T, E)
print("Estimated coefficients with Lasso:", beta_lasso)