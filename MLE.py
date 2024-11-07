"""
This script implements Maximum Likelihood Estimation (MLE) to estimate key parameters
(alpha and delta) of a logistic demand model using simulated data.

Core functions:
1. Defines the log-likelihood function, which measures the fit between the observed data and the logistic model.
2. Simulates demand over multiple time periods and products, based on price sensitivity and baseline demand parameters.
3. Uses numerical optimization to maximize the log-likelihood function and obtain the best-fitting parameter estimates.
4. Provides functionality for visualizing the optimization landscape to help interpret the results.

The script is designed to estimate the model parameters that best explain the simulated demand patterns.
"""

import pandas as pd
import numpy as np
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Partial derivative of the likelihood function with respect to alpha (α)
def dalpha(a, d, df):
    pderiv = -((df.p * np.exp(df.p * a)) * 
              (df.Q * np.exp(df.p * a) + np.exp(d) * df.Q - np.exp(d) * df.lam)) / \
              np.square(np.exp(df.p * a) + np.exp(d))
    return np.sum(pderiv)

# Partial derivative of the likelihood function with respect to delta (δ)
def ddelta(a, d, df):
    pderiv = (np.exp(a * df.p) * 
              ((df.Q - df.lam) * np.exp(d) + np.exp(a * df.p) * df.Q)) / \
              np.square(np.exp(d) + np.exp(a * df.p))
    return np.sum(pderiv)

# Log-likelihood function to be minimized
def loglikelihood(params, df):
    a, d = params
    q_fact = np.array([np.math.factorial(round(q)) for q in df.Q])  # Factorial of demand values
    loglik = (-df.lam * np.exp(d - a * df.p)) / (1 + np.exp(d - a * df.p)) + \
             df.Q * np.log(df.lam) + df.Q * (d - a * df.p) - \
             df.Q * np.log(1 + np.exp(d - a * df.p)) - np.log(q_fact)
    return -np.sum(loglik)

# Compute the gradient of the log-likelihood function
def comp_grad(params, df):
    a, d = params
    return np.array([dalpha(a, d, df), ddelta(a, d, df)])

# Objective function (sum of squared derivatives) for optimization
def objective(params, df):
    a, d = params
    return np.square(dalpha(a, d, df)) + np.square(ddelta(a, d, df))

# Simulate flight demand with given parameters
def simFlightDemand(J, T, delta, alpha, lambd):
    # Generate random prices for products
    p = np.random.normal(loc=50, size=J * T)
    df = pd.DataFrame({'p': p})
    
    # Calculate shares using the logistic model
    df["shares"] = np.exp(delta - alpha * df.p) / (1 + np.exp(delta - alpha * df.p))
    
    # Generate period and product indices
    df["t"] = np.repeat(np.arange(T), J)
    df["j"] = np.tile(np.arange(J), T)
    
    # Simulate Poisson-distributed demand
    demand_data = []
    arrivals = []
    for period in np.arange(T):
        pk = df.loc[df.t == period].shares.values
        arrival = np.random.poisson(lambd)
        demand_data.append([round(arrival * share) for share in pk])
        arrivals.append(arrival)
    
    # Create dataframes for demand and arrivals
    df_demand = pd.DataFrame(demand_data)
    df_demand["t"] = np.arange(T)
    df_arrivals = pd.DataFrame(arrivals, columns=["lam"])
    df_arrivals["t"] = np.arange(T)
    
    # Reshape demand data to long format
    df_demand = df_demand.melt(id_vars="t", var_name="j", value_name="Q")
    
    # Merge with original data and calculate mean arrival rate
    df = df.merge(df_demand, on=["t", "j"]).merge(df_arrivals, on="t")
    df['lam_hat'] = df['lam'].mean()
    return df

# Maximize the log-likelihood function using L-BFGS-B method
def maxLike(df):
    initial_guess = np.random.uniform(low=0, high=1, size=2)
    result = minimize(fun=loglikelihood, x0=initial_guess, args=(df,), 
                      method='L-BFGS-B', bounds=((0, 1), (0, 1)),
                      options={'gtol': 1e-15, 'ftol': 1e-15})
    return result

# Plot the objective function in 3D for visualization
def graph_objective(df):
    a_vals = np.linspace(0, 1, 100)
    d_vals = np.linspace(0, 1, 100)
    Z = np.zeros((len(a_vals), len(d_vals)))
    
    # Compute objective values for the grid
    for i in range(len(a_vals)):
        for j in range(len(d_vals)):
            Z[i, j] = objective([a_vals[i], d_vals[j]], df)
    
    # Plot the surface in 3D
    x, y = np.meshgrid(a_vals, d_vals)
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, Z, cmap='viridis', edgecolor='green')
    plt.show()

def main():
 # Timing the simulation process
    t0 = time.time()

    # Number of simulations, products, and periods
    N = 100
    J = 2
    T = 90

    # Parameters for demand simulation
    delta = 0.5
    alpha = 0.03
    lambd = 30

    # Simulate demand for N flights
    dfs = [simFlightDemand(J, T, delta, alpha, lambd).assign(sim=i) for i in range(N)]
    df = pd.concat(dfs, ignore_index=True)

    # Optimize multiple times to avoid local minima
    min_guess = np.array([1, 1])
    for _ in range(50):
        current_result = maxLike(df)
        if objective(current_result.x, df) < objective(min_guess, df):
            min_guess = current_result.x

    # Output the best result
    print(f"Optimized parameters: {min_guess}")

    t1 = time.time()
    print(f"Execution time: {t1 - t0} seconds")
    return min_guess

# MAIN FUNCTION
if __name__ == '__main__':
    main()
