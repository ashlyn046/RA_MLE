# MLE Logistic Demand Estimator

This project implements a simple Maximum Likelihood Estimation (MLE) model to estimate parameters of a logistic demand function based on simulated data.

## Overview

This script provides an MLE approach to estimate the parameters **alpha** and **delta**, which represent price sensitivity and baseline demand, in a logistic demand model. The process uses simulated demand data and numerical optimization to achieve best-fit parameter estimates.

### Key Features

- **Log-Likelihood Calculation**: Computes the likelihood of observing the given data under different parameter values.
- **Simulation**: Generates demand across multiple periods and products based on specified parameters.
- **Optimization**: Uses the `L-BFGS-B` algorithm to maximize the log-likelihood function, finding the optimal parameters.
- **Visualization**: Allows for 3D visualization of the objective function landscape to interpret the optimization results.

## Structure and Functions

- `dalpha` and `ddelta`: Compute partial derivatives of the likelihood with respect to **alpha** and **delta**.
- `loglikelihood`: Defines the log-likelihood function to be maximized.
- `comp_grad`: Computes the gradient of the log-likelihood function.
- `objective`: Calculates the squared derivatives for optimization.
- `simFlightDemand`: Simulates demand for flights based on the logistic model.
- `maxLike`: Uses numerical optimization to maximize the log-likelihood.
- `graph_objective`: Visualizes the objective function landscape in 3D.
