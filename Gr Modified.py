import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_predicted):
    cost = np.sum((y_true - y_predicted) ** 2) / len(y_true)
    return cost

def gradient_descent(x, y, iterations=1000, learning_rate=0.0001, stopping_threshold=1e-6):
    current_weight = 0.1
    current_bias = 0.01
    n = float(len(x))

    costs = []
    weights = []
    previous_cost = None

    for i in range(iterations):
        y_predicted = (current_weight * x) + current_bias
        current_cost = mean_squared_error(y, y_predicted)
        if previous_cost and abs(previous_cost - current_cost) <= stopping_threshold:
            break
        previous_cost = current_cost
        costs.append(current_cost)
        weights.append(current_weight)

        weight_derivative = -(2/n) * np.sum(x * (y - y_predicted))
        bias_derivative = -(2/n) * np.sum(y - y_predicted)

        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)

        print(f"Iteration {i + 1}: Cost {current_cost}, Weight {current_weight}, Bias {current_bias}")

    plt.figure(figsize=(8, 6))
    plt.plot(weights, costs)
    plt.scatter(weights, costs, marker='o', color='green')
    plt.title("Cost vs Weights")
    plt.ylabel("Cost")
    plt.xlabel("Weight")
    plt.show()

    return current_weight, current_bias

def main():
    x = np.array([52.50234527, 63.42680403, 81.53035803, 47.47563963, 89.81320787,
                  55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
                  45.41973014, 54.35163488, 44.1640495, 58.16847072, 56.72720806,
                  48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
    y = np.array([41.70700585, 78.77759598, 82.5623823, 91.54663223, 77.23092513,
                  78.21151827, 79.64197305, 59.17148932, 75.3312423, 71.30087989,
                  55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
                  60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])

    estimated_weight, estimated_bias = gradient_descent(x, y, iterations=2000)
    print(f"Estimated Weight: {estimated_weight}\nEstimated Bias: {estimated_bias}")

    y_pred = estimated_weight * x + estimated_bias

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, marker='o', color='red')
    plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color='blue', markerfacecolor='red', markersize=10, linestyle='dashed')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

if __name__ == "__main__":
    main()
