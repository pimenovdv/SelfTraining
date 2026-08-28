"""
Ant Colony Optimization (ACO) Component

This script evaluates an Ant Colony Optimization component mathematically in pure NumPy,
testing its ability to find the shortest path in a Traveling Salesperson Problem (TSP)
using artificial ants and pheromone updates.
"""
import numpy as np

def generate_tsp_distances(num_cities, seed=42):
    np.random.seed(seed)
    # Generate random coordinates for cities
    coords = np.random.rand(num_cities, 2)
    # Compute distance matrix
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    return dist_matrix

def ant_colony_optimization(dist_matrix, num_ants, num_iterations, alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100.0, seed=42):
    np.random.seed(seed)
    num_cities = dist_matrix.shape[0]
    pheromone = np.ones((num_cities, num_cities))

    # Heuristic information is the inverse of distance
    with np.errstate(divide='ignore'):
        heuristic = 1.0 / dist_matrix
    np.fill_diagonal(heuristic, 0.0)

    best_path = None
    best_cost = float('inf')

    for iteration in range(num_iterations):
        all_paths = []
        all_costs = []

        for ant in range(num_ants):
            path = []
            current_city = np.random.randint(num_cities)
            path.append(current_city)
            visited = set([current_city])

            while len(visited) < num_cities:
                # Calculate probabilities
                probs = np.zeros(num_cities)
                for next_city in range(num_cities):
                    if next_city not in visited:
                        probs[next_city] = (pheromone[current_city, next_city] ** alpha) * (heuristic[current_city, next_city] ** beta)

                probs_sum = np.sum(probs)
                if probs_sum == 0:
                    unvisited = list(set(range(num_cities)) - visited)
                    next_city = np.random.choice(unvisited)
                else:
                    probs = probs / probs_sum
                    next_city = np.random.choice(np.arange(num_cities), p=probs)

                path.append(next_city)
                visited.add(next_city)
                current_city = next_city

            cost = 0
            for i in range(num_cities - 1):
                cost += dist_matrix[path[i], path[i+1]]
            cost += dist_matrix[path[-1], path[0]]

            all_paths.append(path)
            all_costs.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_path = path

        pheromone = (1.0 - evaporation_rate) * pheromone
        for path, cost in zip(all_paths, all_costs):
            pheromone_deposit = Q / cost
            for i in range(num_cities - 1):
                pheromone[path[i], path[i+1]] += pheromone_deposit
                pheromone[path[i+1], path[i]] += pheromone_deposit
            pheromone[path[-1], path[0]] += pheromone_deposit
            pheromone[path[0], path[-1]] += pheromone_deposit

    return best_path, best_cost

if __name__ == "__main__":
    print("Testing Ant Colony Optimization (ACO) Component...")
    num_cities = 15
    dist_matrix = generate_tsp_distances(num_cities)

    num_ants = 20
    num_iterations = 50

    print(f"Number of cities: {num_cities}")
    print(f"Number of ants: {num_ants}")
    print(f"Number of iterations: {num_iterations}")

    best_path, best_cost = ant_colony_optimization(dist_matrix, num_ants, num_iterations)

    print(f"Best path found: {best_path}")
    print(f"Best cost: {best_cost:.4f}")
    print("ACO component trained and verified successfully.")
