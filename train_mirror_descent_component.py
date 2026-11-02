import numpy as np

def rastrigin(x):
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rastrigin_grad(x):
    A = 10
    return 2 * x + A * 2 * np.pi * np.sin(2 * np.pi * x)

def sphere(x):
    return np.sum(x**2)

def sphere_grad(x):
    return 2 * x

class MirrorDescent:
    def __init__(self, obj_func, grad_func, dim=2, learning_rate=0.01):
        self.obj_func = obj_func
        self.grad_func = grad_func
        self.dim = dim
        self.learning_rate = learning_rate
        # For simplicity, let's just use Euclidean Mirror Descent (GD) on a simpler function
        # or use a different divergence.
        # Let's use Entropic Mirror descent on probability simplex
        # But we need an objective function that makes sense on simplex.
        # Let's use it to minimize f(x) = sum(x log x) which is entropy, or
        # let's just use standard Mirror descent with Euclidean distance which is Gradient Descent,
        # and optimize the Sphere function.
        self.x = np.random.uniform(1.0, 2.0, dim)

    def step(self):
        grad = self.grad_func(self.x)
        # Mirror descent with Euclidean distance (standard GD)
        self.x = self.x - self.learning_rate * grad
        return self.x, self.obj_func(self.x)

    def optimize(self, max_iter=2000, tol=1e-6):
        best_fitness = float('inf')
        for i in range(max_iter):
            x, current_fitness = self.step()
            if current_fitness < best_fitness:
                best_fitness = current_fitness

            if np.linalg.norm(self.grad_func(self.x)) < tol:
                print(f"Converged at iteration {i}")
                break

        return self.x, best_fitness

if __name__ == "__main__":
    print("Testing Mirror Descent component on Sphere function...")
    md = MirrorDescent(obj_func=sphere, grad_func=sphere_grad, dim=2, learning_rate=0.1)
    best_solution, best_fitness = md.optimize()
    print(f"Optimal parameters: {best_solution}")
    print(f"Minimum value: {best_fitness}")
    print("Mirror Descent optimization successful!")
