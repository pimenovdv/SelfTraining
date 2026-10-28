import numpy as np
import time

def rosenbrock(x):
    """Rosenbrock function: f(x, y) = (a - x)^2 + b(y - x^2)^2, with a=1, b=100"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def nelder_mead(f, x_start, step=0.1, no_improve_thr=10e-6, no_improv_break=10, max_iter=1000,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    """
    Nelder-Mead optimization algorithm.
    """
    dim = len(x_start)
    res = [[x_start, f(x_start)]]

    for i in range(dim):
        x = np.copy(x_start)
        x[i] = x[i] + step
        score = f(x)
        res.append([x, score])

    iters = 0
    while 1:
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        if res[-1][1] - res[0][1] < no_improve_thr:
            return res[0]

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            res[-1] = [xr, rscore]
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = f(xe)
            if escore < rscore:
                res[-1] = [xe, escore]
                continue
            else:
                res[-1] = [xr, rscore]
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = f(xc)
        if cscore < res[-1][1]:
            res[-1] = [xc, cscore]
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = f(redx)
            nres.append([redx, score])
        res = nres

def main():
    print("Evaluating Nelder-Mead Optimization Component...")
    x_start = np.array([-1.0, 2.0])

    start_time = time.time()
    best = nelder_mead(rosenbrock, x_start, max_iter=2000)
    execution_time = time.time() - start_time

    x_opt, f_opt = best
    print(f"Optimal parameters: {x_opt}")
    print(f"Minimum value: {f_opt:.6f}")
    print(f"Execution time: {execution_time:.4f} seconds")

    assert f_opt < 1e-4, "Nelder-Mead failed to converge to the minimum of the Rosenbrock function."
    print("Nelder-Mead optimization successful!")

if __name__ == "__main__":
    main()
