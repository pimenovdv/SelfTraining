import numpy as np
import argparse

def srelu(x, t_r=1.0, a_r=0.1, t_l=-1.0, a_l=0.1):
    return np.piecewise(x,
                        [x >= t_r, (x < t_r) & (x > t_l), x <= t_l],
                        [lambda x: t_r + a_r * (x - t_r),
                         lambda x: x,
                         lambda x: t_l + a_l * (x - t_l)])

def srelu_derivative(x, t_r=1.0, a_r=0.1, t_l=-1.0, a_l=0.1):
    return np.piecewise(x,
                        [x >= t_r, (x < t_r) & (x > t_l), x <= t_l],
                        [lambda x: a_r,
                         lambda x: 1.0,
                         lambda x: a_l])

def test_srelu():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    y = srelu(x)
    dy = srelu_derivative(x)

    print("SReLU Activation test:")
    print("x:", x)
    print("srelu(x):", y)
    print("srelu_derivative(x):", dy)

    assert np.allclose(y, [-1.1, -0.5, 0.0, 0.5, 1.1])
    assert np.allclose(dy, [0.1, 1.0, 1.0, 1.0, 0.1])
    print("SReLU Component tests passed!")

def main():
    parser = argparse.ArgumentParser(description="Test SReLU Activation Component.")
    args = parser.parse_args()

    test_srelu()

if __name__ == "__main__":
    main()
