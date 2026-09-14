import numpy as np

def stdp_weight_update(delta_t, A_plus=1.0, A_minus=1.0, tau_plus=10.0, tau_minus=10.0):
    """
    Calculate STDP weight update based on spike timing difference.
    delta_t = t_post - t_pre
    """
    if delta_t > 0:
        return A_plus * np.exp(-delta_t / tau_plus)
    elif delta_t < 0:
        return -A_minus * np.exp(delta_t / tau_minus)
    else:
        return 0.0

def test_component():
    print("Testing STDP Component...")
    delta_ts = np.array([-20, -10, -5, -1, 1, 5, 10, 20])

    print("Delta T (ms) | Weight Change")
    print("----------------------------")
    for dt in delta_ts:
        dw = stdp_weight_update(dt)
        print(f"{dt:10.1f} | {dw:10.4f}")
        if dt > 0:
            assert dw > 0
        elif dt < 0:
            assert dw < 0

    print("STDP weight updates computed and verified successfully.")

if __name__ == "__main__":
    test_component()
