import numpy as np

def generate_vector(dim):
    """Generates a random zero-mean vector."""
    return np.random.normal(0, 1 / np.sqrt(dim), dim)

def circular_convolution(x, y):
    """Binds two vectors using circular convolution via FFT."""
    return np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(y), n=len(x))

def circular_correlation(x, y):
    """Unbinds two vectors using circular correlation via FFT."""
    return np.fft.irfft(np.conj(np.fft.rfft(x)) * np.fft.rfft(y), n=len(x))

def cosine_similarity(x, y):
    """Calculates cosine similarity between two vectors."""
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

if __name__ == "__main__":
    np.random.seed(42)
    dim = 1000

    # Generate random vectors
    A = generate_vector(dim)
    B = generate_vector(dim)
    C = generate_vector(dim)

    # Bind A and B
    bound_AB = circular_convolution(A, B)

    # Bundle bound_AB with C
    memory = bound_AB + C

    # Unbind A to recover B
    recovered_B = circular_correlation(A, memory)

    # Check similarity
    sim_B = cosine_similarity(B, recovered_B)
    sim_C = cosine_similarity(C, recovered_B)

    print(f"Similarity with B (expected high): {sim_B:.4f}")
    print(f"Similarity with C (expected low): {sim_C:.4f}")

    if sim_B > 0.5 and sim_C < 0.2:
        print("Holographic Reduced Representation (HRR) operations verified successfully.")
        exit(0)
    else:
        print("HRR operations failed.")
        exit(1)
