import numpy as np

class SparseDistributedMemory:
    def __init__(self, address_size, data_size, hard_locations_count, activation_radius):
        """
        address_size: Length of address bit vectors
        data_size: Length of data bit vectors
        hard_locations_count: Number of physical storage locations
        activation_radius: Hamming distance threshold for activation
        """
        self.address_size = address_size
        self.data_size = data_size
        self.hard_locations_count = hard_locations_count
        self.activation_radius = activation_radius

        # Hard locations addresses (randomly initialized binary vectors)
        self.hard_addresses = np.random.randint(2, size=(hard_locations_count, address_size))

        # Counters for data (initialized to zero)
        self.counters = np.zeros((hard_locations_count, data_size), dtype=int)

    def _hamming_distance(self, a, b):
        return np.sum(a != b, axis=1)

    def write(self, address, data):
        # Find active hard locations
        distances = self._hamming_distance(self.hard_addresses, address)
        active_indices = np.where(distances <= self.activation_radius)[0]

        if len(active_indices) == 0:
            return False

        # Update counters (increment for 1, decrement for 0)
        update = np.where(data == 1, 1, -1)
        self.counters[active_indices] += update
        return True

    def read(self, address):
        # Find active hard locations
        distances = self._hamming_distance(self.hard_addresses, address)
        active_indices = np.where(distances <= self.activation_radius)[0]

        if len(active_indices) == 0:
            return np.zeros(self.data_size, dtype=int)

        # Sum counters across active locations
        sums = np.sum(self.counters[active_indices], axis=0)

        # Threshold at 0
        return np.where(sums > 0, 1, 0)

def test_sdm():
    print("Testing Sparse Distributed Memory...")
    sdm = SparseDistributedMemory(address_size=20, data_size=10, hard_locations_count=1000, activation_radius=8)

    address = np.random.randint(2, size=20)
    data = np.random.randint(2, size=10)

    sdm.write(address, data)
    read_data = sdm.read(address)

    if np.array_equal(data, read_data):
        print("Sparse Distributed Memory completed successfully.")
    else:
        print(f"SDM failed. Expected {data}, got {read_data}")

if __name__ == "__main__":
    test_sdm()
