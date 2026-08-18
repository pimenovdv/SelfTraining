import numpy as np

class RingAllReduce:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def all_reduce(self, node_data):
        """
        Mathematically simulates Ring All-Reduce.
        node_data: list of numpy arrays, length num_nodes
        """
        assert len(node_data) == self.num_nodes
        data_size = len(node_data[0])
        chunk_size = data_size // self.num_nodes

        # Scatter-Reduce
        for step in range(self.num_nodes - 1):
            for i in range(self.num_nodes):
                send_chunk_idx = (i - step) % self.num_nodes
                recv_chunk_idx = send_chunk_idx
                recv_node = (i + 1) % self.num_nodes

                start_idx = send_chunk_idx * chunk_size
                end_idx = start_idx + chunk_size

                node_data[recv_node][start_idx:end_idx] += node_data[i][start_idx:end_idx]

        # All-Gather
        for step in range(self.num_nodes - 1):
            for i in range(self.num_nodes):
                send_chunk_idx = (i + 1 - step) % self.num_nodes
                recv_chunk_idx = send_chunk_idx
                recv_node = (i + 1) % self.num_nodes

                start_idx = send_chunk_idx * chunk_size
                end_idx = start_idx + chunk_size

                node_data[recv_node][start_idx:end_idx] = node_data[i][start_idx:end_idx]

        return node_data

if __name__ == "__main__":
    np.random.seed(42)
    num_nodes = 4
    data_size = 12
    # Ensure data_size is divisible by num_nodes for simplicity
    initial_data = [np.random.randn(data_size) for _ in range(num_nodes)]

    # Deep copy for verification
    original_data = [d.copy() for d in initial_data]
    expected_sum = sum(original_data)

    rar = RingAllReduce(num_nodes)
    result = rar.all_reduce(initial_data)

    for i in range(num_nodes):
        assert np.allclose(result[i], expected_sum)

    print("Ring All-Reduce simulation successful.")
