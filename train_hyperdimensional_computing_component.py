import numpy as np
import time

class HDCComponent:
    def __init__(self, d=10000, num_classes=3):
        self.d = d
        self.num_classes = num_classes
        self.item_memory = {}
        self.class_hypervectors = np.zeros((num_classes, d), dtype=np.int8)

    def generate_random_hypervector(self):
        return np.random.choice([-1, 1], size=self.d)

    def get_item_hypervector(self, key):
        if key not in self.item_memory:
            self.item_memory[key] = self.generate_random_hypervector()
        return self.item_memory[key]

    def bind(self, hv1, hv2):
        return hv1 * hv2

    def bundle(self, hvs):
        sum_hv = np.sum(hvs, axis=0)
        return np.where(sum_hv >= 0, 1, -1)

    def train_step(self, features, label):
        hvs = []
        for i, val in enumerate(features):
            val_quantized = int(val * 10)
            key = f"feat_{i}_val_{val_quantized}"
            hvs.append(self.get_item_hypervector(key))

        bundled_hv = self.bundle(hvs)
        self.class_hypervectors[label] += bundled_hv
        self.class_hypervectors[label] = np.where(self.class_hypervectors[label] >= 0, 1, -1)

    def predict(self, features):
        hvs = []
        for i, val in enumerate(features):
            val_quantized = int(val * 10)
            key = f"feat_{i}_val_{val_quantized}"
            hvs.append(self.get_item_hypervector(key))

        bundled_hv = self.bundle(hvs)

        similarities = []
        for i in range(self.num_classes):
            sim = np.dot(bundled_hv, self.class_hypervectors[i]) / self.d
            similarities.append(sim)

        return np.argmax(similarities)

def test_hdc_component():
    print("Testing Hyperdimensional Computing Component...")
    hdc = HDCComponent(d=10000, num_classes=2)

    # Train
    for _ in range(50):
        # Class 0: features around 0.2
        features0 = np.random.normal(0.2, 0.05, 5)
        hdc.train_step(features0, 0)

        # Class 1: features around 0.8
        features1 = np.random.normal(0.8, 0.05, 5)
        hdc.train_step(features1, 1)

    # Test
    correct = 0
    total = 20
    for _ in range(total // 2):
        pred = hdc.predict(np.random.normal(0.2, 0.05, 5))
        if pred == 0: correct += 1

        pred = hdc.predict(np.random.normal(0.8, 0.05, 5))
        if pred == 1: correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
    if accuracy > 0.8:
        print("Test passed successfully.")
    else:
        print("Test failed: Accuracy too low.")

if __name__ == "__main__":
    test_hdc_component()
