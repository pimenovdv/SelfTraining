# Isomap Dimensionality Reduction Component

## Objective
Implement and evaluate an Isomap algorithm mathematically in pure NumPy to demonstrate manifold learning by preserving geodesic distances.

## Implementation Details
The component constructs a neighborhood graph, calculates shortest paths using Floyd-Warshall, and then applies Classical Multi-Dimensional Scaling (MDS) to compute the low-dimensional embedding.

## Results
The component was executed successfully on a synthetic dataset (Swiss Roll-like).
- Data correctly embedded in 2D space.
- Shape matches expected output.

**Script:** `train_isomap_component.py`
