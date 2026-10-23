# Experiment: Bidirectional Associative Memory (BAM)

**Script:** `train_bam_component.py`
**Status:** Success

## Objective
The objective was to mathematically implement and evaluate a Bidirectional Associative Memory (BAM) component in pure NumPy. This component models hetero-associative memory, testing its capability to bidirectionally map inputs (X) to outputs (Y) and recover patterns from noisy or incomplete inputs.

## Methodology
The script instantiated a BAM component initialized with zero weights. The training step used Hebbian-style learning ($W = X^T Y$) on a set of bipolar patterns ({-1, 1}). We evaluated the recall of Y from X (both clean and corrupted versions) and the recall of X from Y using the sign threshold activation logic.

## Results
```text
Starting Bidirectional Associative Memory (BAM) Training...
Training complete. Testing recall...
Recall Y for X[0]: [ 1 -1  1 -1] (Expected: [ 1 -1  1 -1])
Recall Y for corrupted X[0]: [ 1 -1  1 -1] (Expected: [ 1 -1  1 -1])
Recall X for Y[0]: [ 1 -1  1 -1  1 -1] (Expected: [ 1 -1  1 -1  1 -1])
Execution time: 0.0008 seconds.
BAM successfully achieved bidirectional recall.
```

## Conclusion
The script successfully implemented the Bidirectional Associative Memory (BAM) algorithm. The component accurately established the bidirectional mappings, achieving perfect recall of associated patterns and demonstrating robustness to input corruption. This validates the mathematical principles of the associative memory model and its suitability for pattern reconstruction tasks.
