"""
MPS-native LBM operations (Stage 3 - Future Work).

This module will contain operations for running LBM entirely in MPS space:
- Collision as MPO application
- Streaming as permutation MPO
- Moment computation via MPS contraction

Goal: True O(log N) scaling without decompression.
"""
