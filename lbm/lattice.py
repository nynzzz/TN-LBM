"""
Lattice definitions for LBM velocity sets.

Each lattice defines:
  - c: velocity vectors
  - w: weights
  - cs2: speed of sound squared
  - opposite: index of opposite direction for each velocity
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Lattice:
    """Base class for lattice velocity sets."""
    name: str
    d: int  # spatial dimensions
    q: int  # number of velocities
    c: np.ndarray  # velocity vectors, shape (q,) for 1D or (q, d) for 2D+
    w: np.ndarray  # weights, shape (q,)
    cs2: float  # speed of sound squared
    opposite: np.ndarray  # opposite direction indices, shape (q,)

    def __post_init__(self):
        # Validate shapes
        assert len(self.w) == self.q
        assert len(self.opposite) == self.q


# D1Q3: 1D lattice with 3 velocities
# i=0: stationary (c=0)
# i=1: right (c=+1)
# i=2: left (c=-1)
D1Q3 = Lattice(
    name="D1Q3",
    d=1,
    q=3,
    c=np.array([0, 1, -1]),
    w=np.array([2/3, 1/6, 1/6]),
    cs2=1/3,
    opposite=np.array([0, 2, 1]),
)


# D2Q9: 2D lattice with 9 velocities
# Velocity ordering:
#   6 2 5
#   3 0 1
#   7 4 8
#
# i=0: stationary
# i=1: East   (+1, 0)
# i=2: North  (0, +1)
# i=3: West   (-1, 0)
# i=4: South  (0, -1)
# i=5: NE     (+1, +1)
# i=6: NW     (-1, +1)
# i=7: SW     (-1, -1)
# i=8: SE     (+1, -1)
D2Q9 = Lattice(
    name="D2Q9",
    d=2,
    q=9,
    c=np.array([
        [0, 0],    # 0: stationary
        [1, 0],    # 1: E
        [0, 1],    # 2: N
        [-1, 0],   # 3: W
        [0, -1],   # 4: S
        [1, 1],    # 5: NE
        [-1, 1],   # 6: NW
        [-1, -1],  # 7: SW
        [1, -1],   # 8: SE
    ]),
    w=np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]),
    cs2=1/3,
    opposite=np.array([0, 3, 4, 1, 2, 7, 8, 5, 6]),
)
