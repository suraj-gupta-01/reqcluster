from typing import Annotated

from fastapi import Path

# SQLite ints are signed 64-bit: out-of-range path ids are 422, not an OverflowError 500.
PathId = Annotated[int, Path(ge=1, le=2**63 - 1)]
PathClusterId = Annotated[int, Path(ge=-1, le=2**63 - 1)]  # -1 = noise
