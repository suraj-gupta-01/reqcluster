from typing import Annotated

from reqcluster.domain import DB_INT_MAX
from fastapi import Path

# SQLite ints are signed 64-bit: out-of-range path ids are 422, not an OverflowError 500.
PathId = Annotated[int, Path(ge=1, le=DB_INT_MAX)]
PathClusterId = Annotated[int, Path(ge=-1, le=DB_INT_MAX)]  # -1 = noise
PathJobId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]  # uuid4().hex
