from __future__ import annotations

from itertools import count
from uuid import uuid4

_event_counter = count(1)


def make_event_id(prefix: str = "EVT") -> str:
    return f"{prefix}-{next(_event_counter):06d}-{uuid4().hex[:8]}"
