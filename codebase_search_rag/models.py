from dataclasses import dataclass

@dataclass
class Chunk:
    id: str
    content: str
    start_line: int
    end_line: int
    path: str
    metadata: dict
