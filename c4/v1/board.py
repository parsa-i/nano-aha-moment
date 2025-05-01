from __future__ import annotations
from dataclasses import dataclass
import itertools

Token = str  # 'X', 'O', or '.'

@dataclass(frozen=True)
class Board:
    width: int
    height: int
    grid: tuple[tuple[Token, ...], ...]

    @classmethod
    def from_sequence(cls, width: int, height: int, moves: list[int]) -> "Board":
        """Create a board by dropping X/O in alternating order according to `moves`."""
        grid = [['.' for _ in range(width)] for _ in range(height)]
        top = [height - 1] * width
        players = itertools.cycle(['X', 'O'])
        for move, player in zip(moves, players):
            if 0 <= move < width and top[move] >= 0:
                grid[top[move]][move] = player
                top[move] -= 1
        return cls(width, height, tuple(tuple(row) for row in grid))

    def count(self, token: Token, row: int | None = None, col: int | None = None) -> int:
        """Count occurrences of `token` on the whole board, or in one row/column."""
        if row is not None:
            return self.grid[row].count(token)
        if col is not None:
            return sum(self.grid[r][col] == token for r in range(self.height))
        return sum(row.count(token) for row in self.grid)

    def full_columns(self) -> int:
        """Number of columns with no empty cells."""
        return sum(self.grid[0][c] != '.' for c in range(self.width))

    def available_columns(self) -> int:
        """Number of columns with at least one empty cell."""
        return self.width - self.full_columns()

    def _longest_in_dirs(self, token: Token, dirs: list[tuple[int,int]]) -> int:
        """Internal: longest contiguous run of `token` in given directions."""
        best = 0
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c] != token:
                    continue
                for dr, dc in dirs:
                    length, rr, cc = 0, r, c
                    while 0 <= rr < self.height and 0 <= cc < self.width and self.grid[rr][cc] == token:
                        length += 1
                        rr += dr
                        cc += dc
                    best = max(best, length)
        return best

    def longest_horizontal(self, token: Token) -> int:
        """Longest horizontal run of `token`."""
        return self._longest_in_dirs(token, [(0, 1)])

    def longest_vertical(self, token: Token) -> int:
        """Longest vertical run of `token`."""
        return self._longest_in_dirs(token, [(1, 0)])

    def longest_diagonal(self, token: Token) -> int:
        """Longest diagonal run of `token` (both SE and SW)."""
        return self._longest_in_dirs(token, [(1, 1), (1, -1)])

    def render(self) -> str:
        """Return a textual rendering of the board (rows top→bottom)."""
        lines = [' '.join(row) for row in self.grid]
        lines.append(' '.join(map(str, range(self.width))))
        return '\n'.join(lines)
