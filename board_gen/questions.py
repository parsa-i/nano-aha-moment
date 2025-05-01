import random
from typing import Callable, Literal, TypedDict
from board import Board

Category = Literal["base", "longest"]

class QuestionSpec(TypedDict):
    name: str
    templates: list[str]
    answer_fn: Callable[[Board], int]
    category: Category   

# ------------------------------------------------------------------
# ➊  Base-question helpers
# ------------------------------------------------------------------
def _row_name(idx_from_bottom: int, h: int) -> str:
    # idx_from_bottom = 0 → bottom row, 1 → second-bottom, …
    names = ["bottom", "second-bottom", "third-bottom"]
    if idx_from_bottom < len(names):
        return names[idx_from_bottom]
    return f"{idx_from_bottom+1}-th bottom"

def _row_idx(b: Board, idx_from_bottom: int) -> int:
    return b.height - 1 - idx_from_bottom

def _col_name(idx_from_right: int, w: int) -> str:
    names = ["right-most", "second-right-most", "third-right-most"]
    if idx_from_right < len(names):
        return names[idx_from_right]
    return f"{idx_from_right+1}-th right-most"

def _col_idx(b: Board, idx_from_right: int) -> int:
    return b.width - 1 - idx_from_right

BASE_SPECS: list[QuestionSpec] = [
    # Dimensions
    dict(
        name="rows",
        templates=["How many rows are there on the board?",
                   "What is the height of the board?"],
        answer_fn=lambda b: b.height,
        category="base",
    ),
    dict(
        name="columns",
        templates=["How many columns are there on the board?",
                   "What is the width of the board?"],
        answer_fn=lambda b: b.width,
        category="base",
    ),

    # Whole-board counts
    dict(
        name="num_X",
        templates=["How many X pieces are on the board?"],
        answer_fn=lambda b: b.count('X'),
        category="base",
    ),
    dict(
        name="num_O",
        templates=["How many O pieces are on the board?"],
        answer_fn=lambda b: b.count('O'),
        category="base",
    ),
    dict(
        name="num_empty",
        templates=["How many empty cells are on the board?"],
        answer_fn=lambda b: b.count('.'),
        category="base",
    ),

    # Column availability
    dict(
        name="full_columns",
        templates=["How many columns are completely full?"],
        answer_fn=lambda b: b.full_columns(),
        category="base",
    ),
    dict(
        name="available_columns",
        templates=["How many columns still have at least one empty slot?"],
        answer_fn=lambda b: b.available_columns(),
        category="base",
    ),
]

# ──────────────────────────────────────────────────────────────────
# ➋  Row-specific and column-specific counts (bottom 3 & right-most 3)
# ──────────────────────────────────────────────────────────────────
for token in ('X', 'O'):
    # bottom / second-bottom / third-bottom rows
    for i in range(3):
        BASE_SPECS.append(dict(
            name=f"num_{token}_row_{i}",
            templates=[f"How many {token} pieces are in the {_row_name(i, 0)} row?"],
            answer_fn=lambda b, j=i, t=token: b.count(t, row=_row_idx(b, j)),
            category="base",
        ))

    # right-most / second-right-most / third-right-most columns
    for i in range(3):
        BASE_SPECS.append(dict(
            name=f"num_{token}_col_{i}",
            templates=[f"How many {token} pieces are in the {_col_name(i, 0)} column?"],
            answer_fn=lambda b, j=i, t=token: b.count(t, col=_col_idx(b, j)),
            category="base",
        ))

# ------------------------------------------------------------------
# ➌  Directional longest-run specs (unchanged, but tagged “longest”)
# ------------------------------------------------------------------
LONGEST_SPECS: list[QuestionSpec] = []
for token in ('X', 'O'):
    for direction, method in [
        ('horizontal', 'longest_horizontal'),
        ('vertical',   'longest_vertical'),
        ('diagonal',   'longest_diagonal'),
    ]:
        LONGEST_SPECS.append(dict(
            name=f"longest_{token}_{direction}",
            templates=[f"What is the length of the longest {direction} contiguous chain of {token}s?"],
            answer_fn=lambda b, m=method, t=token: getattr(b, m)(t),
            category="longest",
        ))

# Public, flat list (if ever needed)
QUESTION_SPECS: list[QuestionSpec] = BASE_SPECS + LONGEST_SPECS
