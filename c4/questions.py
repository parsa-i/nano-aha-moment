import random
from typing import Callable
from board import Board

class QuestionSpec:
    def __init__(self, name: str, templates: list[str], answer_fn: Callable[[Board], int]):
        self.name = name
        self.templates = templates
        self.answer_fn = answer_fn

    def make_qa(self, board: Board) -> tuple[str, int]:
        template = random.choice(self.templates)
        prompt = f"{template}\n\n{board.render()}\n\n(just the number)."
        answer = self.answer_fn(board)
        return prompt, answer

# Base questions
QUESTION_SPECS: list[QuestionSpec] = [
    # Dimensions
    QuestionSpec(
        "rows",
        ["How many rows are there on the board?", "What is the height of the board?"],
        lambda b: b.height
    ),
    QuestionSpec(
        "columns",
        ["How many columns are there on the board?", "What is the width of the board?"],
        lambda b: b.width
    ),

    # Token counts
    QuestionSpec(
        "num_X",
        ["How many X pieces are on the board?"],
        lambda b: b.count('X')
    ),
    QuestionSpec(
        "num_O",
        ["How many O pieces are on the board?"],
        lambda b: b.count('O')
    ),
    QuestionSpec(
        "num_empty",
        ["How many empty cells are on the board?"],
        lambda b: b.count('.')
    ),

    # Column availability
    QuestionSpec(
        "full_columns",
        ["How many columns are completely full?"],
        lambda b: b.full_columns()
    ),
    QuestionSpec(
        "available_columns",
        ["How many columns still have at least one empty slot?"],
        lambda b: b.available_columns()
    ),
]

# Directional longest-run questions
for token in ('X', 'O'):
    for direction, method in [
        ('horizontal', 'longest_horizontal'),
        ('vertical',   'longest_vertical'),
        ('diagonal',   'longest_diagonal'),
    ]:
        QUESTION_SPECS.append(
            QuestionSpec(
                f"longest_{token}_{direction}",
                [f"What is the length of the longest {direction} contiguous chain of {token}s?"],
                lambda b, m=method, t=token: getattr(b, m)(t)
            )
        )
