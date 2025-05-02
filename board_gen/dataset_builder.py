# dataset_builder.py
import random, json
from board import Board
from questions import BASE_SPECS, LONGEST_SPECS   # ➊ import lists
from formats import FORMATTERS

def choose_spec(pct_base: float = 0.7):
    """Return a QuestionSpec chosen with the given weighting."""
    if random.random() < pct_base:
        return random.choice(BASE_SPECS)
    return random.choice(LONGEST_SPECS)

def gen_random_moves(width: int, height: int) -> list[int]:
    heights = [0] * width
    seq = []
    max_moves = random.randint(0, width * height)
    for _ in range(max_moves):
        available = [c for c in range(width) if heights[c] < height]
        if not available:
            break
        col = random.choice(available)
        seq.append(col)
        heights[col] += 1
    return seq

def render_board(board: Board) -> tuple[str, str]:
    """
    Convert the Board into text using a randomly-chosen formatter.
    Returns (board_text, formatter_name) so we can store the style in metadata.
    """
    fmt_fn = random.choice(FORMATTERS)
    # board.grid is a tuple-of-tuples; turn it into a list-of-lists for the formatter
    rows = [list(row) for row in board.grid]
    return fmt_fn(rows), fmt_fn.__name__

def make_sample(sample_id: int, pct_base: float = 0.7) -> dict:
    width  = random.randint(4, 6)
    height = random.randint(4, 6)
    moves  = gen_random_moves(width, height)
    board  = Board.from_sequence(width, height, moves)

    spec = choose_spec(pct_base)

    board_text, fmt_name = render_board(board)     # ➋ NEW – random style
    template   = random.choice(spec["templates"])
    opening_variations = [
        "You are given a Connect Four board:",
        "Here is a Connect Four game board:",
        "Below is the current Connect Four board:",
        "Observe the following Connect Four board:",
        "This is the Connect Four board:",
        "Take a look at this Connect Four board:"
    ]
    opening = random.choice(opening_variations)
    prompt     = f"{opening}\n\n```\n{board_text}\n```\n\n{template}"
    answer     = spec["answer_fn"](board)

    return {
        "id": sample_id,
        "prompt": prompt,
        "expected_answer": answer,
        "metadata": {
            "width": width,
            "height": height,
            "moves": moves,
            "question_type": spec["name"],
            "category": spec["category"],
            "board_format": fmt_name,              # ➌ nice to keep record
        },
    }

def build_dataset(n_samples: int = 10000,
                  pct_base: float = 0.7) -> list[dict]:   # ➌ parameter
    return [make_sample(i + 1, pct_base) for i in range(n_samples)]

if __name__ == "__main__":
    dataset = build_dataset(10000, pct_base=0.7) 
    with open("connect_four_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Generated {len(dataset)} samples to connect_four_dataset.json")
