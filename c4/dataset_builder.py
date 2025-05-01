import random
import json
from board import Board
from questions import QUESTION_SPECS

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


def make_sample(sample_id: int) -> dict:
    width = random.randint(4, 8)
    height = random.randint(4, 8)
    moves = gen_random_moves(width, height)
    board = Board.from_sequence(width, height, moves)
    spec = random.choice(QUESTION_SPECS)
    prompt, answer = spec.make_qa(board)
    return {
        "id": sample_id,
        "prompt": prompt,
        "expected_answer": answer,
        "metadata": {
            "width": width,
            "height": height,
            "moves": moves,
            "question_type": spec.name,
        },
    }


def build_dataset(n_samples: int = 4000) -> list[dict]:
    return [make_sample(i+1) for i in range(n_samples)]


if __name__ == "__main__":
    dataset = build_dataset(4000)
    with open("connect_four_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Generated {len(dataset)} samples to connect_four_dataset.json")
