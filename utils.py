import socket
from typing import Any, Dict, List, Tuple, Union

import torch
from deepspeed import DeepSpeedEngine
from transformers import AutoTokenizer, PreTrainedModel


DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
DEFAULT_PROMPT_TEMPLATE = "Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."


def create_prompt(
    numbers: List[int],
    target: int,
    tokenizer: AutoTokenizer,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> str:
    prefix = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": prompt_template.format(numbers=numbers, target=target),
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    return tokenizer.apply_chat_template(
        prefix, tokenize=False, continue_final_message=True
    )


def prepare_model_inputs(
    query_ids: List[List[int]],
    response_ids: List[List[int]],
    advantages: List[List[float]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Prepare padded model inputs with attention masks, labels, and advantages.
    Args:
        query_ids: List of query ids
        response_ids: List of response ids
        advantages: List of lists of advantage values, matching response_ids structure
        device: Device to move the tensors to
    Returns:
        Dict with input_ids, attention_mask, labels, and advantages

    Example:
        >>> query_ids = [[1, 2, 3], [4, 5]]
        >>> response_ids = [[6, 7], [8]]
        >>> advantages = [[0.5, 0.8], [0.3]]
        >>> outputs = prepare_model_inputs(query_ids, response_ids, advantages, "cuda")
        >>> outputs
        {
            'input_ids': tensor([
                [1, 2, 3, 6, 7],
                [4, 5, 8, 0, 0]
            ]),
            'attention_mask': tensor([
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0]
            ]),
            'labels': tensor([
                [-100, -100, -100, 6, 7],
                [-100, -100, 8, -100, -100]
            ]),
            'advantages': tensor([
                [0.0, 0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.9, 0.0]
            ])
        }
    """
    max_seq_len = max(len(q) + len(r) for q, r in zip(query_ids, response_ids))
    inputs = {"input_ids": [], "attention_mask": [], "labels": [], "advantages": []}

    pad_token_id = 0  # Doesn't matter, will be masked
    ignore_index = -100

    for query, response, advantage in zip(query_ids, response_ids, advantages):
        combined_ids = query + response
        seq_len = len(combined_ids)

        # Create padded sequences
        input_ids = combined_ids + [pad_token_id] * (max_seq_len - seq_len)
        attention_mask = [1] * seq_len + [0] * (max_seq_len - seq_len)
        labels = (
            [ignore_index] * len(query)
            + response
            + [ignore_index] * (max_seq_len - seq_len)
        )
        advantages_seq = (
            [0.0] * len(query) + advantage + [0.0] * (max_seq_len - seq_len)
        )

        assert len(input_ids) == max_seq_len
        assert len(attention_mask) == max_seq_len
        assert len(labels) == max_seq_len
        assert len(advantages_seq) == max_seq_len

        inputs["input_ids"].append(input_ids)
        inputs["attention_mask"].append(attention_mask)
        inputs["labels"].append(labels)
        inputs["advantages"].append(advantages_seq)

    # Convert to tensors
    return {
        k: torch.tensor(
            v, dtype=torch.long if k != "advantages" else torch.float, device=device
        )
        for k, v in inputs.items()
    }


def compute_token_log_probs(
    model: Union[DeepSpeedEngine, PreTrainedModel],
    inputs: Dict[str, torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    """
    Compute log probabilities for each token in the sequence, masked for valid labels only.

    This function:
    1. Runs the model forward pass
    2. Applies temperature scaling to logits
    3. Shifts the sequences for causal language modeling
    4. Computes log probabilities for the actual tokens that appeared in the sequence
    5. Masks the log probabilities to only include valid labels (non -100 positions)

    Args:
        model: The language model (either DeepSpeed-wrapped or regular HuggingFace model)
        inputs: Dictionary containing:
            - input_ids: Tensor of token ids [batch_size, seq_len]
            - attention_mask: Tensor of attention mask [batch_size, seq_len]
            - labels: Tensor of target labels [batch_size, seq_len] with -100 for ignored positions
        temperature: Temperature for scaling the logits before softmax

    Returns:
        torch.Tensor: Log probabilities tensor of shape [batch_size, seq_len-1], where:
            - Each value is the log probability of the actual token that appeared
            - Values are masked to 0.0 for positions where labels were -100
            - The sequence length is reduced by 1 due to the causal shift

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> inputs = {
        ...     "input_ids": torch.tensor([[1, 2, 3]]),
        ...     "attention_mask": torch.tensor([[1, 1, 1]]),
        ...     "labels": torch.tensor([[-100, 2, 3]])
        ... }
        >>> log_probs = compute_token_log_probs(model, inputs, temperature=1.0)
        >>> log_probs.shape
        torch.Size([1, 2])  # batch_size=1, seq_len-1=2
        >>> # First position is 0 (masked), second position has actual log prob
    """
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict=True,
        use_cache=False,
    )

    logits = (
        outputs.logits.float() / temperature
    )  # Shape: [batch_size, seq_len, vocab_size]
    shift_logits = logits[
        ..., :-1, :
    ].contiguous()  # Shape: [batch_size, seq_len-1, vocab_size]
    shift_labels = inputs["labels"][
        ..., 1:
    ].contiguous()  # Shape: [batch_size, seq_len-1]

    # Create mask for valid labels
    label_mask = (shift_labels != -100).float()  # Shape: [batch_size, seq_len-1]
    shift_labels[shift_labels == -100] = 0  # Shape: [batch_size, seq_len-1]

    # Calculate log probabilities
    log_probs = torch.log_softmax(
        shift_logits, dim=-1
    )  # Shape: [batch_size, seq_len-1, vocab_size]
    log_probs = torch.gather(
        log_probs, dim=2, index=shift_labels.unsqueeze(2)
    )  # Shape: [batch_size, seq_len-1, 1]
    log_probs = log_probs.squeeze(2)  # Shape: [batch_size, seq_len-1]
    log_probs = log_probs * label_mask  # Shape: [batch_size, seq_len-1]

    return log_probs


def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

