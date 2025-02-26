import os
# Environment configuration
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', 'your_token_here')
os.environ['HF_HOME'] = os.getenv('HF_HOME', 'your_hf_home_dir_here')
os.environ["CUDA_HOME"] = os.getenv('CUDA_HOME', 'your_cuda_home_dir_here')
import re
import random
import numpy as np
import torch
import deepspeed
import sglang
import wandb
import time
import argparse
import socket
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from typing import List
from tqdm import trange

def generate_r1_prompt(numbers, target, tokenizer):
    r1_prefix = [{
        "role": "system",
        "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
      },
      { 
        "role": "user",
        "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."
      },
      {
        "role": "assistant",
        "content": "Let me solve this step by step.\n<think>"
      }]
    input_ids = tokenizer.apply_chat_template(r1_prefix, tokenize=True, continue_final_message=True)
    prompt = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {"prompt": prompt, "target": target, "input_ids": input_ids}

def format_reward_func(completion, EOS_TOKEN):
    """
    Format: <think>...</think><answer>...</answer>
    
    Also checks that the content within <answer>...</answer> conforms to a 
    specified pattern (only digits, + - * / ( ) . and whitespace).
    
    Args:
        completion (str): Generated output
        EOS_TOKEN (str): End of sequence token
      
    Returns:
        float: Reward score
    """
    # Define the allowed pattern (only numbers, +, -, *, /, (, ), ., and whitespace)
    allowed_pattern = r'^[\d+\-*/().\s]+$'

    try:
        # Synthetically prepend <think> (if your pipeline relies on that to ease matching)
        completion = "<think>" + completion

        # Strip EOS token if present
        if completion.endswith(EOS_TOKEN):
            completion = completion[:-len(EOS_TOKEN)]
        
        # Check if the format is correct
        # Pattern means:
        # 1) <think>...contents not including other <think> tags...</think>
        # 2) \n
        # 3) <answer>...anything...</answer> 
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(regex, completion, re.DOTALL)

        if match is None or len(match.groups()) != 2:
            # Format is incorrect
            return 0.0
        else:
            # Extract the content inside <answer>...</answer>
            answer_content = match.group(2).strip()

            # Check if answer content matches the allowed pattern
            if not re.match(allowed_pattern, answer_content):
                # If it doesn't match, reward is 0.5
                return 0.5
            else:
                # If both format and pattern are correct, reward is 1
                return 1.0
    except Exception:
        # Any error leads to 0 reward
        return 0.0


def equation_reward_func(completion, target, nums):
    """
    Evaluates completion based on mathematical correctness of the answer

    Args:
        completion (str): Generated output
        target (str): Expected answer
        nums (list): Available numbers to use in the equation
    
    Returns:
        float: Reward score
    """
    try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            return 0.0
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
        
        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(nums):
            return 0.0
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation):
           return 0.0
        
        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
        else:
            return 0.0
    except Exception:
        # If evaluation fails, reward is 0
        return 0.0


def initialize_model(model_name: str) -> AutoModelForCausalLM:
    """Initialize model with flash attention and appropriate device placement."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return model

def prepare_model_inputs(query_ids: List[List[int]], response_ids: List[List[int]]) -> dict:
    """Prepare padded model inputs with attention masks and labels."""
    max_seq_len = max(len(q) + len(r) for q, r in zip(query_ids, response_ids))
    
    batch_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    
    pad_token_id = 0
    ignore_index = -100

    for query, response in zip(query_ids, response_ids):
        combined_ids = query + response
        seq_len = len(combined_ids)
        
        # Create padded sequences
        input_ids = combined_ids + [pad_token_id] * (max_seq_len - seq_len)
        attention_mask = [1] * seq_len + [0] * (max_seq_len - seq_len)
        labels = [ignore_index] * len(query) + response + [ignore_index] * (max_seq_len - seq_len)
        
        assert len(input_ids) == max_seq_len
        assert len(attention_mask) == max_seq_len
        assert len(labels) == max_seq_len
        
        batch_inputs["input_ids"].append(input_ids)
        batch_inputs["attention_mask"].append(attention_mask)
        batch_inputs["labels"].append(labels)
    
    # Convert to tensors
    return {
        k: torch.tensor(v, dtype=torch.long, device="cuda")
        for k, v in batch_inputs.items()
    }

def compute_token_log_probs(model_engine, inputs: dict, temperature) -> torch.Tensor:
    outputs = model_engine(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict=True,
        use_cache=False,
    )
    
    logits = outputs.logits.float() / temperature
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["labels"][..., 1:].contiguous()
    
    # Create mask for valid labels
    label_mask = (shift_labels != -100).float()
    shift_labels[shift_labels == -100] = 0
    
    # Calculate log probabilities
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(2))
    token_log_probs = token_log_probs.squeeze(2)
    return token_log_probs * label_mask

def compute_pg_loss(policy_model, reference_model, batch_model_inputs, batch_advantages, total_response_len, kl_coefficient, temperature):
    """
    Compute policy gradient loss with KL penalty.
    
    Args:
        policy_model: The policy model being trained
        reference_model: The reference model for KL divergence calculation
        batch_model_inputs: Dictionary containing input_ids, attention_mask, and labels
        batch_advantages: Tensor of advantage values for each sample in the batch
        total_response_len: Total number of response tokens (for normalization)
        kl_coefficient: Coefficient for KL penalty term
        temperature: Temperature for sampling
        
    Returns:
        tuple: (total_loss, policy_loss, kl_penalty, entropy)
    """
    labels_mask = (batch_model_inputs["labels"][..., 1:] != -100).float()
    
    with torch.no_grad():
        ref_logps = compute_token_log_probs(reference_model, batch_model_inputs, temperature)
    
    logps = compute_token_log_probs(policy_model, batch_model_inputs, temperature)
    
    kl_penalty = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
    kl_penalty = kl_penalty * labels_mask
    
    entropy = -logps.sum() / total_response_len
    
    policy_loss = -logps * batch_advantages.unsqueeze(1)
    policy_loss = policy_loss * labels_mask
    
    total_loss = (policy_loss + kl_coefficient * kl_penalty).sum() / total_response_len
    
    return total_loss, policy_loss, kl_penalty, entropy

def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def evaluate_on_test_set(sglang_engine, test_dataset, tokenizer, EOS_TOKEN, SAMPLING_PARAMS):
    """Evaluate the model on test set"""
    print("Evaluating on test set...")
    eval_start_time = time.time()
    
    # Create evaluation sampling params with fixed values
    eval_sampling_params = {
        "temperature": 0.3,
        "max_new_tokens": 1024,
        "top_p": 1.0,
        "n": 1,  # Only generate one response per question
    }
    
    generations = sglang_engine.generate(input_ids=test_dataset["input_ids"], sampling_params=eval_sampling_params)
    
    # Process responses and calculate rewards
    all_rewards = []
    all_format_rewards = []
    all_equation_rewards = []
    response_lengths = []
    non_stop_reasons = []
    
    for i, datum in enumerate(test_dataset):
        response_token_ids = generations[i]["token_ids"]
        finish_reason = generations[i]["meta_info"]["finish_reason"]["type"]
        
        non_stop_reasons.append(finish_reason != "stop")
        response_lengths.append(len(response_token_ids))
        
        response = tokenizer.batch_decode([response_token_ids], skip_special_tokens=False)[0]
        
        equation_reward = equation_reward_func(
            response, 
            datum["target"], 
            datum["nums"]
        )
        format_reward = format_reward_func(
            response,
            EOS_TOKEN
        )
        
        all_format_rewards.append(format_reward)
        all_equation_rewards.append(equation_reward)
        reward = format_reward + equation_reward
        all_rewards.append(reward)
    
    eval_time = time.time() - eval_start_time
    
    # Compute metrics
    eval_stats = {
        # Generation quality metrics
        "test/generation/non_stop_rate": np.mean(non_stop_reasons),
        "test/generation/mean_response_length": np.mean(response_lengths),
        "test/generation/max_response_length": np.max(response_lengths),
        
        # Overall reward metrics
        "test/reward_mean": np.mean(all_rewards),
        "test/reward_std": np.std(all_rewards),
        
        # Format reward metrics
        "test/format_reward/mean": np.mean(all_format_rewards),
        "test/format_reward/std": np.std(all_format_rewards),
        
        # Equation reward metrics
        "test/equation_reward/mean": np.mean(all_equation_rewards),
        "test/equation_reward/std": np.std(all_equation_rewards),
        
        # Timing metrics
        "test/eval_time": eval_time,
    }
    
    return eval_stats

def training_episode_generator(generations, batch_samples, tokenizer, EOS_TOKEN, EOS_TOKEN_ID, ROLLOUTS_PER_EPISODE):
    """
    Process model generations and calculate rewards for training episodes.
    
    Args:
        generations (list): List of generation results from sglang engine
        batch_samples (Dataset): Batch of samples from the training dataset
        tokenizer: Tokenizer for decoding responses
        EOS_TOKEN (str): End of sequence token
        EOS_TOKEN_ID (int): End of sequence token ID
        ROLLOUTS_PER_EPISODE (int): Number of rollouts per episode
        
    Returns:
        dict: Dictionary containing processed data for training:
            - all_queries: List of input token IDs
            - all_responses: List of response token IDs
            - all_advantages: List of advantage values
            - stats: Dictionary of statistics about the generations
    """
    # Process responses and calculate rewards
    groups = [list(range(i, i+ROLLOUTS_PER_EPISODE)) for i in range(0, len(generations), ROLLOUTS_PER_EPISODE)]
    all_queries, all_responses, all_advantages = [], [], []
    all_rewards = []
    all_format_rewards = []
    all_equation_rewards = []
    response_lengths = []
    non_stop_reasons = []  # Track non-stop finish reasons
    
    has_printed_example = False
    
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    for datum, group_indices in zip(batch_samples, groups):
        response_token_ids = [generations[i]["token_ids"] for i in group_indices]
        finish_reasons = [generations[i]["meta_info"]["finish_reason"]["type"] for i in group_indices]
        
        # Track non-stop finish reasons
        non_stop_reasons.extend([fr != "stop" for fr in finish_reasons])
        
        # Track response lengths
        response_lengths.extend([len(ids) for ids in response_token_ids])
        
        responses = tokenizer.batch_decode(response_token_ids, skip_special_tokens=False)
        
        if not has_printed_example:
            print(f"Query: {datum['prompt']}")
            print(f"Response: {responses[0]}")
            has_printed_example = True
        
        # Process each response individually
        batch_equation_rewards = []
        batch_format_rewards = []
        
        for response in responses:
            equation_reward = equation_reward_func(
                response, 
                datum["target"], 
                datum["nums"]
            )
            format_reward = format_reward_func(
                response,
                EOS_TOKEN
            )
            
            batch_equation_rewards.append(equation_reward)
            batch_format_rewards.append(format_reward)
        
        # Track individual rewards
        all_format_rewards.extend(batch_format_rewards)
        all_equation_rewards.extend(batch_equation_rewards)
        
        rewards = np.array(batch_format_rewards) + np.array(batch_equation_rewards)
        all_rewards.extend(rewards)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        
        response_token_ids = [
            (r + [EOS_TOKEN_ID]) if fr == "stop" else r 
            for r, fr in zip(response_token_ids, finish_reasons)
        ]
        
        all_queries.extend([datum["input_ids"]] * ROLLOUTS_PER_EPISODE)
        all_responses.extend(response_token_ids)
        all_advantages.extend(advantages.tolist())
    
    # Prepare statistics
    stats = {
        # Generation quality metrics
        "non_stop_rate": np.mean(non_stop_reasons),
        "mean_response_length": np.mean(response_lengths),
        "max_response_length": np.max(response_lengths),
        
        # Overall reward metrics
        "reward_mean": np.mean(all_rewards),
        "reward_std": np.std(all_rewards),
        
        # Format reward metrics
        "format_reward_mean": np.mean(all_format_rewards),
        "format_reward_std": np.std(all_format_rewards),
        
        # Equation reward metrics
        "equation_reward_mean": np.mean(all_equation_rewards),
        "equation_reward_std": np.std(all_equation_rewards),
    }
    
    return {
        "all_queries": all_queries,
        "all_responses": all_responses,
        "all_advantages": all_advantages,
        "stats": stats
    }

def main():    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train R1 model with PPO')
    parser.add_argument('--kl_coeff', type=float, default=0.001, help='KL coefficient for PPO')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-3B", help='Model name/path')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate for training')
    args = parser.parse_args()

    # Get a random free port for DeepSpeed
    free_port = find_free_port()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(free_port)
    os.environ['RANK'] = "0"
    os.environ['LOCAL_RANK'] = "0"
    os.environ['WORLD_SIZE'] = "1"

    # Model configuration
    MODEL_NAME = args.model_name
    MODEL_CHAT_NAME = MODEL_NAME + "-Instruct"
    DATASET_ID = "Jiayi-Pan/Countdown-Tasks-3to4"

    # Flash attention configuration
    model_config = {
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.bfloat16,
        "device_map": "auto"
    }

    # Dataset configuration
    TEST_SPLIT_SIZE = 500

    # Load and process dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHAT_NAME)
    EOS_TOKEN_ID = AutoTokenizer.from_pretrained(MODEL_NAME).eos_token_id
    EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)
    dataset = load_dataset(DATASET_ID, split="train")
    dataset = dataset.map(lambda example: generate_r1_prompt(example["nums"], example["target"], tokenizer), num_proc=6)

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=TEST_SPLIT_SIZE)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    # RL parameters
    NUM_ITERATIONS = 1000
    EPISODES_PER_ITERATION = 64
    ROLLOUTS_PER_EPISODE = 4
    SAMPLING_PARAMS = {
        "temperature": args.temperature,
        "top_p": 1.0,
        "max_new_tokens": 1024,
        "n": ROLLOUTS_PER_EPISODE
    }
    
    # Training hyperparameters
    PER_DEVICE_BATCH_SIZE = 4
    grad_acc_steps = EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE
    LEARNING_RATE = args.learning_rate
    KL_COEFFICIENT = args.kl_coeff
    TEMPERATURE = args.temperature

    # DeepSpeed configuration
    deepspeed_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 2,
                              "overlap_comm": False},
        "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
        "gradient_clipping": 1.0,
        "gradient_accumulation_steps": grad_acc_steps,
        "train_batch_size": EPISODES_PER_ITERATION,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": LEARNING_RATE,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "torch_adam": True, # WARNING:crucial, deepspeed optimizer weirdly does not update model at all when > 2B
            }
        },
        # "scheduler": {
        #     "type": "WarmupDecayLR",
        #     "params": {
        #         "warmup_min_lr": 0,
        #         "warmup_max_lr": LEARNING_RATE,
        #         "warmup_num_steps": int(0.06 * NUM_ITERATIONS),
        #         "total_num_steps": NUM_ITERATIONS
        #     }
        # }
    }

    # Reference model config (without optimizer and scheduler)
    ref_ds_config = {
        "bf16": {
            "enabled": True
        },
        "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": grad_acc_steps,
        "train_batch_size": EPISODES_PER_ITERATION,
    }

    # Initialize main and reference models
    policy_model = initialize_model(MODEL_NAME)
    policy_model.gradient_checkpointing_enable()
    reference_model = initialize_model(MODEL_NAME)

    # Initialize DeepSpeed engines
    policy_model, *_ = deepspeed.initialize(
        model=policy_model,
        config=deepspeed_config,
        model_parameters=policy_model.parameters()
    )

    reference_model, *_ = deepspeed.initialize(
        model=reference_model,
        config=ref_ds_config,
    )

    reference_model.module.cpu()

    # Initialize SGLang engine
    sglang_engine = sglang.Engine(
        model_path=MODEL_NAME,
        enable_memory_saver=True,
        skip_tokenizer_init=True,
        mem_fraction_static=0.20,
        schedule_policy="fcfs",
        schedule_conservativeness=0.001,
        max_running_requests=10000,
    )
    sglang_engine.release_memory_occupation()

    # Initialize wandb
    model_name_short = MODEL_NAME.split('/')[-1]  # Get the last part of model name
    run_name = f"{model_name_short}_temp{TEMPERATURE}_kl{KL_COEFFICIENT}_lr{LEARNING_RATE}"
    wandb.init(
        project="r1-aha-moment",
        name=run_name,
        config={
            "model_name": MODEL_NAME,
            "learning_rate": LEARNING_RATE,
            "num_iterations": NUM_ITERATIONS,
            "episodes_per_iteration": EPISODES_PER_ITERATION,
            "rollouts_per_episode": ROLLOUTS_PER_EPISODE,
            "kl_coefficient": KL_COEFFICIENT,
            "temperature": TEMPERATURE,
            "slurm_job_id": os.environ.get('SLURM_JOB_ID', None),
        }
    )
    
    # Enable debugpy for remote debugging
    if os.environ.get("DEBUGPY", "0") == "1":
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached!")

    for iteration in trange(NUM_ITERATIONS):
        print(f"Iteration {iteration}/{NUM_ITERATIONS}")
        iteration_start_time = time.time()
        generation_start_time = time.time()
        
        # Sample training batch
        num_ds_samples = EPISODES_PER_ITERATION // ROLLOUTS_PER_EPISODE
        batch_indices = np.random.choice(len(train_dataset), size=num_ds_samples, replace=False)
        batch_samples = train_dataset.select(batch_indices)
        
        # Update model weights in SGLang engine
        torch.cuda.empty_cache()
        time.sleep(0)
        
        sglang_engine.resume_memory_occupation()
        success, error = sglang_engine.update_weights_from_tensor(list(policy_model.module.named_parameters()))
        if not success:
            raise RuntimeError(f"Weight update failed: {error}")
        
        eval_stats = None
        if iteration % 25 == 0:
            eval_stats = evaluate_on_test_set(
                sglang_engine=sglang_engine,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                EOS_TOKEN=EOS_TOKEN,
                SAMPLING_PARAMS=SAMPLING_PARAMS
            )
            time.sleep(2) # so sglang scheduler cools down

        # Generate responses
        generations = sglang_engine.generate(input_ids=batch_samples["input_ids"], sampling_params=SAMPLING_PARAMS)
        print(f"Generated {len(generations)} responses")
        sglang_engine.release_memory_occupation()
        time.sleep(1) # WARNING: hacky, to make sure the memory is released before training
        
        generation_time = time.time() - generation_start_time
        training_start_time = time.time()

        # Process responses and calculate rewards
        training_episode_data = training_episode_generator(
            generations,
            batch_samples,
            tokenizer,
            EOS_TOKEN,
            EOS_TOKEN_ID,
            ROLLOUTS_PER_EPISODE
        )

        # Prepare training batch
        model_inputs = prepare_model_inputs(training_episode_data["all_queries"], training_episode_data["all_responses"])
        advantages_tensor = torch.tensor(training_episode_data["all_advantages"], device="cuda")
        
        # Calculate losses and update model
        policy_model.train()
        reference_model.module.cuda()
        reference_model.eval()
        
        total_response_len = (model_inputs["labels"] != -100).sum()
        
        # Track metrics
        total_policy_loss = 0
        total_kl_penalty = 0
        total_entropy = 0
        grad_norm = 0
        
        
        for i in range(0, EPISODES_PER_ITERATION, PER_DEVICE_BATCH_SIZE):
            print(f"Processing batch {i}/{EPISODES_PER_ITERATION}")
            batch_model_inputs = {
                "input_ids": model_inputs["input_ids"][i:i+PER_DEVICE_BATCH_SIZE],
                "attention_mask": model_inputs["attention_mask"][i:i+PER_DEVICE_BATCH_SIZE],
                "labels": model_inputs["labels"][i:i+PER_DEVICE_BATCH_SIZE]
            }
            batch_advantages = advantages_tensor[i:i+PER_DEVICE_BATCH_SIZE]
            
            # Compute policy gradient loss
            total_loss, policy_loss, kl_penalty, entropy = compute_pg_loss(
                policy_model=policy_model,
                reference_model=reference_model,
                batch_model_inputs=batch_model_inputs,
                batch_advantages=batch_advantages,
                total_response_len=total_response_len,
                kl_coefficient=KL_COEFFICIENT,
                temperature=TEMPERATURE
            )
            
            # Track metrics
            total_policy_loss += policy_loss.sum().item() / total_response_len
            total_kl_penalty += kl_penalty.sum().item() / total_response_len
            print(f"total_kl_penalty: {total_kl_penalty}")
            total_entropy += entropy.item()
            grad_norm = policy_model.get_global_grad_norm()
            
            # Backpropagation and optimization step
            policy_model.backward(total_loss, scale_wrt_gas=False)
            # del policy_loss, kl_penalty, entropy, total_loss # free memory, avoid Cuda OOM
            if policy_model.is_gradient_accumulation_boundary():
                reference_model.module.cpu()
            
            policy_model.step()
        
        print("Finished training")
        
        training_time = time.time() - training_start_time
        total_iteration_time = time.time() - iteration_start_time
        
        # Log metrics to wandb
        stats = {
            "iteration": iteration,
            # Generation quality metrics
            "train/generation/non_stop_rate": training_episode_data["stats"]["non_stop_rate"],
            "train/generation/mean_response_length": training_episode_data["stats"]["mean_response_length"],
            "train/generation/max_response_length": training_episode_data["stats"]["max_response_length"],
            
            # Overall reward metrics
            "train/reward_mean": training_episode_data["stats"]["reward_mean"],
            "train/reward_std": training_episode_data["stats"]["reward_std"],
            
            # Format reward metrics
            "train/format_reward/mean": training_episode_data["stats"]["format_reward_mean"],
            "train/format_reward/std": training_episode_data["stats"]["format_reward_std"],
            
            # Equation reward metrics
            "train/equation_reward/mean": training_episode_data["stats"]["equation_reward_mean"],
            "train/equation_reward/std": training_episode_data["stats"]["equation_reward_std"],
            
            # Training metrics
            "train/policy_loss": total_policy_loss / (EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE),
            "train/kl_penalty": total_kl_penalty / (EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE),
            "train/entropy": total_entropy / (EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE),
            "train/grad_norm": grad_norm,
            "train/learning_rate": policy_model.get_lr()[0],
            
            # Timing metrics
            "train/generation_time": generation_time,
            "train/training_time": training_time,
            "train/total_iteration_time": total_iteration_time,
        }
        
        if eval_stats is not None:
            stats.update(eval_stats)
        
        wandb.log(stats)
        
        selected_keys = ["train/reward_mean", "train/format_reward/mean", "train/equation_reward/mean"]
        if iteration % 25 == 0:
            selected_keys.extend(["test/reward_mean", "test/format_reward/mean", "test/equation_reward/mean"])
        selected_stats = {k: stats[k] for k in selected_keys}
        print(f"key stats: {selected_stats}")
            
        if iteration % 1001 == 0:
            save_dir = f"/network/scratch/a/aghajohm/aha_models/r1_aha_moment_{iteration}"
            policy_model.module.save_pretrained(save_dir)
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()



