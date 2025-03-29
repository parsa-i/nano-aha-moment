# Nano Aha! Moment: Lunch Break Reproduction of DeepSeek R1-Zero from Scratch  
> Amirhossein Kazemnejad*, Milad Aghajohari*, Aaron Courville, Siva Reddy

Implementation of R1-zero style training with:

- Single 80G GPU
- No RL Library 
- 3B Base Model 
- Full Parameter Tuning 
- Efficient (less than 10h)

Inspired by [TinyZero](https://github.com/Jiayi-Pan/TinyZero) and [Mini-R1](https://www.philschmid.de/mini-deepseek-r1), but designed to be much **simpler**, **cleaner**, and **faster**, with every line of code visible and understandable.

## Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone git@github.com:McGill-NLP/tiny-aha-moment.git
   ```

2. **Install dependencies**  
   First, load the necessay cuda tools:
   ```bash
   module load cudatoolkit/12.5
   ```  
   Next, install torch:  
   ```bash
   pip install torch==2.5
   ```  
   Next, follow the installation guide on the [vllm website](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/index.html) for installing vllm. 
   ```bash
   pip install vllm
   ```  
   Next,
   ```bash
   pip install datasets deepspeed jupyter ipykernel ipywidgets wandb
   ``` 
   Next, install flash attention,
   ```bash
   pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   ``` 
   You should be set.

3. **Run the training script**  
   Open `nano_r1.ipynb` and start training.

## File Descriptions
- `nano_r1.ipynb` is the interactive single file jupyter notebook with tutorial.
- `nano_r1_script.py` is also just the `nano_r1.ipynb` but for convenience of running with python.

## Citation
If you use this codebase in your research, please cite us using:

```bibtex
@misc{Kazemnejad2025:NanoAhaMoment,
  author       = {Amirhossein Kazemnejad and Milad Aghajohari and Aaron Courville and Siva Reddy},
  title        = {Nano Aha! Moment: Lunch Break Reproduction of DeepSeek R1-Zero from Scratch},
  year         = {2025},
  howpublished = {\url{https://github.com/McGill-NLP/nano-aha-moment}},
  note         = {GitHub repository}
}
```