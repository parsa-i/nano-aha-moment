Here's your GitHub markdown with smoother phrasing while keeping it clear and concise:

```markdown
# Recreating R1's Aha Moment
Inspired by [TinyZero](https://github.com/Jiayi-Pan/TinyZero), but designed to be 10X simpler, cleaner, and faster.

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
   pip install torch
   ```  
   Next, follow the installation guide on the [sglang website](https://docs.sglang.ai/start/install.html) for installing sglang. 
   ```bash
   pip install "sglang[all]>=0.4.3.post2" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
   ```  
   Next,
   ```bash
   pip install transformers deepspeed jupyter wandb
   ``` 
   Next, install flash attention,
   ```bash
   pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   ``` 
   You should be set.
    

3. **Start an interactive job on the cluster**  
   Request resources using:  
   ```bash
   salloc --partition=unkillable --gres=gpu:a100l:1 -c 6 --mem=32G -t 12:00:00
   ```  
   Then, connect via VS Code or Cursor.

4. **Run the training script**  
   Open `r1.ipynb`, set `CUDA_HOME` and `HF_HOME` as needed, and start training.

5. **Install VS Code extensions**  
   Make sure to install the **Jupyter** and **Python** extensions in VS Code for a smoother experience.

## File Descriptions
`r1_gold.ipynb` is the ground truth implementation. `r1_todo.ipynb` misses some components and you need to fill those without looking at the `r1_gold.ipynb`. `r1_script.py` is also just the `r1_gold` but for convenience of running with python.
```