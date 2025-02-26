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
   Run the following command:  
   ```bash
   pip install -r requirements.txt
   ```  
   This should work on the Mila cluster. If it fails, manually install the required packages:  
   ```bash
   pip install torch transformers deepspeed sglang
   ```  
   Additionally, follow the installation guide on the [sglang website](https://docs.sglang.ai/start/install.html) for any missing dependencies.

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
```