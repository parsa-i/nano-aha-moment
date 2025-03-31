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
   git clone git@github.com:McGill-NLP/nano-aha-moment.git
   ```

2. **Install dependencies**  
   First, make sure cuda 12.4 is installed.
   Next, install torch:  
   ```bash
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
   ```  
   Next, install the rest of the dependencies:
   ```bash
   pip install -r requirements.txt
   ``` 
   
   You should be set.

3. **Run the training script**  
   Open `nano_r1.ipynb` or `nano_r1_script.py` and start training.

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