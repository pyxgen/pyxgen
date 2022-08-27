# [WORK IN PROGRESS] pyxgen
Pixel art generation from text prompts using AI.

### Pipeline
- Text encoding using [CLIP](https://github.com/openai/CLIP)

### Recommendations
- Python 3.10.6
- Nvidia GPU with CUDA version > 11.3

### Installation
Run
```
pip install -r requirements.txt
```
in a fresh virtual environment. It should install all dependencies, including CLIP and PyTorch.

### Usage
Currently, we can only encode some text using a CLIP text encoder.
To do this, run
```
python example_test_encoding.py "A text to encode" --model="ViT-B/32"
```
in `src/scripts/`. This will encode the text into a feature tensor using a pre-trained CLIP model.

### Contribution
To run the checks locally at each commit, install pre-commit in the repo
```
pre-commit install
```
