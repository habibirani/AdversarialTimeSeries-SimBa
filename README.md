# Adversarial Training Demonstrator

This repository contains a modular Python project using PyTorch for demonstrating the effectiveness of adversarial training across different deep learning models and datasets. The project is structured to facilitate easy experimentation with different models, datasets, and adversarial attacks, particularly focusing on the SimBA (Simple Black-box Adversarial Attacks) method.

## Structure
- `models/`: Contains different model architectures.
- `attacks/`: Implementation of various adversarial attack methods.
- `data/`: Data loading utilities.
- `training/`: Scripts for training models.
- `evaluation/`: Evaluation of models on clean and adversarial data.
- `main.py`: Main script to run experiments.

## Installation

To set up the project, you need to have Python and PyTorch installed. Clone the repository and install the required packages:

```bash
git clone https://github.com/Habibirani/AdversarialTimeSeries-SimBa.git
cd AdversarialTimeSeries-SimBa
conda env create -f environment.yml

```

## Dependencies
- Python 3.10
- TensorFlow 2.x
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Pandas
- Scikit-learn
- torchvision
- PyTorch


## Usage

To use this project, run the main script after installation:

```bash
python main.py

```



<!-- CONTRIBUTING -->
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

