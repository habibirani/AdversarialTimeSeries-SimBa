# AdversarialTimeSeries-SimBa

This repository contains a modular Python project using PyTorch for demonstrating the effectiveness of adversarial training across different deep learning models and datasets. The project is structured to facilitate easy experimentation with different models, datasets, and adversarial attacks, particularly focusing on the SimBA (Simple Black-box Adversarial Attacks) method. This project is built upon the base paper for the SimBa attack, which provides a foundation for generating adversarial examples in the image domain. You can find the paper [here](https://arxiv.org/abs/1905.07121). Modifications have been made to adapt SimBa for time series data, including adjustments to the perturbation generation process and integration with appropriate evaluation metrics for time series classification tasks. Detailed documentation on these modifications can be found in the attacks.py module.

You can find a demo of the project [here](https://www.youtube.com/watch?v=H9Yp2t74K54&t=36s).

## Structure
- `models.py`: Contains different DL model architectures.
- `attacks.py`: Implementation of various adversarial attack methods such as SibMa, SibMA Temporal.
- `data\`: Data loading utilities.
- `train.py`: Scripts for training models.
- `evaluate.py`: Evaluation of models on clean and adversarial data.
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
## Results and Evaluation
### Clean Data Performance
The models implemented in this repository are evaluated on clean data to establish baseline performance metrics. The evaluation includes metrics such as accuracy, precision, recall, and F1 score, depending on the specific task and dataset.

### Adversarial Robustness
The effectiveness of adversarial training in enhancing model robustness against adversarial attacks is a key focus of this repository. Adversarial examples are generated using the SimBA (Simple Black-box Adversarial Attacks) method and its variations implemented in attacks.py. The trained models are then evaluated on both clean and adversarial data to assess their robustness.


<!-- CONTRIBUTING -->
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


<!-- CITATION -->
## Citation
If you use this repository in your research please use the following BibTeX entry:

```bibtex
@Misc{AdversarialTimeSeries-SimBa,
  title = {Demonstrating the effectiveness of adversarial training across 
           different deep learning models in time series data},
  author = {Irani, Habib},
  howpublished = {Github},
  year = {2024}
  url = {https://github.com/habibirani/AdversarialTimeSeries-SimBa}
}
```
