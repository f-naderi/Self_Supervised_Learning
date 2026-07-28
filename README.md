```markdown
# Self-Supervised Learning with SimCLR

A TensorFlow implementation of SimCLR (Simple Framework for Contrastive Learning of Visual Representations) using a custom ResNet-18 encoder on the CIFAR-10 dataset.


## Features


Self-Supervised Pretraining

- data augmentation pipeline
- ResNet-18 encoder
- MLP projection head
- NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
- Mixed Precision Training
- Automatic checkpointing and best model selection
- Encoder export

Linear Evaluation

- Frozen pretrained encoder
- Linear classification head
- Linear evaluation on CIFAR-10 
- Classification report
- Training history
- Accuracy and loss visualization

## Project Structure


Self_Supervised_Learning/
│
├── src/                                  # Source code
│   ├── data_loader.py                    # Loads and preprocesses the CIFAR-10 dataset
│   ├── augmentations.py                  # SimCLR data augmentation pipeline
│   ├── model.py                          # ResNet-18 encoder, projection head, and SimCLR model
│   ├── loss.py                           # NT-Xent contrastive loss implementation
│   ├── train.py                          # Self-supervised SimCLR training script
│   ├── linear_eval.py                    # Linear evaluation using a frozen encoder
│   └── predict.py                        # Single image prediction
│
├── data/                                 # CIFAR-10 dataset
│
├── saved_models/                         # Trained SimCLR models
│   ├── simclr.keras                      # Complete SimCLR model (Encoder + Projection Head)
│   ├── projection_head.keras             # Trained projection head (MLP)
│   └── encoder.keras                     # Trained feature encoder (used for downstream tasks)
│
├── linear_probe/                         # Linear evaluation models
│   ├── best.keras                        # Best linear classifier based on validation accuracy
│   └── linear_probe.keras                # Final trained linear probe model
│
├── results/                              # Evaluation results and training logs
│   ├── training_curve.png                # Accuracy and loss curves
│   ├── history.csv                       # Training history for the linear probe
│   ├── classification_report.txt         # Precision, recall, and F1-score
│   └── linear_eval_results.txt           # Final evaluation metrics
│
├── checkpoints/                          # SimCLR training checkpoints
│
├── SimCLR.ipynb                          # Main Jupyter notebook
│
└── requirements.txt                      # Python package dependencies



## Installation

```bash
https://github.com/f-naderi/Self_Supervised_Learning.git
cd Self_Supervised_Learning
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset

* CIFAR-10
* 50,000 training images
* 10,000 test images
* 10 object categories
* Image size: 32×32

## Requirements

* Python 3.10+
* TensorFlow
* NumPy
* Matplotlib
* Scikit-Learn
* Pandas
* tqdm
* Jupyter Notebook


## References

* Chen et al. A Simple Framework for Contrastive Learning of Visual Representations (SimCLR), ICML 2020.
* He et al. Deep Residual Learning for Image Recognition, CVPR 2016.
* Krizhevsky. CIFAR-10 Dataset.

```


