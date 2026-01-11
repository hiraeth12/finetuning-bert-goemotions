# Fine-Tuning BERT on GoEmotions Dataset

## 👥 Identification

### Group 3 Information
- **Sahrul Ridho Firdaus**: 1103223009
- **Rayhan Diff**: 1103220039



## 📚 Repository Purpose

This repository contains a comprehensive implementation of fine-tuning BERT models for multi-label emotion classification using the GoEmotions dataset. The project demonstrates advanced natural language processing (NLP) techniques for detecting multiple emotions simultaneously in text data.

## 🎯 Project Overview

The GoEmotions dataset is a corpus of carefully curated English Reddit comments, labeled with 28 fine-grained emotion categories. This project fine-tunes pre-trained BERT models to perform multi-label classification, enabling the model to predict multiple emotions present in a single piece of text.

### Key Features:
- **Multi-label Classification**: Predicts multiple emotions simultaneously (e.g., a text can be both "joy" and "gratitude")
- **State-of-the-art Transformers**: Utilizes BERT-base-uncased architecture
- **Comprehensive Evaluation**: Implements F1-Macro score and accuracy metrics
- **Optimized Training**: Includes learning rate scheduling, warmup, and mixed precision training (FP16)

## 🤖 Models and Performance

### Model Architecture
- **Base Model**: `bert-base-uncased` (110M parameters)
- **Alternative Tested**: `distilbert-base-uncased` (66M parameters)
- **Classification Type**: Multi-label (28 emotion categories)
- **Problem Type**: Multi-label classification with sigmoid activation

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Learning Rate | 3e-5 |
| Batch Size | 32 |
| Epochs | 5 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.1 |
| LR Scheduler | Cosine |
| Max Sequence Length | 128 |
| Precision | FP16 (Mixed Precision) |

### Evaluation Metrics

The model is evaluated using:
1. **F1-Macro Score**: Calculates F1 score for each label independently and averages them (treats all emotions equally)
2. **Exact Match Accuracy**: Measures when all predicted labels exactly match the ground truth
3. **Prediction Threshold**: 0.3 (probability threshold for multi-label prediction)

### Emotion Categories (28 Labels)

The GoEmotions dataset includes the following emotion categories:
- **Positive Emotions**: admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief
- **Negative Emotions**: anger, annoyance, confusion, curiosity, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness
- **Ambiguous/Neutral**: realization, surprise, neutral

## 📂 Repository Structure

```
finetuning-bert-goemotions/
│
├── finetuning-bert-goemotions.ipynb    # Main training notebook
├── README.md                            # This file
├── requirements.txt                     # Python dependencies
├── goemotions_bert_base/               # Training checkpoints (generated)
└── final_goemotions_bert/              # Final trained model (generated)
```

## 🚀 How to Navigate

### Prerequisites

#### Installation
Install all required dependencies using the requirements file:
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install transformers datasets evaluate accelerate scikit-learn -U
```

#### Download Pre-trained Models
You can download the pre-trained models from Google Drive:

🔗 **[Download Models from Google Drive](https://drive.google.com/drive/folders/1c36FAYeuR8H2E7bFMKA7txNGLJ5ktDgF?usp=sharing)**

After downloading, extract the model files to the project directory.

### Running the Notebook

1. **Setup & Installation** (Cells 1-3)
   - Install required dependencies
   - Import necessary libraries
   - Configure GPU settings

2. **Data Loading** (Cells 4-5)
   - Load GoEmotions dataset from Hugging Face
   - Inspect dataset structure and labels
   - Verify 28 emotion categories

3. **Data Preprocessing** (Cells 6-7)
   - Tokenize text using BERT tokenizer
   - Convert labels to multi-hot encoded vectors (float32)
   - Prepare dataset in PyTorch format

4. **Model Training** (Cells 8-9)
   - Initialize BERT model with multi-label classification head
   - Configure training arguments
   - Train for 5 epochs with evaluation
   - Save best model based on F1-Macro score

5. **Model Testing** (Cells 10-11)
   - Load trained model
   - Test with custom text examples
   - Predict multiple emotions with confidence scores

### Expected Workflow
```python
# 1. Install dependencies
!pip install transformers datasets evaluate accelerate scikit-learn -U

# 2. Run all cells sequentially
# 3. Training takes ~30-45 minutes on GPU
# 4. Test with your own sentences in the last cell
```

## 💡 Usage Example

After training, you can use the model to predict emotions:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model = AutoModelForSequenceClassification.from_pretrained("./final_goemotions_bert")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Predict emotions
text = "I am so happy that I passed the exam with perfect score!"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.sigmoid(logits[0])
    
# Get emotions with probability > 0.3
predicted_emotions = [(labels_list[i], probs[i].item()) 
                     for i in range(len(probs)) if probs[i] > 0.3]
print(predicted_emotions)
# Output: [('joy', 0.89), ('excitement', 0.72)]
```

## 🔬 Technical Details

### Multi-Label Classification Approach
- **Activation Function**: Sigmoid (allows multiple independent probabilities)
- **Loss Function**: Binary Cross-Entropy (BCEWithLogitsLoss)
- **Label Encoding**: Multi-hot vectors (each emotion is independent)

### Optimization Techniques
1. **Warmup**: Gradual learning rate increase in first 10% of training
2. **Cosine Decay**: Smooth learning rate reduction
3. **Mixed Precision (FP16)**: Faster training with lower memory usage
4. **Gradient Accumulation**: Effective larger batch sizes

## 📊 Results Interpretation

The model outputs probabilities for each of the 28 emotions:
- **Threshold 0.3**: A probability above 0.3 indicates the emotion is present
- **Multiple Predictions**: One text can have multiple emotions (multi-label)
- **F1-Macro**: Balanced measure across all emotion categories

## 📦 Requirements

The project dependencies are listed in `requirements.txt`:

```txt
transformers>=4.30.0
datasets>=2.14.0
evaluate>=0.4.0
accelerate>=0.20.0
scikit-learn>=1.3.0
torch>=2.0.0
numpy>=1.24.0
```

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## 📖 References

- **Dataset**: [GoEmotions: A Dataset of Fine-Grained Emotions](https://arxiv.org/abs/2005.00547)
- **Model**: [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- **Framework**: [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- **Pre-trained Models**: [Google Drive Repository](https://drive.google.com/drive/folders/1c36FAYeuR8H2E7bFMKA7txNGLJ5ktDgF?usp=sharing)

---


## 📝 Notes

- Training requires a CUDA-capable GPU for optimal performance
- The notebook disables Weights & Biases (wandb) logging by default
- Model checkpoints are saved after each epoch
- Best model is selected based on F1-Macro score on test set

## 🤝 Contributing

This project is part of a Deep Learning Final Exam. For questions or improvements, please contact the repository owner.

---

**Last Updated**: January 2026  
