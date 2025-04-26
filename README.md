# BERT Fine-Tuning for Yelp Reviews Sentiment Analysis (PyTorch & Hugging Face)

This repository demonstrates fine-tuning a pre-trained BERT model (`bert-base-uncased`) using PyTorch and Hugging Face libraries (`transformers`, `datasets`) for sentiment analysis on the **Yelp Reviews Full dataset**. The goal is to classify reviews based on their star ratings.

---

### ✨ Key Features & What This Project Demonstrates

*   **End-to-End Fine-Tuning Pipeline:** Complete workflow from data loading to model evaluation.
*   **Hugging Face Integration:** Leverages `datasets` for easy data access and `transformers` for pre-trained models and tokenizers.
*   **Custom PyTorch Implementation:** Includes a custom `BertYelpDataset` (`dataset.py`) for optimised data handling and a bespoke `BertYelpModel` (`model.py`) adapting BERT for sequence classification.
*   **Structured Training:** Utilises a `Trainer` class (`trainer.py`) to manage the training loop, validation, optimization, and checkpointing.
*   **Checkpointing:** Implements saving and loading of model checkpoints for robust training and resuming capabilities.

### 🛠️ Technologies Used

*   Python 3.7+
*   PyTorch
*   Hugging Face `transformers`
*   Hugging Face `datasets`
*   Tqdm (for progress bars)

### ⚙️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-url]
    cd [your-repo-directory]
    ```
2.  **Install requirements:**
    ```bash
    pip install torch transformers datasets tqdm
    # Or install from the requirements file:
    # pip install -r requirements.txt
    ```

### ▶️ Usage

Run the main training script from the **root** directory of the project:

```bash
python src/main.py
```

This script will perform the following steps:

1.  Download and preprocess the Yelp Reviews Full dataset.
2.  Load the `bert-base-uncased` tokenizer and the custom `BertYelpModel`.
3.  Set up DataLoaders using the custom `BertYelpDataset`.
4.  Instantiate the `Trainer`.
5.  Fine-tune the model for a set number of epochs (default: 3).
6.  Save model checkpoints periodically to the `checkpoints/` directory.
7.  Evaluate the final model on the test set and print the accuracy.
8.  Save the final trained model weights to `bert_yelp_model.pth`.

**Resuming Training:**
To resume from a saved checkpoint, modify `main.py` to call `trainer.load_checkpoint('path/to/your/checkpoint.pth')` before starting the training loop.

### 🎛️ Configuration

Key training parameters can be adjusted within `main.py`:

*   `epochs`: Number of training epochs.
*   `batch_size`: Batch size for training and evaluation.
*   `learning_rate`: Learning rate for the AdamW optimizer.
*   `checkpoint_interval`: Frequency (in epochs) for saving checkpoints.

### 📁 Project Structure

```
.
├── data/ # Raw data files (if included locally) or data processing scripts
├── src/ # Source code directory
│ ├── dataset.py # Custom PyTorch Dataset class
│ ├── model.py # Custom PyTorch BERT model class
│ ├── trainer.py # Training and evaluation logic class
│ └── main.py # Main script to run training & evaluation
├── checkpoints/ # Directory for saved model checkpoints (created during training)
├── requirements.txt # Project dependencies
├── bert_yelp_model.pth # Saved final trained model (output of main.py)
├── LICENSE # MIT License file
└── README.md # This file
```

### 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Happy coding! ✨🐇
