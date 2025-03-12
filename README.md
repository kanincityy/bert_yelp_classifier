# BERT Fine-Tuning for Yelp Reviews Sentiment Analysis
This repository contains code for fine-tuning a pre-trained BERT model to perform sentiment analysis on the Yelp Reviews Full dataset.

## Overview
This project demonstrates how to:

* Load and preprocess the Yelp Reviews Full dataset using the Hugging Face `datasets` library.
* Fine-tune a pre-trained BERT model (`bert-base-uncased`) for sequence classification using PyTorch and the `transformers` library.
* Implement a custom `Dataset` class for efficient data loading and batching.
* Create a `Trainer` class for streamlined training and evaluation.
* Implement checkpoint saving and loading for robust training.

## Requirements

* Python 3.7+
* PyTorch
* Transformers (`transformers`)
* Datasets (`datasets`)
* Tqdm

You can install the required packages using pip:

```bash
pip install torch transformers datasets tqdm
```

## Dataset

The project uses the Yelp Reviews Full dataset from the Hugging Face Datasets library. This dataset contains full reviews and star ratings from Yelp.

## Files

* `main.py`: The main script for training and evaluating the model.
* `dataset.py`: Contains the custom `BertYelpDataset` class.
* `model.py`: Contains the `BertYelpModel` class.
* `trainer.py`: Contains the `Trainer` class for training and evaluation.
* `checkpoints/`: (Directory) Contains saved model checkpoints.
* `bert_yelp_model.pth`: The final trained model.
* `requirements.txt`: All necessary imports to make running the code easier :)
* `README.md`: This file.

## Usage

1.  **Clone the repository:**

     ```bash
    git clone [your-repo-url]
    cd [your-repo-directory]
    ```

2.  **Install the requirements:**

     ```bash
    pip install torch transformers datasets tqdm
     ```

3.  **Run the main script:**

     ```bash
     python src/main.py
     ```

#     This will:

     * Download the Yelp Reviews Full dataset.
     * Load the `bert-base-uncased` tokenizer and model.
     * Create `DataLoader` instances.
     * Fine-tune the model for 3 epochs (default).
     * Evaluate the model on the test set.
     * Save the trained model to `bert_yelp_model.pth`.
     * Save checkpoints to the `checkpoints` folder.

 4.  **To resume training from a checkpoint:**

     Modify `main.py` to load the checkpoint using the `trainer.load_checkpoint()` method.

 ## Training Parameters

 You can modify the following parameters in `main.py`:

 * `epochs`: The number of training epochs.
 * `batch_size`: The batch size for training and evaluation.
 * `learning_rate`: The learning rate for the AdamW optimizer.
 * `checkpoint_interval`: how many epochs to save checkpoints.

 ## Model Evaluation

 After training, the script will print the evaluation accuracy on the test set. The trained model is saved to `bert_yelp_model.pth`.

 ## Checkpoints

 The `checkpoints` directory will contain saved model checkpoints at specified intervals. These checkpoints can be used to resume training or to evaluate the model at different stages of training.

 ## Contributing 

This project is a reflection of my learning, but feel free to fork the repository and contribute if you have ideas or improvements!

## License 

This repository is licensed under the MIT License. See the LICENSE file for details.

---

Happy coding! ✨🐇
