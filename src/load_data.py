from datasets import load_dataset

# Load Yelp data
def load_yelp_data():
    print("Loading data from Hugging Face datasets...")
    dataset = load_dataset("yelp_review_full")
    return dataset["train"], dataset["test"]

