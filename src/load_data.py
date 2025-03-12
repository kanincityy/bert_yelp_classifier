import pandas as pd

def load_yelp_data():
    splits = {
        'train': 'yelp_review_full/train-00000-of-00001.parquet',
        'test': 'yelp_review_full/test-00000-of-00001.parquet'
    }
    
    print("Loading data...")
    train_data = pd.read_parquet(f"hf://datasets/Yelp/yelp_review_full/{splits['train']}")
    test_data = pd.read_parquet(f"hf://datasets/Yelp/yelp_review_full/{splits['test']}")
    
    print(f"Data loaded: {len(train_data)} training samples, {len(test_data)} test samples.")
    return train_data, test_data