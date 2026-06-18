
# Word2Vec from Scratch — AG News

This is a Skip-gram Word2Vec project built from scratch using TensorFlow.
Trained on a sample of the AG News dataset as part of a word embeddings assignment.

## Dataset

**AG News** — a news topic classification dataset with 4 categories:
World, Sports, Business, and Sci/Tech.

- Source: [fancyzhx/ag_news](https://huggingface.co/datasets/TheFactoryX/edition_0000_fancyzhx-ag_news-readymade) on Hugging Face
- We sampled 400 articles per category = 1600 total
- Only the raw `text` field is used, no labels

The dataset was chosen because it covers clearly different topics,
which makes it easier to see if the model actually learned something —
sports words should cluster away from business words and so on.

## How to run

1. Open `Word2Vec.ipynb` in Google Colab
2. Run the first cell to install dependencies:
   `!pip install -U datasets huggingface_hub tensorflow scikit-learn matplotlib`
3. Run all cells from top to bottom

No local files needed — the dataset loads directly from Hugging Face.

## Model details

- Architecture: Skip-gram (center word → predict context words)
- Embedding dimension: 300
- Window size: 2
- Optimizer: Adam (lr = 0.01)
- Epochs: 10
- Batch size: 256
- Loss: SparseCategoricalCrossentropy (from logits)

## What the notebook covers

1. Load raw untokenized text from AG News
2. Clean and tokenize manually (lowercase, remove punctuation)
3. Build vocabulary and filter rare words (min count = 2)
4. Generate skip-gram (center, context) training pairs
5. Train the model and plot the loss curve
6. Extract the embedding matrix (W_in)
7. Find similar words using cosine similarity
8. Visualize word clusters using t-SNE

## Reference

Mikolov et al. (2013) — *Efficient Estimation of Word Representations in Vector Space*
https://arxiv.org/abs/1301.3781
