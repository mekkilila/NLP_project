# Sentiment analysis for movie reviews - with modern approach

## Objective

This project replicates and extends the work of [Maas et al. (2011)](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf), who introduced sentiment-aware word vectors trained on a large corpus of IMDb movie reviews. We:

- Replicate classical methods like LSA, LDA, and Word2Vec
- Compare them to contextual embeddings like BERT
- Perform both semantic similarity tests and sentiment classification
- Evaluate model performance using classification accuracy and qualitative insights

---

## Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/mekkilila/NLP_project.git

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the file 
sentiment_notebook.ipynb 

```
> For BERT and PyTorch, make sure to install the appropriate PyTorch version from [https://pytorch.org](https://pytorch.org) depending on your CUDA setup.

---

## Repository Structure

```
├── data_movie/               # IMDb dataset (downloaded or preprocessed)
├── report                    # PDF file containing the final report for this project (NEURIPS style)
├── models/                   # Classification models (e.g., logistic regression, BERT)
├── sentiment_notebook.ipynb  # Jupyter notebook to run to get the entire 
├── tokenizer.py              # Custom tokenizer for the project
├── utils.py                  # Functions useful for visualisations and others
├── requirements.txt          # Required Python packages
└── README.md                 # Project overview (this file)
```

---

## Main Results

### ➤ Semantic Similarity

We tested each embedding model on query words such as `sadness`, `witty`, `dull`, and `romantic`. Results showed:

- **BERT**: Most consistent and meaningful results (e.g., `grief`, `regret`, `compassion` for `sadness`)
- **Word2Vec**: Good coherence and relevance
- **LSA/LDA**: Some noise and abstract associations

### ➤ Sentiment Classification

| Model      | Accuracy (%) | Notes                                       |
|------------|--------------|---------------------------------------------|
| LSA        | 84.3         | Highest accuracy overall                    |
| Word2Vec   | 81.9         | Strong and stable baseline                  |
| LDA        | 80.7         | Slightly lower due to topic focus           |
| BERT       | 82.2         | Trained on smaller subset (1500 samples)    |

Despite using fewer samples, **BERT** achieved performance comparable to LSA, demonstrating the power of pre-trained contextual embeddings.

---

## Future Work

- Implement neural network architectures like **CNNs** or **LSTMs**
- Fine-tune the learned **word embeddings** during classification
- Move from binary classification to **regression**, predicting review scores from **1 to 10** as in the original dataset

---

## Main reference

Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011).  
*Learning Word Vectors for Sentiment Analysis*. ACL 2011.  
🔗 [https://aclanthology.org/P11-1015/](https://aclanthology.org/P11-1015/)

---