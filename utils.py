"""
utils.py

Ce module contient un ensemble de fonctions utiles dans la partie 
principale du projet, notamment pour générer des visualisations. 
"""

from collections import Counter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np
import pandas as pd


def plot_wordclouds_by_label(
    texts, labels, tokenizer, label_names=("Negative", "Positive"), stop_words=None
):
    """
    Generates and plots word clouds separately for each label.

    Parameters
    ----------
    texts : list of str
        List of text reviews.
    labels : list of int
        List of labels corresponding to each review (typically 0 or 1).
    tokenizer : object
        Tokenizer with a `.tokenize(text)` method to split texts into tokens.
    label_names : tuple or dict, optional
        Names for each label to display in the titles (default is ("0", "1")).
    stop_words : set of str, optional
        Set of words to exclude from the word cloud 
        (default excludes {"br", "the", "and", "is", "it", "to", "of"}).

    Returns
    -------
    None
        Displays matplotlib figures with word clouds for each label.
    """
    if stop_words is None:
        stop_words = {"br", "the", "and", "is", "it", "to", "of"}

    for lbl in sorted(set(labels)):
        label_texts = [texts[i] for i in range(len(labels)) if labels[i] == lbl]

        # Tokenize and filter words
        flat_list = [word for text in label_texts for word in tokenizer.tokenize(text)]
        filtered_words = [word for word in flat_list if word.lower() not in stop_words]
        filtered_text = " ".join(filtered_words)

        # Generate word cloud
        cloud = WordCloud(width=800, height=300, background_color="white").generate(
            filtered_text
        )

        # Plot
        plt.figure(figsize=(10, 5))
        plt.imshow(cloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(
            (
                f"Word Cloud for Label {lbl} "
                f"({label_names[lbl] if isinstance(label_names, dict) else lbl})"
            )
        )
        plt.show()


def plot_review_length_distribution_plotly(
    train_data, tokenizer, max_length=1200, bin_size=100
):
    """
    Generates and plots a grouped bar chart showing the distribution of review lengths
    for two different labels (e.g., positive and negative) using Plotly.

    Parameters
    ----------
    train_texts : list of str
        List of text reviews.
    train_labels : list of int
        List of labels corresponding to each review (typically 0 or 1).
    tokenizer : object
        Tokenizer with a `.tokenize(text)` method to split texts into tokens.
    max_length : int, optional
        Maximum length (in number of tokens) to consider for the x-axis (default is 1200).
    bin_size : int, optional
        Size of each bin for grouping review lengths (default is 100).

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure displaying the grouped bar chart.
    """
    train_texts = [sample["text"] for sample in train_data]
    train_labels = [sample["label"] for sample in train_data]
    review_lengths = [len(tokenizer.tokenize(text)) for text in train_texts]

    lengths_label_0 = [
        review_lengths[i] for i in range(len(train_labels)) if train_labels[i] == 0
    ]
    lengths_label_1 = [
        review_lengths[i] for i in range(len(train_labels)) if train_labels[i] == 1
    ]

    bins = np.arange(0, max_length + bin_size, bin_size)
    counts_0, _ = np.histogram(lengths_label_0, bins)  # number of reviews in each bin
    counts_1, _ = np.histogram(lengths_label_1, bins)

    bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]

    fig = go.Figure(
        data=[
            go.Bar(
                name="Label 0 (Negative)",
                x=bin_labels,
                y=counts_0,
                marker_color="mediumpurple",
            ),
            go.Bar(
                name="Label 1 (Positive)",
                x=bin_labels,
                y=counts_1,
                marker_color="orange",
            ),
        ]
    )

    # Custom layout
    fig.update_layout(
        title="Distribution of Review Lengths by Label",
        xaxis_title="Length of Review (Number of Tokens)",
        yaxis_title="Frequency",
        barmode="group",
        bargap=0.15,
        bargroupgap=0.1,
        width=1000,
        height=600,
    )

    fig.show()


def plot_text_length_histograms(
    train_data, test_data, tokenizer, max_length=1200, bin_size=100
):
    """
    Plots side-by-side histograms of text lengths for training and test sets using Plotly,
    with a maximum length filter and a custom bin size.

    Parameters
    ----------
    train_data : list of dict
        List of training samples, each sample being a dictionary with 'text' and 'label' keys.
    test_data : list of dict
        List of test samples, each sample being a dictionary with 'text' and 'label' keys.
    tokenizer : object
        Tokenizer with a `.tokenize(text)` method to split texts into tokens.
    max_length : int, optional
        Maximum length (in number of tokens) to consider for the x-axis (default is 1200).
    bin_size : int, optional
        Size of each bin for grouping review lengths (default is 100).

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure containing the grouped histograms.
    """
    train_texts = [sample["text"] for sample in train_data]
    test_texts = [sample["text"] for sample in test_data]

    train_lengths = [len(tokenizer.tokenize(text)) for text in train_texts]
    test_lengths = [len(tokenizer.tokenize(text)) for text in test_texts]

    train_lengths_filtered = [
        length for length in train_lengths if length <= max_length
    ]
    test_lengths_filtered = [length for length in test_lengths if length <= max_length]

    bins = np.arange(0, max_length + bin_size, bin_size)

    train_counts, _ = np.histogram(train_lengths_filtered, bins)
    test_counts, _ = np.histogram(test_lengths_filtered, bins)

    bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]

    fig = go.Figure(
        data=[
            go.Bar(
                name="Train",
                x=bin_labels,
                y=train_counts,
                marker_color="blue",
                opacity=0.6,
            ),
            go.Bar(
                name="Test",
                x=bin_labels,
                y=test_counts,
                marker_color="orange",
                opacity=0.6,
            ),
        ]
    )

    # Update the layout of the figure
    fig.update_layout(
        title="Histogram of Text Lengths (Tokens) - Filtered by Max Length",
        xaxis_title="Length of Text (Number of Tokens)",
        yaxis_title="Frequency",
        barmode="group",
        bargap=0.2,
        width=1000,
        height=600,
    )

    fig.show()


def describe_text_data(train_data, test_data, tokenizer):
    """
    Computes descriptive statistics for text datasets and returns a summary DataFrame.
    """
    train_texts = [sample["text"] for sample in train_data]
    train_labels = [sample["label"] for sample in train_data]
    test_texts = [sample["text"] for sample in test_data]
    test_labels = [sample["label"] for sample in test_data]

    tokenizer.build_vocab(train_texts + test_texts)

    train_lengths = [len(tokenizer.tokenize(text)) for text in train_texts]
    test_lengths = [len(tokenizer.tokenize(text)) for text in test_texts]
    train_label_counts = Counter(train_labels)
    test_label_counts = Counter(test_labels)

    # Prepare summary
    summary = {
        "Set": ["Train", "Test"],
        "Size": [len(train_texts), len(test_texts)],
        "Label 0 (%)": [
            train_label_counts.get(0, 0) / len(train_labels) * 100,
            test_label_counts.get(0, 0) / len(test_labels) * 100,
        ],
        "Label 1 (%)": [
            train_label_counts.get(1, 0) / len(train_labels) * 100,
            test_label_counts.get(1, 0) / len(test_labels) * 100,
        ],
        "Avg Length": [np.mean(train_lengths), np.mean(test_lengths)],
        "Min Length": [np.min(train_lengths), np.min(test_lengths)],
        "Max Length": [np.max(train_lengths), np.max(test_lengths)],
    }

    summary_df = pd.DataFrame(summary)

    return summary_df


def longest_and_shortest_reviews(train_data, tokenizer, num_reviews=5):
    """
    Prints the shortest and longest reviews based on token length.

    Parameters
    ----------
    train_data : list of dict
        List of training data where each entry contains 'text' and 'label'.
    tokenizer : object
        Tokenizer with a `.tokenize(text)` method to split texts into tokens.
    num_reviews : int, optional
        Number of shortest and longest reviews to display (default is 5).

    Returns
    -------
    None
    """
    # Extract texts from train data
    train_texts = [sample["text"] for sample in train_data]

    # Calculate token lengths
    train_lengths = [len(tokenizer.tokenize(text)) for text in train_texts]

    # Pair text lengths with original texts
    length_text_pairs = list(zip(train_lengths, train_texts))

    # Sort reviews by token length
    length_text_pairs.sort()

    # Print shortest reviews
    print(f"\n{num_reviews} Shortest Reviews:\n")
    for length, text in length_text_pairs[:num_reviews]:
        print(f"Length: {length} tokens")
        print(text[:300], "...\n")

    # Print longest reviews
    print(f"\n{num_reviews} Longest Reviews:\n")
    for length, text in length_text_pairs[-num_reviews:]:
        print(f"Length: {length} tokens")
        print(text[:500], "...\n")


def longest_and_shortest_reviews_by_sentiment(train_data, tokenizer, num_reviews=5):
    """
    Returns the longest and shortest reviews, categorized by sentiment (positive/negative).
    
    Parameters
    ----------
    train_data : list of dict
        List of training data where each entry contains 'text' and 'label'.
    tokenizer : object
        Tokenizer with a `.tokenize(text)` method to split texts into tokens.
    num_reviews : int, optional
        Number of shortest and longest reviews to display (default is 5).
        
    Returns
    -------
    dict
        A dictionary with keys 'positive_shortest', 'negative_shortest', 'positive_longest', 'negative_longest'.
    """
    train_texts = [sample["text"] for sample in train_data]
    train_labels = [sample["label"] for sample in train_data]  # 0 for negative, 1 for positive
    
    train_lengths = [len(tokenizer.tokenize(text)) for text in train_texts]

    length_text_pairs = list(zip(train_lengths, train_texts, train_labels))


    length_text_pairs.sort()

    # Separate reviews 
    shortest_reviews_positive = []
    shortest_reviews_negative = []
    longest_reviews_positive = []
    longest_reviews_negative = []

    # Categorize reviews
    for length, text, label in length_text_pairs[:num_reviews]:  # Shortest reviews
        if label == 0:
            shortest_reviews_negative.append((length, text))
        else:
            shortest_reviews_positive.append((length, text))

    for length, text, label in length_text_pairs[-num_reviews:]:  # Longest reviews
        if label == 0:
            longest_reviews_negative.append((length, text))
        else:
            longest_reviews_positive.append((length, text))

    # Prepare result dictionary with the reviews
    result = {
        "positive_shortest": shortest_reviews_positive,
        "negative_shortest": shortest_reviews_negative,
        "positive_longest": longest_reviews_positive,
        "negative_longest": longest_reviews_negative
    }

    
    print(f"\nShortest {num_reviews} Positive Reviews:")
    for length, text in shortest_reviews_positive:
        print(f"Length: {length} tokens\n{text[:300]}...\n")

    print(f"\nShortest {num_reviews} Negative Reviews:")
    for length, text in shortest_reviews_negative:
        print(f"Length: {length} tokens\n{text[:300]}...\n")

    print(f"\nLongest {num_reviews} Positive Reviews:")
    for length, text in longest_reviews_positive:
        print(f"Length: {length} tokens\n{text[:500]}...\n")

    print(f"\nLongest {num_reviews} Negative Reviews:")
    for length, text in longest_reviews_negative:
        print(f"Length: {length} tokens\n{text[:500]}...\n")

    return result


def get_extreme_reviews_by_sentiment(train_data, tokenizer, num_reviews=5):
    """
    Returns the shortest and longest reviews by sentiment (positive and negative).

    Parameters
    ----------
    train_data : list of dict
        Each dict must contain 'text' and 'label' (0 = negative, 1 = positive).
    tokenizer : object
        HuggingFace tokenizer with a .tokenize() method.
    num_reviews : int
        Number of shortest/longest reviews to return per class.

    Returns
    -------
    Dict with keys:
        - 'shortest_negative'
        - 'shortest_positive'
        - 'longest_negative'
        - 'longest_positive'
      Each maps to a list of tuples: (token_length, review_text)
    """
    # Separate reviews by label
    negative_reviews = [
        (len(tokenizer.tokenize(sample['text'])), sample['text'])
        for sample in train_data if sample['label'] == 0
    ]
    positive_reviews = [
        (len(tokenizer.tokenize(sample['text'])), sample['text'])
        for sample in train_data if sample['label'] == 1
    ]

    # Sort each group by token length
    negative_reviews.sort(key=lambda x: x[0])
    positive_reviews.sort(key=lambda x: x[0])

    return {
        'shortest_negative': negative_reviews[:num_reviews],
        'longest_negative': negative_reviews[-num_reviews:],
        'shortest_positive': positive_reviews[:num_reviews],
        'longest_positive': positive_reviews[-num_reviews:]
    }
