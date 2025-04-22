"""
utils.py

Ce module contient un ensemble de fonctions utiles dans la partie principale du projet, notamment pour générer des visualisations. 
"""

from collections import Counter
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud

def make_wordcloud(data, col):
    """
    Generates and plots a word cloud from a specified column of a DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data.
    col : str
        Column containing text data (e.g., delimited by semicolons).

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure with the word cloud.
    collections.Counter
        Counter object with word frequencies.
    """
    tokens = [str(cat).split(";") for cat in data[col]]
    item_all = [item for sublist in tokens for item in sublist]  # découpage en mots
    item_count = Counter(item_all)  # décompte occurrences

    # Création d'un nuage de mots
    wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate_from_frequencies(item_count)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    return fig, item_count
