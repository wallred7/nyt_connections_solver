# NYT Connections Solver

## Overview

This project aimed to develop a lightweight solution for solving the NYT Connections game using clustering and word embeddings. The core idea was to leverage pre-trained GloVe embeddings to represent words as vectors and then apply clustering algorithms to group related words.

## Approach

The implemented approach involved the following steps:

1. **Data Preprocessing:**
   - Fetching NYT Connections game data (e.g., connection groups).
   - Preprocessing the data, including cleaning and tokenization.

2. **Word Embeddings:**
   - Utilizing pre-trained GloVe embeddings to transform words into numerical vectors.

3. **Clustering:**
   - Applying clustering algorithms (e.g., k-means) to group words based on their embeddings' similarity.

## Challenges and Outcomes

Unfortunately, the initial approach did not yield satisfactory results. The clustering algorithm struggled to accurately group words into their respective connection groups. This is likely due to the fact that the overall meaning captured by the GloVe embeddings did not align well with the specific semantic relationships required to solve the Connections game. For example, words with multiple meanings might have been clustered based on a dominant meaning that was irrelevant to the game's context.

## Potential Improvements

Several potential improvements could be explored to enhance the solver's performance:

- **Alternative Embedding Comparison:** Experiment with different distance metrics or similarity measures to compare word embeddings, potentially capturing more nuanced relationships between words.
- **Different Embedding Types:** Explore alternative embedding models, such as BERT or Word2Vec, which might provide better representations for the specific task of solving Connections.
- **Contextual Embeddings:** Consider using contextual embeddings trained on large corpus, which take into account the surrounding words in a sentence or phrase, to capture vaied word meanings.

## Lessons Learned

A key takeaway from this project is the importance of validating an approach before investing heavily in production-ready code. In this case, building the codebase before thoroughly testing the core clustering and embedding approach led to wasted effort when the initial method proved ineffective. It took the wind out of my motivational sails.

## Future Directions

While the current implementation did not achieve the desired accuracy, the project provides a foundation for further exploration. Future work could focus on addressing the challenges mentioned above and experimenting with alternative approaches to improve the solver's performance.

## Getting Started

**Prerequisites:**

- Python 3.9
- Poetry

**Installation:**

```bash
poetry install
```

**Running Experiments:**

The code for running experiments and evaluating the model is located in the `src/model` directory. Refer to the individual files for specific instructions on running experiments and analyzing results.

**Note:** This project is currently in an experimental stage and does not provide a fully functional Connections solver.
