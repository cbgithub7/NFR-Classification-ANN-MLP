# Natural Language Processing with Neural Networks

This project utilizes neural networks for text classification, specifically focusing on categorizing sentences into different classes based on their content. The code is implemented in a Jupyter notebook environment.

## Introduction

The purpose of this project is to demonstrate the application of neural networks in natural language processing tasks, particularly in text classification. The notebook contains code snippets for data preprocessing, model training, cross-validation, and evaluation metrics calculation.

## Requirements

To run the code in this notebook, the following libraries are required:

- `nltk`: Natural Language Toolkit for text processing
- `sklearn`: Scikit-learn library for machine learning algorithms
- `numpy`: Numerical computing library for array operations
- `json`: Library for JSON data manipulation
- `datetime`: Library for handling date and time operations

## Dataset

The training data consists of sentences categorized into four classes:

1. Performance
2. Usability
3. Security
4. Operability

Each sentence is associated with a class label and a class name.

## Implementation Details

The notebook is divided into sections covering different aspects of the implementation:

1. **Data Preprocessing**: Tokenization, stemming, and feature extraction from the text data.
2. **Model Training**: Training a neural network model using backpropagation and gradient descent.
3. **Cross-Validation**: Implementing stratified k-fold cross-validation for model evaluation.
4. **Evaluation Metrics**: Calculating accuracy, precision, recall, and F1-score for model performance assessment.
5. **Mean Average Calculation**: Computing the mean average of model scores across multiple cross-validation folds.

## Usage

To use this notebook:

1. Install the required libraries mentioned in the "Requirements" section.
2. Load the notebook in a Jupyter environment.
3. Execute the code cells sequentially.

## Results

The notebook provides insights into the model's performance through evaluation metrics such as accuracy, precision, recall, and F1-score. Additionally, it calculates the mean average of these metrics across multiple cross-validation folds, providing a comprehensive overview of the model's effectiveness.

## Conclusion

This project demonstrates the application of neural networks in text classification tasks. By leveraging techniques such as data preprocessing, model training, and cross-validation, it showcases a systematic approach to building and evaluating NLP models.

## Accreditation

The neural network implementation in this project draws inspiration from a blog post titled ["A Neural Network in Python, Part 2"](https://iamtrask.github.io//2015/07/27/python-network-part2/) by Andrej Karpathy. The blog post provides insights into building neural networks using Python and serves as a valuable resource for understanding the underlying concepts. We express our gratitude to Andrej Karpathy for sharing this knowledge and contributing to the development of this project.

