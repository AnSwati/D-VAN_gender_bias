# Debias- Gender stereoptypical relationship in word embedding

The repo consists code for Debiased Variational Autoendoder Network(D-VAN) Approach to create debiased word embedding.


Link to download D-VAN-Glove (https://drive.google.com/file/d/1YOx1JsFDV3rKf5T8ZsSbgB8m9v9eFMES/view?usp=sharing)


Key Libraries:

torch: The main PyTorch library for tensor computations and neural networks.
torch.nn: Contains neural network layers and loss functions.
torch.nn.functional (as F): Provides functions for neural network operations.
torch.optim: Contains optimization algorithms.
torchtext.vocab: Used for loading pre-trained word embeddings.

The code implements a Variational Autoencoder (VAE) based approach to debias word embeddings. Here's a breakdown of the main components and their functions:

Encoder Class:

Encodes input embeddings into a latent space.
Uses linear layers (nn.Linear) and ReLU activations.
Outputs mean (mu) and log variance (logvar) of the latent distribution.


Decoder Class:

Decodes latent representations back to the original embedding space.
Uses linear layers and ReLU activations.


GenderClassifier Class:

Predicts gender based on the latent representation.
Uses a sigmoid activation for binary classification.


VAE Class:

Combines Encoder, Decoder, and GenderClassifier.
Implements the reparameterization trick for sampling from the latent distribution.


HistogramApproximator Class:

Approximates the distribution of latent variables using histograms.
Uses torch.histogram for binning values.
Calculates probabilities for reweighting.


cosine_similarity Function:

Calculates cosine similarity between two vectors using F.cosine_similarity.


debias_embeddings Function:

Main function that implements the debiasing algorithm.
Initializes the VAE and optimizer.
Iterates for T epochs, performing the following steps in each epoch:
a. Calculate latent representations and update histogram approximation.
b. Calculate weights for sampling based on the histogram.
c. Sample a batch of embeddings using these weights.
d. Perform forward pass through the VAE.
e. Calculate various losses:

Reconstruction loss (MSE between input and reconstructed embeddings)
KL divergence loss
Gender classification loss
Debiasing loss (based on cosine similarities with gendered vectors)
f. Combine losses and perform backpropagation.


After training, generate debiased embeddings using the trained VAE.



Important Functions and Methods:

torch.clamp():

Used to constrain values within a specified range.
Example: probs = torch.clamp(probs, min=1e-10) ensures all probabilities are at least 1e-10.


torch.nan_to_num():

Replaces NaN, positive infinity, and negative infinity with specified values.
Used to handle potential numerical instabilities.


torch.multinomial():

Samples from a multinomial distribution.
Used for weighted sampling of embeddings.


F.mse_loss():

Calculates Mean Squared Error loss.
Used for reconstruction loss.


F.cosine_similarity():

Calculates cosine similarity between vectors.
Used in debiasing loss calculation.


torch.log(), torch.exp(), torch.pow():

Mathematical operations used in various loss calculations.


optimizer.zero_grad(), loss.backward(), optimizer.step():

Standard PyTorch operations for gradient calculation and parameter updates.



This code aims to learn a debiased latent representation of word embeddings by balancing reconstruction accuracy, gender neutrality, and preservation of non-gendered semantic information. It uses a VAE architecture with an additional gender classifier to achieve this, along with a reweighting scheme based on the estimated latent distribution.

If you like my work please do proper citation : 

@article{TYAGI2024100283,
title = {Enhancing gender equity in resume job matching via debiasing-assisted deep generative model and gender-weighted sampling},
journal = {International Journal of Information Management Data Insights},
volume = {4},
number = {2},
pages = {100283},
year = {2024},
issn = {2667-0968},
doi = {https://doi.org/10.1016/j.jjimei.2024.100283},
url = {https://www.sciencedirect.com/science/article/pii/S2667096824000727},
author = {Swati Tyagi and  Anuj and Wei Qian and Jiaheng Xie and Rick Andrews},
keywords = {Supervised learning, Natural language processing, Algorithmic fairness, Gender bias, Online recruiting, Compounding injustices, Automated recruitment, Mitigating bias, Equality of opportunity},
abstract = {Our work aims to mitigate gender bias within word embeddings and investigates the effects of these adjustments on enhancing fairness in resume job-matching problems. By conducting a case study on resume data, we explore the prevalence of gender bias in job categorization—a significant barrier to equal career opportunities, particularly in the context of machine learning applications. This study scrutinizes how biased representations in job assignments, influenced by a variety of factors such as skills and resume descriptors within diverse semantic frameworks, affect the classification process. The investigation extends to the nuanced language of resumes and the presence of subtle gender biases, including the employment of gender-associated terms, and examines how these terms’ vector representations can skew fairness, leading to a disproportionate mapping of resumes to job categories based on gender. Our findings reveal a significant correlation between gender discrepancies in classification true positive rate and gender imbalances across professions that potentially deepen these disparities. The goal of this study is to (1) mitigate bias at the level of word embeddings via a debiasing-assisted deep generative modeling approach, thereby fostering more equitable and gender-fair vector representations; (2) evaluate the resultant impact on the fairness of job classification; (3) explore the implementation of a gender-weighted sampling technique to achieve a more balanced representation of genders across various job categories when such an imbalance exists. This approach involves modifying the data distribution according to gender before it is input into the classifier model, aiming to ensure equal opportunity and promote gender fairness in occupational classifications. The code for this paper is publicly available on GitHub.}
}
