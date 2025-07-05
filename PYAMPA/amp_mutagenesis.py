import pickle, os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PYAMPA.utils import split_sequence, calc_half_life, fitness

def mutagenesis(sequence : str, 
                output_dir : str,
                w1 : int = 1,
                w2 : int = 1,
                w3 : int = 1,
                model_amp : str = r"params/AMPValidate.pkl",
                vectorizer_amp : str = r"params/amp_validate_vectorizer.pkl",
                model_hemo : str = r"params/hemolysis_model.pkl",
                vectorizer_hemo : str = r"params/hemolysis_vectorizer.pkl"):

    print(sequence, type(sequence))
    # If the sequence ends with 'X', remove it
    if sequence.endswith('X'):
        sequence = sequence[:-1]

    # Generate all possible point mutations
    mutated_sequences = []
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    for i in range(len(sequence)):
        for aa in amino_acids:
            mutated_sequences.append(sequence[:i] + aa + sequence[i+1:])

    # Load the AMPValidate model and vectorizer
    with open(model_amp, 'rb') as file:
        model_amp = pickle.load(file)
    with open(vectorizer_amp, 'rb') as file:
        vectorizer_amp = pickle.load(file)

    # Load the Hemolysis model and vectorizer
    with open(model_hemo, 'rb') as file:
        model_hemo = pickle.load(file)
    with open(vectorizer_hemo, 'rb') as file:
        vectorizer_hemo = pickle.load(file)

    # Preprocess the mutated sequences and transform them
    sequences_split = [split_sequence(seq) for seq in mutated_sequences]
    X_amp = vectorizer_amp.transform(sequences_split)
    X_hemo = vectorizer_hemo.transform(sequences_split)

    # Make predictions
    probabilities_amp = model_amp.predict_proba(X_amp)[:, 1]  # take the second value from the output of predict_proba
    probabilities_hemo = model_hemo.predict_proba(X_hemo)[:, 1]  # take the second value from the output of predict_proba

    # Calculate the half-lives
    half_lives = [calc_half_life(seq) for seq in mutated_sequences]

    # Reshape the probabilities and half-lives into a 2D array for the heatmap
    prob_matrix_amp = np.array(probabilities_amp).reshape(len(sequence), len(amino_acids))
    prob_matrix_hemo = np.array(probabilities_hemo).reshape(len(sequence), len(amino_acids))
    half_life_matrix = np.array(half_lives).reshape(len(sequence), len(amino_acids))

    # Create a DataFrame for the heatmap
    heatmap_df_amp = pd.DataFrame(prob_matrix_amp, columns=list(amino_acids), index=list(sequence))
    heatmap_df_hemo = pd.DataFrame(prob_matrix_hemo, columns=list(amino_acids), index=list(sequence))
    heatmap_df_half_life = pd.DataFrame(half_life_matrix, columns=list(amino_acids), index=list(sequence))

    # Calculate the fitness scores
    w1, w2, w3 = 1, 1, 1
    fitness_scores = fitness(probabilities_amp, probabilities_hemo, half_lives, w1, w2, w3)

    # Reshape the fitness scores into a 2D array for the heatmap
    fitness_matrix = np.array(fitness_scores).reshape(len(sequence), len(amino_acids))

    # Create a DataFrame for the heatmap
    heatmap_df_fitness = pd.DataFrame(fitness_matrix, columns=list(amino_acids), index=list(sequence))

    # Create the heatmaps
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    sns.heatmap(heatmap_df_amp, cmap='YlGnBu', ax=axs[0, 0])
    axs[0, 0].set_title('AMPValidate Probabilities for Point Mutations')

    sns.heatmap(heatmap_df_hemo, cmap='YlGnBu', ax=axs[0, 1])
    axs[0, 1].set_title('Hemolysis Probabilities for Point Mutations')

    sns.heatmap(heatmap_df_half_life, cmap='YlGnBu', ax=axs[1, 0])
    axs[1, 0].set_title('Half-lives for Point Mutations')

    sns.heatmap(heatmap_df_fitness, cmap='YlGnBu', ax=axs[1, 1])
    axs[1, 1].set_title('Fitness Scores for Point Mutations')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'point_mutations_heatmaps.png'))
    print("Heatmaps saved to 'output/point_mutations_heatmaps.png'")