import math, random, re
import numpy as np
import pickle
from tqdm import tqdm

import PYAMPA.utils as utils
import PYAMPA.viz as viz

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
amino_acids_mutate = ['A', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


def optimization(sequence : str, 
                 POPULATION_SIZE : int = 100, 
                 NUM_GENERATIONS : int = 100,
                 MAX_NO_IMPROVEMENT : int = 20,
                 w1 : int = 1,
                 w2 : int = 1,
                 w3 : int = 1,
                 CROSSOVER_RATE : float = 0.8,
                 MUTATION_RATE : float = 0.2):
    
    assert len(sequence) >= 7, "The sequence must be at least 7 amino acids long."

    sequence = re.sub("C", "S", sequence).upper()
    # Initialize the population with the selected sequence and its mutated versions
    population = [sequence] + [utils.mutate_sequence(sequence) for _ in range(POPULATION_SIZE - 1)] 

    # Initialize the best fitness and best sequence
    best_fitness = -1
    best_sequence = None

    # Counter for generations with no improvement
    no_improvement_counter = 0

    for _ in tqdm(range(NUM_GENERATIONS)):
        # Evaluation
        fitness_scores = []
        for seq in population:
            prob_amp = predict_proba_amp(seq)  # You need to define this function.
            prob_hemo = predict_proba_hemo(seq)  # You need to define this function.
            half_life = utils.calc_half_life(seq)
            fitness_score = utils.fitness(prob_amp, prob_hemo, half_life, w1, w2, w3)
            fitness_scores.append(fitness_score)

        # Check for improvement
        max_fitness = max(fitness_scores)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_sequence = population[fitness_scores.index(max_fitness)]
            no_improvement_counter = 0  # reset the counter
        else:
            no_improvement_counter += 1

        # Check the stopping criterion
        if no_improvement_counter >= MAX_NO_IMPROVEMENT:
            break

        # Selection
        selected_individuals = random.choices(population, weights=fitness_scores, k=POPULATION_SIZE)

        # Crossover
        offspring = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1 = selected_individuals[i]
            parent2 = selected_individuals[i+1]
            if random.random() < CROSSOVER_RATE:
                crossover_point = random.randint(1, len(parent1) - 1)
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
            else:
                child1, child2 = parent1, parent2
            offspring.append(child1)
            offspring.append(child2)

        # Mutation
        for i in range(POPULATION_SIZE):
            if random.random() < MUTATION_RATE:
                mutation_point = random.randint(0, len(offspring[i]) - 1)
                # Exclude cysteine ('C') from the possible amino acids
                new_amino_acid = random.choice([aa for aa in amino_acids if aa != 'C'])
                offspring[i] = offspring[i][:mutation_point] + new_amino_acid + offspring[i][mutation_point+1:]

        # Replacement
        population = offspring

    # Generate and save the helical wheel plots without displaying them
    viz.helical_wheel(sequence, moment=True, filename=r'output/original_helix.png')
    viz.helical_wheel(best_sequence, moment=True, filename=r'output/optimized_helix.png')

    # Add the original and optimized sequences (and their properties)
    prob_amp_orig = predict_proba_amp(sequence)
    prob_hemo_orig = predict_proba_hemo(sequence)
    half_life_orig = utils.calc_half_life(sequence)
    prob_amp_opt = predict_proba_amp(best_sequence)
    prob_hemo_opt = predict_proba_hemo(best_sequence)
    half_life_opt = utils.calc_half_life(best_sequence)

    print(f"Original Sequence: {sequence}")
    print(f"AMP Probability: {prob_amp_orig}, Hemolytic Probability: {prob_hemo_orig}, Half-life: {half_life_orig}")
    print(f"Optimized Sequence: {best_sequence}")   
    print(f"AMP Probability: {prob_amp_opt}, Hemolytic Probability: {prob_hemo_opt}, Half-life: {half_life_opt}")



def predict_proba_amp(sequence, model_amp : str =r'params/AMPValidate.pkl', vectorizer_amp : str =r'params/amp_validate_vectorizer.pkl'):
    with open(model_amp, 'rb') as f:
        model_amp = pickle.load(f)
    with open(vectorizer_amp, 'rb') as f:
        vectorizer_amp = pickle.load(f)

    # Split the sequence into all possible subsequences of length 2
    sequence_split = utils.split_sequence(sequence)
    
    # Count the occurrences of each subsequence
    X = vectorizer_amp.transform([sequence_split])
    
    # Predict the probability using the trained model
    prediction_proba = model_amp.predict_proba(X)
    
    # Return the probability of the class being 1
    return prediction_proba[0][1]

# Probability of a sequence to be hemolytic
def predict_proba_hemo(sequence, model_hemo : str = r'params/hemolysis_model.pkl', vectorizer_hemo : str = r'params/hemolysis_vectorizer.pkl'):
    with open(model_hemo, 'rb') as file:
        model_hemo = pickle.load(file)
    with open(vectorizer_hemo, 'rb') as file:
        vectorizer_hemo = pickle.load(file)
    # Split the sequence into all possible subsequences of length 2
    sequence_split = utils.split_sequence(sequence)
    
    # Count the occurrences of each subsequence
    X = vectorizer_hemo.transform([sequence_split])
    
    # Predict the probability using the trained model
    prediction_proba = model_hemo.predict_proba(X)
    
    # Return the probability of the class being 1
    return prediction_proba[0][1]


