import numpy as np
import random
import math

from Bio.SeqUtils.ProtParam import ProteinAnalysis

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
amino_acids_mutate = ['A', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def calc_features(sequence):
    # Define the set of nonpolar residues
    nonpolar_residues = set('ACGILMFPWYV')

    # Calculate the percentage of nonpolar residues
    NP = sum(1 for aa in sequence if aa in nonpolar_residues) / len(sequence)

    # Calculate the presence of Trp and Tyr
    W = 1 if 'W' in sequence else 0
    Y = 1 if sequence.count('Y') >= 2 else 0

    # Calculate the isoelectric point
    protein_analysis = ProteinAnalysis(sequence)
    IP_val = protein_analysis.isoelectric_point()
    IP = 1 if IP_val > 10 else 0

    return NP, W, Y, IP

def split_sequence(sequence):
    return ' '.join([sequence[i:i+2] for i in range(len(sequence) - 1)])

def load_scale(scalename):
    """Method to load scale values for a given amino acid scale

    :param scalename: amino acid scale name, for available scales see the
        :class:`modlamp.descriptors.PeptideDescriptor()` documentation.
    :return: amino acid scale values in dictionary format.
    """
    # predefined amino acid scales dictionary
    scales = {
        'eisenberg': {'I': [1.4], 'F': [1.2], 'V': [1.1], 'L': [1.1], 'W': [0.81], 'M': [0.64], 'A': [0.62],
                      'G': [0.48], 'C': [0.29], 'Y': [0.26], 'P': [0.12], 'T': [-0.05], 'S': [-0.18], 'H': [-0.4],
                      'E': [-0.74], 'N': [-0.78], 'Q': [-0.85], 'D': [-0.9], 'K': [-1.5], 'R': [-2.5]}
    }

    if scalename in scales:
        return scales[scalename]
    else:
        raise ValueError(f"The scale {scalename} is not defined.")
    
def calculate_hydrophobic_moment(sequence, window=1000, angle=100):
    # if sequence is shorter than window, take the whole sequence instead
    window = min(window, len(sequence))
    
    # calculate descriptor values for each amino acid in the sequence
    d_eisberg = load_scale('eisenberg')
    descriptors = [d_eisberg[aa][0] for aa in sequence]
    
    # calculate the hydrophobic moment for each window in the sequence
    moments = []
    for i in range(len(descriptors) - window + 1):
        window_descriptors = descriptors[i:i + window]
        
        # calculate actual moment (radial)
        rads = angle * (np.pi / 180) * np.asarray(range(window))
        vcos = sum(window_descriptors * np.cos(rads))
        vsin = sum(window_descriptors * np.sin(rads))
        
        moment = np.sqrt(vsin**2 + vcos**2) / window
        moments.append(moment)
    
    # return the maximum hydrophobic moment
    return max(moments)    

def calculate_hydrophobic_moment(sequence, window=1000, angle=100):
    # if sequence is shorter than window, take the whole sequence instead
    window = min(window, len(sequence))
    
    # calculate descriptor values for each amino acid in the sequence
    d_eisberg = load_scale('eisenberg')
    descriptors = [d_eisberg[aa][0] for aa in sequence]
    
    # calculate the hydrophobic moment for each window in the sequence
    moments = []
    for i in range(len(descriptors) - window + 1):
        window_descriptors = descriptors[i:i + window]
        
        # calculate actual moment (radial)
        rads = angle * (np.pi / 180) * np.asarray(range(window))
        vcos = sum(window_descriptors * np.cos(rads))
        vsin = sum(window_descriptors * np.sin(rads))
        
        moment = np.sqrt(vsin**2 + vcos**2) / window
        moments.append(moment)
    
    # return the maximum hydrophobic moment
    return max(moments)    


def load_scale(scalename):
    """Method to load scale values for a given amino acid scale

    :param scalename: amino acid scale name, for available scales see the
        :class:`modlamp.descriptors.PeptideDescriptor()` documentation.
    :return: amino acid scale values in dictionary format.
    """
    # predefined amino acid scales dictionary
    scales = {
        'eisenberg': {'I': [1.4], 'F': [1.2], 'V': [1.1], 'L': [1.1], 'W': [0.81], 'M': [0.64], 'A': [0.62],
                      'G': [0.48], 'C': [0.29], 'Y': [0.26], 'P': [0.12], 'T': [-0.05], 'S': [-0.18], 'H': [-0.4],
                      'E': [-0.74], 'N': [-0.78], 'Q': [-0.85], 'D': [-0.9], 'K': [-1.5], 'R': [-2.5]}
    }

    if scalename in scales:
        return scales[scalename]
    else:
        raise ValueError(f"The scale {scalename} is not defined.")
    
def mutate_sequence(sequence):
    # Choose a random index for the mutation
    mutation_index = random.randint(0, len(sequence) - 1)

    # Choose a new amino acid that is different from the current one
    new_amino_acid = random.choice([aa for aa in amino_acids_mutate if aa != sequence[mutation_index]])

    # Create the new sequence with the mutation
    new_sequence = sequence[:mutation_index] + new_amino_acid + sequence[mutation_index + 1:]

    return new_sequence

def calc_half_life(sequence):
    # Calculate the peptide features
    NP, W, Y, IP = calc_features(sequence)

    # Calculate the natural log of the half-life
    ln_t_half = 2.226 + (0.053 * NP * 100) - (1.515 * W) + (1.290 * Y) - (1.052 * IP)

    # Convert to the half-life
    t_half = math.exp(ln_t_half)

    return t_half

def fitness(antimicrobial_proba, hemolytic_proba, half_life, w1, w2, w3):
    # Convert half_life to a NumPy array
    half_life = np.array(half_life)

    # Limit the half-life to a maximum of 360 minutes
    half_life_limited = np.where(half_life > 360, 360, half_life)

    # Normalize the half-life to the range 0-1
    half_life_normalized = half_life_limited / 360

    # Calculate the fitness score as a weighted sum of the three factors
    return w1 * antimicrobial_proba - w2 * hemolytic_proba + w3 * half_life_normalized
