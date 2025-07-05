import pandas as pd
import pickle
import numpy as np

from PYAMPA.utils import split_sequence

def run_amp_validate(output_directory: str, pyampa_results_csv: str):
    # Load the model and the vectorizer
    with open(r"params/AMPValidate.pkl", 'rb') as file:
        model = pickle.load(file)
    with open(r"params/amp_validate_vectorizer.pkl", 'rb') as file:
        vectorizer = pickle.load(file)

    df = pd.read_csv(pyampa_results_csv)

    sequences_split = df['Peptide'].apply(split_sequence)
    X = vectorizer.transform(sequences_split)
    df['AMPValidate Probability'] = model.predict_proba(X)[:, 1]  # take the second value from the output of predict_proba

    df.to_csv(pyampa_results_csv, index=False)

    # Load the results.csv file
    results_df = pd.read_csv(pyampa_results_csv)

    # Load the trained model
    with open(r'params/activities_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load the fitted CountVectorizer
    with open(r'params/activities_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Load the LabelEncoder
    with open(r'params/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    # Add an 'X' at the end of each peptide sequence to amidate at the C.terminus
    results_df['Peptide'] = results_df['Peptide'].apply(lambda x: x + 'X')

    # Preprocess the peptide sequences
    results_df['sequence_split'] = results_df['Peptide'].apply(split_sequence)

    # Count the occurrences of each subsequence using the fitted CountVectorizer
    X_results = vectorizer.transform(results_df['sequence_split'])

    # Create a new DataFrame with the counts
    counts_results_df = pd.DataFrame(X_results.toarray(), columns=vectorizer.get_feature_names_out())

    # Make predictions for each bacterium
    predictions = []
    for bacterium in range(len(label_encoder.classes_)):
        counts_results_df['bacterium'] = bacterium
        prediction = model.predict(counts_results_df)
        predictions.append(prediction)

    # Transpose the list of predictions and convert it to a DataFrame
    predictions_df = pd.DataFrame(list(map(list, zip(*predictions))), columns=label_encoder.classes_)

    # Add the predictions to the results DataFrame
    results_df = pd.concat([results_df, predictions_df], axis=1)

    # Drop the 'sequence_split' column
    results_df = results_df.drop(columns=['sequence_split'])

    # Undo the log transformation for the bacteria predictions
    for bacterium in label_encoder.classes_:
        results_df[bacterium] = np.exp(results_df[bacterium])

    # Remove the `X` residue 
    results_df['Peptide'] = results_df['Peptide'].str.replace('X', '')

    # Save the updated DataFrame to the results.csv file
    results_df.to_csv(pyampa_results_csv, index=False)

    # Load the model and the vectorizer for hemolysis
    with open(r"params/hemolysis_model.pkl", 'rb') as file:
        hemo_model = pickle.load(file)
    with open(r"params/hemolysis_vectorizer.pkl", 'rb') as file:
        hemo_vectorizer = pickle.load(file)

    # Load the model and the vectorizer for cell-penetrating capacity
    with open(r"params/cpp_model.pkl", 'rb') as file:
        cpp_model = pickle.load(file)
    with open(r"params/cpp_vectorizer.pkl", 'rb') as file:
        cpp_vectorizer = pickle.load(file)

    # Load the model and the vectorizer for toxicity
    with open(r"params/tox_model.pkl", 'rb') as file:
        tox_model = pickle.load(file)
    with open(r"params/tox_vectorizer.pkl", 'rb') as file:
        tox_vectorizer = pickle.load(file)

    # Read results.csv
    pyampa_results_csv = f"{output_directory}/results.csv"
    results_df = pd.read_csv(pyampa_results_csv)

    # Preprocess the peptide sequences and transform them
    sequences_split = results_df['Peptide'].apply(split_sequence)

    # Make predictions for hemolysis
    X_hemo = hemo_vectorizer.transform(sequences_split)
    results_df['Hemolytic Probability'] = hemo_model.predict_proba(X_hemo)[:, 1]  # take the second value from the output of predict_proba

    # Make predictions for cell-penetrating capacity
    X_cpp = cpp_vectorizer.transform(sequences_split)
    results_df['Cell-penetrating Probability'] = cpp_model.predict_proba(X_cpp)[:, 1]  # take the second value from the output of predict_proba

    # Make predictions for toxicity
    X_tox = tox_vectorizer.transform(sequences_split)
    results_df['Toxic Probability'] = tox_model.predict_proba(X_tox)[:, 1]  # take the second value from the output of predict_proba

    # Save the predictions in the results.csv file
    results_df.to_csv(pyampa_results_csv, index=False)



