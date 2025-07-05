import argparse

import sklearn

import PYAMPA.pyampa as pyampa
import PYAMPA.viz as viz
import PYAMPA.amp_validate as amp_validate
import PYAMPA.amp_optimization as amp_optimization
import PYAMPA.amp_mutagenesis as amp_mutagenesis


print(sklearn.__version__)

def main_pyampa(testseq: str, output: str):
    pyampa_results_csv = pyampa.pyampa(testseq, output)
    viz.create_summary(pyampa_results_csv=pyampa_results_csv, output_directory=output)
    amp_validate.run_amp_validate(output_directory=output, pyampa_results_csv=pyampa_results_csv)

def main_mut_opti(sequence : str):
    amp_mutagenesis.mutagenesis(sequence=sequence)
    amp_optimization.optimization(sequence=sequence)

def main_amp_validate(pyampa_results_csv: str, output: str):
    amp_validate.run_amp_validate(output_directory=output, pyampa_results_csv=pyampa_results_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PYAMPA pipeline")
    
    parser.add_argument("--mode", type=str, choices=['pyampa', 'amp_validate', 'amp_mutagenesis', 'amp_optimization'],
                        help="Mode to run: " \
                        "'pyampa' for initial analysis, " \
                        "'amp_validate' for validation of PyAMPA (calculate probability of toxicity/hemolysis/cell penetration), " \
                        "'amp_mutagenesis' for mutagenesis analysis (Still unclear), " \
                        "'amp_optimization' for optimization analysis (Optimization of a single peptide sequence to maximize AMP properties)")

    parser.add_argument("-i", "--infasta", type=str, default=r'data\test\proteome.fasta', help='Path to the sequence file (FASTA format)')
    parser.add_argument("-o", "--output", type=str, default=r'output', help='Output directory')

    parser.add_argument("--sequence", type=str, help="Sequence to analyze or optimize")
    parser.add_argument("--csv", type=str, help="Path to the results CSV file from PyAMPA")

    args = parser.parse_args()

    assert args.mode is not None, "Mode must be specified. Use --mode to select a mode."
    mode = args.mode

    if mode == 'pyampa':
        assert args.infasta is not None, "Input FASTA file is required for PyAMPA mode"
        assert args.output is not None, "Output directory is required for PyAMPA mode"
        testseq = args.infasta
        output = args.output
        main_pyampa(testseq=testseq, output=output)
    
    if mode == 'amp_validate':
        assert args.csv is not None, "Results CSV file from PyAMPA is required for AMP validation mode"
        assert args.output is not None, "Output directory is required for AMP validation mode"
        pyampa_results_csv = args.csv
        output = args.output
        main_amp_validate(pyampa_results_csv=pyampa_results_csv, output=output)
    
    if mode == 'amp_mutagenesis':
        assert args.sequence is not None, "Sequence is required for mutagenesis mode"
        assert args.output is not None, "Output directory is required for mutagenesis mode"
        sequence = args.sequence
        amp_mutagenesis.mutagenesis(sequence=sequence, output_dir=args.output)

    if mode == 'amp_optimization':
        assert args.sequence is not None, "Peptide Sequence is required for optimization mode"
        assert (len(args.sequence) > 7) and (len(args.sequence) < 30), "Sequence must be a peptide of length between 8 and 30 amino acids"
        sequence = args.sequence
        amp_optimization.optimization(sequence=sequence)

    
