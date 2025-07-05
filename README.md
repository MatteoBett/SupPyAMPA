# SupPyAMPA - Ongoing development

Refactoring and improvement of the PyAMPA published in Ramos-Llorens M, Bello-Madruga R, Valle J, Andreu D, Torrent M. 0. PyAMPA: a high-throughput prediction and optimization tool for antimicrobial peptides. mSystems 0:e01358-23.

The current changes brought:
- Removed GUI, switched to a fully Command Line Interface.
- Enabled custom optimization of user-provided peptides sequences.

## Installation

To install the program, ensure you have `conda` or `miniconda` installed. Then, type the following commands to install:

```
https://github.com/MatteoBett/SupPyAMPA
```

Then move in the created directory and create the conda environment:

```
conda env create -n pyampa-env -f env.yml
```

Activate the environment:

```
conda activate pyampa-env
```

You are now ready to use the program!

## Usage 

There are 4 use-mode of the progam. 

### Classic PyAMPA

Classic generation of peptides to target a given proteome. The proteome must be provided through a fasta file. Output is a csv file containing general statistics indicating the efficiency of peptides. Namely:
- Minimum inhibitory concentration: The lowest concentration of the test peptide at which microorganism growth was visibly absent. Tested experimentally on 8 different bacterial species. Given in µM.
- Hemolytic Probability: Probability of red blood cell hemolysis (**conditionned by the model and the data on which it was trained**)
- Cell-penetrating Probability: Probability of the peptide to bypass the cell barrier (**conditionned by the model and the data on which it was trained**)
- Toxic Probability: chances of the peptide being cytotoxic in mammalian cells (**conditionned by the model and the data on which it was trained**)

```
python -m PYAMPA --mode pyampa -i /PATH/TO/input_proteins.fasta -o /PATH/TO/output_directory
```

### PyAMPA validation

In case you already have a list of peptides in a csv and would like to determine whether they are (or not) fit for anti-microbial activity, you can limit yourself to do:

```
python -m PYAMPA --mode amp_validate --csv /PATH/TO/your_peptides.csv -o /PATH/TO/output_directory
```

The output will be a similar .csv file as for the classic pyampa described before.

### Sequence mutagenesis

To investigate comprehensively the effect of point mutations on an AMP sequence and its antimicrobial activity, you may use the following command:

```
python -m PYAMPA --mode amp_mutagenesis --sequence <YOUR PEPTIDE> -o /PATH/TO/output_directory
```

Heatmaps are saved in the output directory.

### Sequence optimization

To optimize the efficiency of a single peptide sequence for antimicrobial activity, you may use the following command:

```
python -m PYAMPA --mode amp_mutagenesis --sequence <YOUR PEPTIDE> 
```

**The optimized sequence will be displayed in the terminal**

