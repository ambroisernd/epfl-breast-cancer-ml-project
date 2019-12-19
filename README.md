![PyPI - Python Version](https://img.shields.io/pypi/pyversions/matplotlib)

# Final ML project @EPFL (Brisken lab)

70\% of breast cancers can be classified as estrogen receptor positive (ER+). Recent evidences describe tumor as a very complex and heterogeneous disease, highlighting the importance of taking into consideration inter- and intra-tumor variability. Patient-derived xenografts (PDXs) emerged as promising and clinically-relevant preclinical model able to recapitulate the clinical settings.
One promising way to estimate human cells receptivity to hormones is to analyse patient genes expression.
In this document, we present a set of machine learning algorithms and features selection techniques to classify human cells implanted into mice according to their hormone receptivity by using only their gene expression.
According to the results of our study, we show that this method can be used to classify patient's cells receptivity to hormones.

This Github contains all the code and the notebooks used for our experiments.

Final report can be found [here](report/final_report.pdf).


## Dependencies

- numpy==`1.17.2`
- pandas==`0.25.1`
- scikit-learn==`0.21.3`
- matplotlib==`3.1.1`

## Dataset

The dataset has been collected at the EPFL Brisken Laboratory, Switzerland and is part of a PhD research project. It contains information related to 15004 genes describing a total of 29 samples collected from mice which were exposed to 3 different hormonal treatments.

## Notebooks

We provide a set of notebooks presenting the different algorithms tried and the associated plots and results.

#### Feature selection and visualization

- `feature_selection.ipynb` : Comparison of different feature selection methods
- `gaussian_mixture_model.ipynb` : Cluster visualization with Gaussian mixture model
- `pca_visualization.ipynb` : 2D and 3D visualization with Principal Component Analysis (PCA)

#### Classification algorithms

- `knn.ipynb` : K-Nearest Neighbours algorithm
- `logistic_regression.ipynb` : Logistic regression
- `random_forest.ipynb` : Random Forest
- `decision_trees.ipynb` : Decision Trees


## Dataset preprocessing

The preprocessed dataset can be generated from the raw dataset using `preprocessing.py` script with the following options:

- `python preprocessing.py --norm features`: This option will normalize the data with respect to the columns (features).
- `python preprocessing.py --norm samples`: This option will normalize the data with respect to the rows (samples).

Other options:

- `--saving_path`: Path where you want to save preprocessed dataset CSV file.
- `--dataset_path`: Path where the dataset CSV file is located.

## Prediction on unlabeled dataset

The prediction obtained with the different models can be generated using `generate_prediction.py` script with the following options:

- `python generate_prediction.py --model all`: This option will produce prediction for all algorithms implemented in different CSV files.
- `python generate_prediction.py --model your_model`: This option will produce prediction for a specific algorithm (*your_model*) in a CSV files.

Other options:

- `--saving_path path`: Path where you want to save your results.
- `--dataset_path path`: Path where the dataset CSV file is located.
- `--unlabeled_dataset_path path`: Path where the unlabeled dataset CSV file is located.
