# Final ML project @EPFL (Brisken lab)


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
