# Quickstart

1. Run `create_data_windows.ipynb`
2. Run `exploratory_data_analysis.ipynb`
3. Run `run_models.ipynb`

# Methodology

## Create Data Windows

Data is sections where every visit creates an input, with n timesteps before it.
Classification is determined by text within the n timesteps output window.
Data is vectorised using pretrained Glove 300 dimension
Test?Train/Validation set produced and saved in data

## Exploratory Data Analysis

Produces data based on copora - X_train dataset.
Visit culmative day of year graph
Top N words graph overall corpora, also by category (revisit / no revisit)
Shannon Diversity Equation demonstration for data imbalance, showing that SMOTE improves SDE scoring.

## Models

Run a Timedistributed LSTM, BiLSTM, LSTM (stacked), BiLSTM (stacked) and CNN on the dataset.

# In progress / Todo

### File: `run_models.ipynb`

Save produced models
Produce training graphs for each model
Implement model evaluation and selection of highest scoring model

### File: `pretrain_glove_on_training_data.ipynb`

Uses Mittens to pretrain Glove Embeddings on the training corpora.

- Sort out GLOVE file saving

### File: `visualise_glove_embeddings.ipynb`

Visualising Glove Embeddings, using TSNE (to show pretraining worked..!)

- Takes a long time to load!

### File: `saliency_measurement.ipynb`

Need to create this file and implement saliency metrics
