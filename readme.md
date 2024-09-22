# fMRI foundation model roadmap

This repository explores the building of a foundation model for brain data.

Here we focus on ICA fMRI data of HCP, but the model is general and could work for other tasks.

*check_foundation_model.ipynb* and *prova_HCP_ICA* are didactic notebooks to familiarize with data, self-supervised masking logic and check the implementation.

*optimize_foundation_model.ipynb* runs the hyperparameters search.

*train_and_evaluate.ipynb* trains the best model (look at wanbd for best parameters) and do some simple visualization and analysis on latent space.

TO DO:

[ ] Run for all ICAs, create a table of reported performances.
[ ] Train baselines model

[ ] Extend to other tasks rather than genre


TO DO (COMPLEX):
[ ] Adapt the model to work on a voxel-wise basis on whole-brain data. (Hard) This could be potentially done, preselecting areas belonging to the same lobe and concatenating region embeddings?