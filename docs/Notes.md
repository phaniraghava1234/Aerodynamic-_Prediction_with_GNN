look through the following official resources. They contain essential information about how the data was generated, what it represents, and how it's intended to be used.

The Official Paper (Preprint on arXiv): https://arxiv.org/abs/2212.07564 - This provides the scientific context and the most up-to-date details.

The Dataset Documentation: https://airfrans.readthedocs.io/ - This is a practical guide for using the data.

The GitHub Repository:(https://github.com/Extrality/AirfRANS) - This contains example scripts for training models on the dataset.

------------------Output from inspect_data.py-----------------------------------

$ python inspect_data.py 
Loading the 'scarce' task data...
Looking for data in: E:\OneDrive\Aerodynamic-_Prediction_with_GNN\data\Dataset
This may take several minutes...
Loading dataset (task: scarce, split: train): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [02:05<00:00,  1.60it/s] 

--- Loading Test Data ---
Loading dataset (task: full, split: test): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [02:15<00:00,  1.47it/s] 

Data loaded and processed successfully!
------------------------------
x_train shape: (35907642, 7)
y_train shape: (35907642, 4)

x_test shape: (35849332, 7)
y_test shape: (35849332, 4)

Shape format is: (total_number_of_points, number_of_features)
