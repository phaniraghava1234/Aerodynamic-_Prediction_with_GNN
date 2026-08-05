import airfrans
import os
import numpy as np

# Correct the path to point to the subdirectory created during extraction
data_path = os.path.join('..', 'data', 'Dataset')

print("Loading the 'scarce' task data...")
print(f"Looking for data in: {os.path.abspath(data_path)}")
print("This may take several minutes...")

# --- Load Training Data ---
# The 'train=True' argument tells the function to load the training split.
# The function returns the data and a list of simulation names. We ignore the names for now using '_'
train_data_list, _ = airfrans.dataset.load(root=data_path, task='scarce', train=True)

print("\n--- Loading Test Data ---")
# --- Load Test Data ---
# The 'train=False' argument tells the function to load the test split.
test_data_list, _ = airfrans.dataset.load(root=data_path, task='scarce', train=False)

# The data is loaded as a list of arrays (one for each simulation).
# Let's concatenate them into single large NumPy arrays for easier handling.
train_full_data = np.concatenate(train_data_list, axis=0)
test_full_data = np.concatenate(test_data_list, axis=0)

# According to the documentation, the columns are structured as:
# 7 input features (x), 4 target variables (y), 1 boolean flag (is_airfoil)
# We will split them accordingly.
x_train = train_full_data[:, :7]
y_train = train_full_data[:, 7:11]

x_test = test_full_data[:, :7]
y_test = test_full_data[:, 7:11]


print("\nData loaded and processed successfully!")
print("-" * 30)

# Print the shape of each final array
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}\n")

print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}\n")

print("Shape format is: (total_number_of_points, number_of_features)")