import os
import numpy as np
import cupy as cp
import pandas as pd
from scipy.spatial.distance import cosine

def exact_equal(arr1, arr2):
    return cp.all(arr1 == arr2)

def approx_equal(arr1, arr2, tol):
    return cp.all(cp.isclose(arr1, arr2, atol=tol))

def mean_equal(arr1, arr2, tol):
    return cp.isclose(cp.mean(arr1), cp.mean(arr2), atol=tol)

def var_equal(arr1, arr2, tol):
    return cp.isclose(cp.var(arr1), cp.var(arr2), atol=tol)

def cosine_similarity(arr1, arr2):
    return 1 - cosine(cp.asnumpy(arr1).flatten(), cp.asnumpy(arr2).flatten())

# Initialize DataFrame
df = pd.DataFrame(columns=['File1', 'File2', 'Metric', 'Tolerance', 'Value'])
rows_list = []

# Define folder and tolerances
folder_path = "cspaces"
tolerances = [0.01, 0.001, 0.0001, 1e-05]

# Read .npy files
npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
arrays_dict = {}

for file in npy_files:
    full_path = os.path.join(folder_path, file)
    arrays_dict[file] = cp.load(full_path)

# Compare each pair of files
for file1, arr1 in arrays_dict.items():
    for file2, arr2 in arrays_dict.items():
        if file1 == file2 or arr1.shape != arr2.shape:
            continue

        #rows_list.append({'File1': file1, 'File2': file2, 'Metric': 'Exact Equal', 'Tolerance': 'N/A', 'Value': exact_equal(arr1, arr2).tolist()})

        for tol in tolerances:
            rows_list.append({'File1': file1, 'File2': file2, 'Metric': 'Approx Equal', 'Tolerance': tol, 'Value': approx_equal(arr1, arr2, tol).tolist()})
            rows_list.append({'File1': file1, 'File2': file2, 'Metric': 'Mean Equal', 'Tolerance': tol, 'Value': mean_equal(arr1, arr2, tol).tolist()})
            rows_list.append({'File1': file1, 'File2': file2, 'Metric': 'Var Equal', 'Tolerance': tol, 'Value': var_equal(arr1, arr2, tol).tolist()})

        rows_list.append({'File1': file1, 'File2': file2, 'Metric': 'Cosine Similarity', 'Tolerance': 'N/A', 'Value': cosine_similarity(arr1, arr2)})

# Append rows to DataFrame
df = pd.concat([df, pd.DataFrame(rows_list)], ignore_index=True)

# Save to CSV
#df.to_csv('array_comparison_results.csv', index=False)

# Display the DataFrame (useful in Jupyter Notebook)
print(df)
