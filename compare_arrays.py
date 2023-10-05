import numpy as np


cspace_seq = np.load('segments_seq.npy')
cspace_vec = np.load('segments_vec.npy')

result = np.isclose(cspace_seq, cspace_vec, atol=1e-2, rtol=1e-2)

print(result)


#Find the indices where the arrays are not equal
indices = np.where(cspace_seq != cspace_vec)

# Display those indices
print("Indices where arrays are not equal:", indices[0])


differing_indices = np.where(~np.isclose(cspace_seq, cspace_vec, atol=1e-5, rtol=1e-5))
print("Differing indices:", differing_indices)


for ind in zip(*differing_indices):
    print(f"At index {ind}, cspace_seq = {cspace_seq[ind]}, cspace_vec = {cspace_vec[ind]}")
