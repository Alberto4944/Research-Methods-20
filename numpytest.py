import numpy as np

column_headers = []

row1 = np.array(
    [[1,1,1,1]])

row2 = np.array([2,2,2,2])

row1 = np.vstack((row1,row2))

print(row1)