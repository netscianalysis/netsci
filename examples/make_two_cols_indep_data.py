import numpy as np

# 100 rows
N = 100

# 2 columns
M = 2

gaussian_2D = np.zeros((M, N)).astype(np.float32)
for i in range(M):
    gaussian_2D[i,:] = np.random.normal(size=N)
    
with open("sample.dat", "w") as f:
    f.write("column1\tcolumn2\n")
    for i in range(N):
        f.write(str(gaussian_2D[0,i])+"\t"+str(gaussian_2D[1,i])+"\n")
        

