# The hybrid finite-difference—finite-elements approach for the forward modeling of the 3D Magnetotelluric problem 

**Author** : Deniz Varilsuha

**Email** : deniz.varilsuha@itu.edu.tr

<div align="justify">
DBM_model_run.m, Hill_model_run.m, and TwoMountain_model_run.m are the main files the user should interact with. They are for the assembly of the coefficient matrices and solving the resulting linear equation sets for all frequencies specified for the respective models, the Double Brick Model (DBM), the Hill model, and the Two-Mountain model. There are specified options/switches in these scripts that will perform the assembly and the solution using the pure finite-differences (FD), the pure-finite elements (FE), and finally the hybrid method using both FD and FE methods jointly. The Hill model and the Two-mountain model have large topographies and thus distorted elements in their respective mesh so for these models, only the FE and hybrid options are available because the FD approach for those models would require another mesh. These scripts will time the solution for the given frequency range and print out the total iteration number and memory consumption of the matrices in MB (MegaBytes).

## *.mat files
The abovementioned scripts will load in a *.mat file related to those models. The loaded variable can be described as
1. 'm' is the vector that holds the conductivities for the respective models. It doesn't contain the air conductivity or the repeated conductivity values in the padding regions of a 3D mesh
2. 'f' is another vector for the frequency range that will be used for the assembly of the coefficient matrix for a given frequency.
3.  'NK' holds the local position of the nodes of a mesh and it has a 4D structure. The first dimension has a size of (ny+1) where ny is the number of cells in the y-direction. Similarly, the second dimension has a size of (nx+1) where nx is the number of blocks in x-direction. Finally, the third dimension is for the z-direction and it has a size of (nz+1). The last dimension has 3 entries. The first entry is for the x location of the node, the following entries are for the y and z locations.
4.  'EL' is a matrix which holds cell information. It has nx*ny*nz number of rows which is the number of cells/blocks in a given structured mesh. It also has 24 columns. The first 12 columns are for the edge vector potentials and the 8 columns after that are for the scalar potentials defined on nodes. The values in these columns indicate the respective edge or node number in the global matrix. -1 in these entries indicates that the edge/node is located on the edge of the mesh and it will not be included in the global coefficient matrix. The 3 columns after the first 20 columns indicate the cells x-, y- and z- index in a structured 3D hexahedral mesh. The last column gives the conductivity value index which will give out the conductivity value when looked into vector 'm'.
5.  The 'FD' and 'FE' are 3D matrices that have sizes of ny, nx, and nz in their respective dimensions. These are for labeling the cells/blocks to be subjected to the finite-difference method or finite-element method. The values in these matrices can be either 1 or 0. No cell/block could have 1 for both FD and FE simultaneously. At the same time, a single cell shouldn't have a 0 in both matrices. In short, FE and FD matrices are inverse of each other in the logical sense. There are 2 to 3 different FD and FE matrices such as FD0-FE0 or FD1-FE1. These pairs are for different case scenarios where one might want to assemble and solve the linear equations with pure FD, pure FE, or the hybrid approach. These options are specified in the main scripts mentioned above.
</div>

## *.cu files
.cu files are the cuda files which are for creating the global coefficient matrices and vectors (createStiffnessMatrix) and also for solving them (BlockGPBiCG). They are already compiled into .mexw64 files. However one may wish to recompile them using the command in those files. The input and output structure for these mex files are stated in the respective .cu files however it will be also stated here:

### createStiffnessMatrix.cu
**createStiffnessMatrix** is to form the matrix A and vector b to be able to solve Ax=b later on. It also forms the preconditioner matrix M.  

*The inputs should be on the GPU memory unless otherwise stated. They are in the following order:*
1. The node information is stored in a 4D array in a double real precision (8-byte) format. The first dimension of this 4D array represents the y-direction in a structured 3D mesh and it has a size of (ny+1) where ny is the number of blocks in the y-direction
The second dimension is for the x-direction and it has a size of (nx+1), similarly the third dimension is for the z-direction and it has a size of (nz+1). The 4th dimension has a size of three. The first value represents the x coordinates, 
The second value represents the y coordinate and finally, the third one is for the z coordinate in local coordinates. 
2. The second input is a matrix that stores the element information which holds the node numbers for each element. This matrix should have 24 columns and the first 20 of them show the node or edge number for the vector potentials A and scalar potentials Phi in the order shown in the paper.
The following 3 columns are for the elements x y and z index in a structured 3D hexahedral mesh. The last column holds the index for the conductivity value. The values -1 in this matrix represent air conductivities in the conductivity index. 
The other -1 values for edge or node indices are for labeling those edges and nodes that are on the mesh boundaries and they are not included in the coefficient matrix. This input should be in int32 (4-byte integer) format.
3. The 3D labeling matrix labels cells as the finite-difference cells which are not distorted. It has a size of (ny*nx*nz) The values 1 tell that those cells are subject to the finite-difference method. The zeros mean the opposite. This input should be in int32 format.
4. The labeling matrix for the finite-element method. Similarly to the previous input, this one indicates the cell that will be subject to the finite-element numerical method. The input should be in int32 format.
5. The number of blocks in the x-direction (nx) stored on the host side in int32 format.
6. The number of blocks in y-direction (ny) stored on the host side in int32 format.
7. The number of blocks in z-direction (nz) stored on the host side in int32 format.
8. The frequency value is stored on the host and provided in int32 format.
9. The unique conductivity vector is stored in real double format. It doesn't include the air conductivities or the repeating conductivity values in the padding regions.

*The outputs are all on the GPU side and it is ordered in this fashion.*
1. The vector containing the values of the sparse matrix A stored in double-complex format.
2. The column indices vector for the matrix A represented in CSR (compressed sparse row) format and stored in int32 format.
3. The row indices vector for matrix A in CSR format and stored in int32 format.
4. The vector for the sparse matrix M in double complex format.
5. The vector for the column indices of the matrix M in int32 format.
6. The vector for the row indices of the matrix M in int32 format.
7. The right-hand-side (RHS) vector b is stored in double-precision format. If the matrix a has a size of (NxN), this vector has a size of (2N) because it represents the RHS' for both polarizations.
### BlockGPBiCG.cu
**BlockGPBiCG** is the block version of the GPBiCG algorithm that solves the Ax=b for the forward modeling of the 3D MT problem for both polarizations. The inputs should be on the GPU memory (unless otherwise stated explicitly) in this order:

1. Row indices of matrix A stored in CSR (compressed sparse row) format. The vector should be in int32 (4-byte integer) data format.
2. Column indices of matrix A stored in a vector. The data format should be int32.
3. The values of the sparse matrix A. The input should be in a double-complex format.
4. The right-hand side vector b is stored in a double-complex format. The input contains the values for both polarizations. So if the matrix A has a size of NxN then the vector b should have a size of 2N.
5. Row indices of the preconditioner matrix M stored in CSR format. The vector should be in int32 format.
6. Column indices of the matrix M that are stored in int32 format.
7. The values of matrix M are stored in double-complex format.
8. The relative residual norm to end the iterative solution. It is defined such as norm(Ax-b)/norm(b). The input should be on the host side (not on the GPU memory) and should be in real double precision format (8-byte real floating point)
9. Maximum allowed number of iterations for the solution of Ax=b. The input should be on the host side and it should be in real double-precision format.
10. The stagnation detection number is used to determine if the iterative process stagnated and cannot reduce the error level in the specified amount of steps. The input should be in real double-precision format.
11. The initial x values are stored in a vector for the solution of Ax=b. The input should be in a complex double-precision format and it has a size of 2N.

*The outputs are in this order:* 
1. The x values after solving the Ax=b. The values are in double complex format and stored in the GPU memory
2. The relative residual norm of Ax=b (norm(Ax-b)/norm(b)) is calculated at each iteration. This vector is stored on the host side as a real double precision.
3. The final relative residual norm after the solution of Ax=b. The value is a real double precision value and is stored on the host side.

        
