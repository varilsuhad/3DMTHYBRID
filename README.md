Author : Deniz Varilsuha
Email : deniz.varilsuha@itu.edu.tr

DBM_model_run.m , Hill_model_run.m and TwoMountain_model_run.m are the main files the user should interact with. They are for the assembly of the coefficient matrices and solving the resulting linear equation sets for all frequencies specified for the respective models, Double Brick Model (DBM), the Hill model and the Two-Mountain model. There are specified options/switches in these scripts that will perform the assembly and the solution using the pure finite-differences (FD), the pure-finite elements (FE) and finally the hybrid method using both FD and FE methods jointly. The Hill model and the Two-mountain model has large topographies and thus distorted elements in their respective mesh so that for these models, only the FE and hybrid options are available beacuse the FD approach for those models would require another mesh. These scripts will time the solution for the given frequency range and print out the total iteration number and memory consumption of the matrices in MB (MegaBytes).

Abovementioned scripts will load in a *.mat file related to those models. The loaded variable can be described such as
1. 'm' is vector holds the conductivities for the respective models. It doesn't contain the air conductivies or the repeated conductivity values in the padding regions of a 3D mesh
2. 'f' is another vector for the frequency range that will be used for the assembly of the coefficient matrix for a given frequency.
3.  'NK' holds the local position of the nodes of a mesh and it has a 4D structure. The first dimension has a size of (ny+1) where ny is the number of cells in y-direction. Similarly, the second dimension has a size of (nx+1) where nx is the number of blocks in x-direction. Finally the third dimension is for the z-direction and it has a size of (nz+1). The last dimension has 3 entries. The first entry is for the x location of the node, the following entries are for the y and z locations.
4.  'EL' is a matrix which holds cell information. It has nx*ny*nz number of rows which is the number of cells/blocks in a given structured mesh. It also has 24 columns. The first 12 column are for the edge vector potentials and the 8 columns after that are for the scalar potentials defined on nodes. The values in these columns indicate the respective edge or node number in the global matrix. -1 in these entries indicate that the edge/node is located on the edge of the mesh and it will not be included into the global coefficient matrix. The 3 columns after the first 20 columns indicate the cells x-, y- and z- index in a structured 3D hexahedral mesh. The last column gives the conductivity value index which will give out the conductivity value when looked into vector 'm'.
5.  The FD and FE are 3D matrices which have sizes of ny, nx and nz in their respective dimensions. These are for labeling the cells/blocks to be subjected to finite-difference method or finite-element method. The values in these matrices can be either 1 or 0. No cell/block could have 1 for both FD and FE simultaneously. At the same time, a single cell shouldn't have a 0 in both matrices. In short, FE and FD matrices are inverse of each other in logical sense. There are 2 to 3 different FD and FE matrices such as FD0-FE0 or FD1-FE1. These pairs are for different case scenarios where one might want to assemble and solve the linear equations with pure FD, pure FE or the hybrid approach. These options are specified in the main scripts mentioned above.

.cu files are the cuda files which are for creating the global coefficient matrices and vectors (createStiffnessMatrix) and also for solving them (BlockGPBiCG). They are already compiled into .mexw64 files. However one may wish to recompile them using the command in those files. The input and output structure for these mex files are stated in the respective .cu files however it will be also stated here:

The purpose of 'createStiffnessMatrix' is to form the matrix A and vector b to be able to solve Ax=b later on. It also forms the preconditioner matrix M.  

The inputs should be on the GPU memory unless otherwise stated. They are in the following order:
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

The outputs are all on the GPU side and it is ordered in this fashion.
1. The vector containing the values of the sparse matrix A stored in double-complex format.
2. The column indices vector for the matrix A represented in CSR (compressed sparse row) format and stored in int32 format.
3. The row indices vector for the matrix A in CSR format and stored in int32 format.
4. The vector for the sparse matrix M in double complex format.
5. The vector for the column indices of the matrix M in int32 format.
6. The vector for the row indices of the matrix M in int32 format.
7. The right-hand-side (RHS) vector b is stored in double-precision format. If the matrix a has a size of (NxN), this vector has a size of (2N) because it represents the RHS' for both polarizations.  

        
