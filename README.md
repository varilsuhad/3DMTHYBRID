.mat files contain the node (NK), element (EL), frequency(f), FE-FD region (FE and FD) and parameter(m) information for the three models; the DBM, Hill and Two-Mountain models .
.cu files are the cuda files which are for creating the Global matrices and vectors (createStiffnessMatrix) and also for solving them (BlockGPBiCG). 
.cu files are already compiled into .mexw64 files. However one may wish to recompile them using the command in those files.
.m files are the main scripts that are for loading the .mat files and creating the linear systems (Ax=b) and solving those equations for different numerical methods. 
These scripts will time the solution for the given frequency range and print out the total iteration number and matrix sizes.
