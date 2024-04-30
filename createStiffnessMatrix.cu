// Author: Deniz Varilsuha
// Email: deniz.varilsuha@itu.edu.tr
// Compilation date: 29/02/2024
// Compiled using Cuda version 12.3 and Matlab R2023b

// To compile the code use the following line in Matlab's command line (change the paths if necessary)
// mexcuda -R2018a createStiffnessMatrix.cu -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\include" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64" NVCCFLAGS='"-Wno-deprecated-gpu-targets -arch=sm_86"' -lcusparse -lcublas


// The purpose of this routine is to form the matrix A and vector b to be able to solve Ax=b later on. It also forms the preconditioner matrix M.  
// The inputs should be on the GPU memory unless otherwise stated. They are in the following order:
// 1. The node information is stored in a 4D array in a double real precision (8-byte) format. The first dimension of this 4D array represents the y-direction in a structured 3D mesh and it has a size of (ny+1) where ny is the number of blocks in the y-direction
// The second dimension is for the x-direction and it has a size of (nx+1), similarly the third dimension is for the z-direction and it has a size of (nz+1). The 4th dimension has a size of three. The first value represents the x coordinates, 
// The second value represents the y coordinate and finally, the third one is for the z coordinate in local coordinates. 
// 2. The second input is a matrix that stores the element information which holds the node numbers for each element. This matrix should have 24 columns and the first 20 of them show the node or edge number for the vector potentials A and scalar potentials Phi in the order shown in the paper.
// The following 3 columns are for the elements x y and z index in a structured 3D hexahedral mesh. The last column holds the index for the conductivity value. The values -1 in this matrix represent air conductivities in the conductivity index. 
// The other -1 values for edge or node indices are for labeling those edges and nodes that are on the mesh boundaries and they are not included in the coefficient matrix. This input should be in int32 (4-byte integer) format.
// 3. The 3D labeling matrix labels cells as the finite-difference cells which are not distorted. It has a size of (ny*nx*nz) The values 1 tell that those cells are subject to the finite-difference method. The zeros mean the opposite. This input should be in int32 format.
// 4. The labeling matrix for the finite-element method. Similarly to the previous input, this one indicates the cell that will be subject to the finite-element numerical method. The input should be in int32 format.
// 5. The number of blocks in the x-direction (nx) stored on the host side in int32 format.
// 6. The number of blocks in y-direction (ny) stored on the host side in int32 format.
// 7. The number of blocks in z-direction (nz) stored on the host side in int32 format.
// 8. The frequency value is stored on the host and provided in int32 format.
// 9. The unique conductivity vector is stored in real double format. It doesn't include the air conductivities or the repeating conductivity values in the padding regions.

// The outputs are all on the GPU side and it is ordered in this fashion.
// 1. The vector containing the values of the sparse matrix A stored in double-complex format.
// 2. The column indices vector for the matrix A represented in CSR (compressed sparse row) format and stored in int32 format.
// 3. The row indices vector for the matrix A in CSR format and stored in int32 format.
// 4. The vector for the sparse matrix M in double complex format.
// 5. The vector for the column indices of the matrix M in int32 format.
// 6. The vector for the row indices of the matrix M in int32 format.
// 7. The right-hand-side (RHS) vector b is stored in double-precision format. If the matrix a has a size of (NxN), this vector has a size of (2N) because it represents the RHS' for both polarizations.  


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math_constants.h>
#include <cuComplex.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"

__device__ int NKindex(int i, int j, int k, int nx, int ny, int nz, int ok){
int say=0;
int oki=(nx+1)*(ny+1)*(nz+1);
int zyon=(ny+1)*(nx+1);
int xyon=(ny+1);    

if(ok==1){
say=j+(i-1)*xyon+(k-1)*zyon;
}else if(ok==2){
say=j+(i-1)*xyon+(k-1)*zyon+oki*1;
}else if(ok==3){
say=j+(i-1)*xyon+(k-1)*zyon+oki*2;
}
return say;
}

__device__ void xyzlxlylz(double *NK, double *x, double *y, double *z, double *lx, double *ly, double *lz, int i, int j, int k, int nx, int ny, int nz){
x[0]=NK[NKindex(i, j, k, nx, ny, nz, 1)-1];
x[1]=NK[NKindex(i+1, j, k, nx, ny, nz, 1)-1];
x[2]=NK[NKindex(i+1, j+1, k, nx, ny, nz, 1)-1];
x[3]=NK[NKindex(i, j+1, k, nx, ny, nz, 1)-1];
x[4]=NK[NKindex(i, j, k+1, nx, ny, nz, 1)-1];
x[5]=NK[NKindex(i+1, j, k+1, nx, ny, nz, 1)-1];
x[6]=NK[NKindex(i+1, j+1, k+1, nx, ny, nz, 1)-1];
x[7]=NK[NKindex(i, j+1, k+1, nx, ny, nz, 1)-1];

y[0]=NK[NKindex(i, j, k, nx, ny, nz, 2)-1];
y[1]=NK[NKindex(i+1, j, k, nx, ny, nz, 2)-1];
y[2]=NK[NKindex(i+1, j+1, k, nx, ny, nz, 2)-1];
y[3]=NK[NKindex(i, j+1, k, nx, ny, nz, 2)-1];
y[4]=NK[NKindex(i, j, k+1, nx, ny, nz, 2)-1];
y[5]=NK[NKindex(i+1, j, k+1, nx, ny, nz, 2)-1];
y[6]=NK[NKindex(i+1, j+1, k+1, nx, ny, nz, 2)-1];
y[7]=NK[NKindex(i, j+1, k+1, nx, ny, nz, 2)-1]; 

z[0]=NK[NKindex(i, j, k, nx, ny, nz, 3)-1];
z[1]=NK[NKindex(i+1, j, k, nx, ny, nz, 3)-1];
z[2]=NK[NKindex(i+1, j+1, k, nx, ny, nz, 3)-1];
z[3]=NK[NKindex(i, j+1, k, nx, ny, nz, 3)-1];
z[4]=NK[NKindex(i, j, k+1, nx, ny, nz, 3)-1];
z[5]=NK[NKindex(i+1, j, k+1, nx, ny, nz, 3)-1];
z[6]=NK[NKindex(i+1, j+1, k+1, nx, ny, nz, 3)-1];
z[7]=NK[NKindex(i, j+1, k+1, nx, ny, nz, 3)-1];  

lx[0]=sqrt((x[1]-x[0])*(x[1]-x[0])+ (y[1]-y[0])*(y[1]-y[0]) + (z[1]-z[0])*(z[1]-z[0]));
lx[1]=sqrt((x[2]-x[3])*(x[2]-x[3])+ (y[2]-y[3])*(y[2]-y[3]) + (z[2]-z[3])*(z[2]-z[3]));
lx[2]=sqrt((x[5]-x[4])*(x[5]-x[4])+ (y[5]-y[4])*(y[5]-y[4]) + (z[5]-z[4])*(z[5]-z[4]));
lx[3]=sqrt((x[6]-x[7])*(x[6]-x[7])+ (y[6]-y[7])*(y[6]-y[7]) + (z[6]-z[7])*(z[6]-z[7]));

ly[0]=sqrt((x[3]-x[0])*(x[3]-x[0])+ (y[3]-y[0])*(y[3]-y[0]) + (z[3]-z[0])*(z[3]-z[0]));
ly[1]=sqrt((x[7]-x[4])*(x[7]-x[4])+ (y[7]-y[4])*(y[7]-y[4]) + (z[7]-z[4])*(z[7]-z[4]));
ly[2]=sqrt((x[2]-x[1])*(x[2]-x[1])+ (y[2]-y[1])*(y[2]-y[1]) + (z[2]-z[1])*(z[2]-z[1]));
ly[3]=sqrt((x[6]-x[5])*(x[6]-x[5])+ (y[6]-y[5])*(y[6]-y[5]) + (z[6]-z[5])*(z[6]-z[5]));    
    
lz[0]=sqrt((x[4]-x[0])*(x[4]-x[0])+ (y[4]-y[0])*(y[4]-y[0]) + (z[4]-z[0])*(z[4]-z[0]));
lz[1]=sqrt((x[5]-x[1])*(x[5]-x[1])+ (y[5]-y[1])*(y[5]-y[1]) + (z[5]-z[1])*(z[5]-z[1]));
lz[2]=sqrt((x[7]-x[3])*(x[7]-x[3])+ (y[7]-y[3])*(y[7]-y[3]) + (z[7]-z[3])*(z[7]-z[3]));
lz[3]=sqrt((x[6]-x[2])*(x[6]-x[2])+ (y[6]-y[2])*(y[6]-y[2]) + (z[6]-z[2])*(z[6]-z[2]));
}

__device__ double Jabc3DMTF(double *x, double *y, double *z, double a,double b, double c, double *JJ){
double q1,q2,q3,q4,q5,q6,q7,q8;
double w1,w2,w3,w4,w5,w6,w7,w8;
double e1,e2,e3,e4,e5,e6,e7,e8;
double s1,s2,s3,t1,t2,t3,r1,r2,r3;
double dd;
q1=(1-b)*(1-c)*(-1.0);
q2=(1-b)*(1-c);
q3=(1+b)*(1-c);
q4=(1+b)*(1-c)*(-1.0);
q5=(1-b)*(1+c)*(-1.0);
q6=(1-b)*(1+c);
q7=(1+b)*(1+c);
q8=(1+b)*(1+c)*(-1.0);

w1=(1-a)*(1-c)*(-1.0);
w2=(1+a)*(1-c)*(-1.0);
w3=(1+a)*(1-c);
w4=(1-a)*(1-c);
w5=(1-a)*(1+c)*(-1.0);
w6=(1+a)*(1+c)*(-1.0);
w7=(1+a)*(1+c);
w8=(1-a)*(1+c);
   
e1=(1-a)*(1-b)*(-1.0);
e2=(1+a)*(1-b)*(-1.0);
e3=(1+a)*(1+b)*(-1.0);
e4=(1-a)*(1+b)*(-1.0);
e5=(1-a)*(1-b);
e6=(1+a)*(1-b);
e7=(1+a)*(1+b);
e8=(1-a)*(1+b);

s1=q1*x[0]+q2*x[1]+q3*x[2]+q4*x[3]+q5*x[4]+q6*x[5]+q7*x[6]+q8*x[7];
s2=q1*y[0]+q2*y[1]+q3*y[2]+q4*y[3]+q5*y[4]+q6*y[5]+q7*y[6]+q8*y[7];
s3=q1*z[0]+q2*z[1]+q3*z[2]+q4*z[3]+q5*z[4]+q6*z[5]+q7*z[6]+q8*z[7];

t1=w1*x[0]+w2*x[1]+w3*x[2]+w4*x[3]+w5*x[4]+w6*x[5]+w7*x[6]+w8*x[7];
t2=w1*y[0]+w2*y[1]+w3*y[2]+w4*y[3]+w5*y[4]+w6*y[5]+w7*y[6]+w8*y[7];
t3=w1*z[0]+w2*z[1]+w3*z[2]+w4*z[3]+w5*z[4]+w6*z[5]+w7*z[6]+w8*z[7];

r1=e1*x[0]+e2*x[1]+e3*x[2]+e4*x[3]+e5*x[4]+e6*x[5]+e7*x[6]+e8*x[7];
r2=e1*y[0]+e2*y[1]+e3*y[2]+e4*y[3]+e5*y[4]+e6*y[5]+e7*y[6]+e8*y[7];
r3=e1*z[0]+e2*z[1]+e3*z[2]+e4*z[3]+e5*z[4]+e6*z[5]+e7*z[6]+e8*z[7];    

r1=0.125*r1;
r2=0.125*r2;
r3=0.125*r3;
t1=0.125*t1;
t2=0.125*t2;
t3=0.125*t3;
s1=0.125*s1;
s2=0.125*s2;
s3=0.125*s3;

dd=s1*(t2*r3-t3*r2)-s2*(t1*r3-r1*t3)+s3*(t1*r2-t2*r1);

JJ[0]=(t2*r3-r2*t3)/dd;
JJ[3]=(-1.0)*(s2*r3-r2*s3)/dd;
JJ[6]=(s2*t3-s3*t2)/dd;
JJ[1]=(-1.0)*(t1*r3-r1*t3)/dd;
JJ[4]=(s1*r3-r1*s3)/dd;
JJ[7]=(-1.0)*(s1*t3-t1*s3)/dd;
JJ[2]=(t1*r2-r1*t2)/dd;
JJ[5]=(-1.0)*(s1*r2-s2*r1)/dd;
JJ[8]=(s1*t2-t1*s2)/dd;

return dd;

}

__device__ void rotNUnF(double *lx, double *ly, double *lz, double *J, double a,double b, double c, double *RN, double *N, double *M){
double s1,s2,s3,t1,t2,t3,r1,r2,r3;
double n1,n2,n3,n4,q1,q2,q3,q4,w1,w2,w3,w4;
double k1,k2,k3,k4,e1,e2,e3,e4,e5,e6,e7,e8;
double y1,y2,y3,y4,y5,y6,y7,y8;
double u1,u2,u3,u4,u5,u6,u7,u8;

s1=J[0];
s2=J[1];
s3=J[2];
r1=J[3];
r2=J[4];
r3=J[5];
t1=J[6];
t2=J[7];
t3=J[8];

n1=lx[0]*0.125*(1-b)*(1-c);
n2=lx[1]*0.125*(1+b)*(1-c);
n3=lx[2]*0.125*(1-b)*(1+c);
n4=lx[3]*0.125*(1+b)*(1+c);

N[0]=n1*s1;
N[1]=n2*s1;
N[2]=n3*s1;
N[3]=n4*s1;
N[12]=n1*s2;
N[13]=n2*s2;
N[14]=n3*s2;
N[15]=n4*s2;
N[24]=n1*s3;
N[25]=n2*s3;
N[26]=n3*s3;
N[27]=n4*s3; 

q1=(-1.0)*(r1+t1)+b*t1+c*r1;
q2=(r1-t1)-b*t1-c*r1;
q3=(t1-r1)-b*t1-c*r1;
q4=(r1+t1)+b*t1+c*r1;

w1=(-1.0)*(r2+t2)+b*t2+c*r2;
w2=(r2-t2)-b*t2-c*r2;
w3=(t2-r2)-b*t2-c*r2;
w4=(r2+t2)+b*t2+c*r2;

e1=(-1.0)*(r3+t3)+b*t3+c*r3;
e2=(r3-t3)-b*t3-c*r3;
e3=(t3-r3)-b*t3-c*r3;
e4=(r3+t3)+b*t3+c*r3;    

k1=lx[0]*0.125;    
k2=lx[1]*0.125;
k3=lx[2]*0.125;
k4=lx[3]*0.125;

RN[0]=(w1*s3-e1*s2)*k1; RN[12]=(e1*s1-q1*s3)*k1; RN[24]=(q1*s2-w1*s1)*k1;
RN[1]=(w2*s3-e2*s2)*k2; RN[13]=(e2*s1-q2*s3)*k2; RN[25]=(q2*s2-w2*s1)*k2;
RN[2]=(w3*s3-e3*s2)*k3; RN[14]=(e3*s1-q3*s3)*k3; RN[26]=(q3*s2-w3*s1)*k3;
RN[3]=(w4*s3-e4*s2)*k4; RN[15]=(e4*s1-q4*s3)*k4; RN[27]=(q4*s2-w4*s1)*k4;    

n1=ly[0]*0.125*(1-a)*(1-c);
n2=ly[1]*0.125*(1-a)*(1+c);
n3=ly[2]*0.125*(1+a)*(1-c);
n4=ly[3]*0.125*(1+a)*(1+c);

N[4]=n1*r1; N[16]=n1*r2;  N[28]=n1*r3;
N[5]=n2*r1; N[17]=n2*r2;  N[29]=n2*r3;
N[6]=n3*r1; N[18]=n3*r2;  N[30]=n3*r3;
N[7]=n4*r1; N[19]=n4*r2;  N[31]=n4*r3;   

q1=(-1.0)*(s1+t1)+a*t1+c*s1;
q2=(-s1+t1)-a*t1-c*s1;
q3=s1-t1-a*t1-c*s1;
q4=s1+t1+a*t1+c*s1;

w1=(-s2-t2)+a*t2+c*s2;
w2=(-s2+t2)-a*t2-c*s2;
w3=s2-t2-a*t2-c*s2;
w4=s2+t2+a*t2+c*s2;

e1=(-s3-t3)+a*t3+c*s3;
e2=(-s3+t3)-a*t3-c*s3;
e3=s3-t3-a*t3-c*s3;
e4=s3+t3+a*t3+c*s3;

k1=ly[0]*0.125; k2=ly[1]*0.125; k3=ly[2]*0.125; k4=ly[3]*0.125;
RN[4]=(w1*r3-e1*r2)*k1; RN[16]=(e1*r1-q1*r3)*k1; RN[28]=(q1*r2-w1*r1)*k1;
RN[5]=(w2*r3-e2*r2)*k2; RN[17]=(e2*r1-q2*r3)*k2; RN[29]=(q2*r2-w2*r1)*k2;
RN[6]=(w3*r3-e3*r2)*k3; RN[18]=(e3*r1-q3*r3)*k3; RN[30]=(q3*r2-w3*r1)*k3;
RN[7]=(w4*r3-e4*r2)*k4; RN[19]=(e4*r1-q4*r3)*k4; RN[31]=(q4*r2-w4*r1)*k4;    

n1=lz[0]*0.125*(1-a)*(1-b);
n2=lz[1]*0.125*(1+a)*(1-b);
n3=lz[2]*0.125*(1-a)*(1+b);
n4=lz[3]*0.125*(1+a)*(1+b);

N[8]=n1*t1; N[20]=n1*t2;  N[32]=n1*t3;
N[9]=n2*t1; N[21]=n2*t2;  N[33]=n2*t3;
N[10]=n3*t1; N[22]=n3*t2;  N[34]=n3*t3;
N[11]=n4*t1; N[23]=n4*t2;  N[35]=n4*t3;

q1=(-s1-r1)+a*r1+b*s1;
q2=s1-r1-a*r1-b*s1;
q3=(-s1+r1)-a*r1-b*s1;
q4=s1+r1+a*r1+b*s1;

w1=(-s2-r2)+a*r2+b*s2;
w2=s2-r2-a*r2-b*s2;
w3=(-s2+r2)-a*r2-b*s2;
w4=s2+r2+a*r2+b*s2;

e1=(-s3-r3)+a*r3+b*s3;
e2=s3-r3-a*r3-b*s3;
e3=(-s3+r3)-a*r3-b*s3;
e4=s3+r3+a*r3+b*s3;

k1=lz[0]*0.125; k2=lz[1]*0.125; k3=lz[2]*0.125; k4=lz[3]*0.125;
RN[8]=(w1*t3-e1*t2)*k1; RN[20]=(e1*t1-q1*t3)*k1; RN[32]=(q1*t2-w1*t1)*k1;
RN[9]=(w2*t3-e2*t2)*k2; RN[21]=(e2*t1-q2*t3)*k2; RN[33]=(q2*t2-w2*t1)*k2;
RN[10]=(w3*t3-e3*t2)*k3; RN[22]=(e3*t1-q3*t3)*k3; RN[34]=(q3*t2-w3*t1)*k3;
RN[11]=(w4*t3-e4*t2)*k4; RN[23]=(e4*t1-q4*t3)*k4; RN[35]=(q4*t2-w4*t1)*k4;    

e1=(1-b)*(1-c)*(-0.125);
e2=(1-b)*(1-c)*0.125;
e3=(1+b)*(1-c)*0.125;
e4=(1+b)*(1-c)*(-0.125);
e5=(1-b)*(1+c)*(-0.125);
e6=(1-b)*(1+c)*0.125;
e7=(1+b)*(1+c)*0.125;
e8=(1+b)*(1+c)*(-0.125);

y1=(1-a)*(1-c)*(-0.125);
y2=(1+a)*(1-c)*(-0.125);
y3=(1+a)*(1-c)*0.125;
y4=(1-a)*(1-c)*0.125;
y5=(1-a)*(1+c)*(-0.125);
y6=(1+a)*(1+c)*(-0.125);
y7=(1+a)*(1+c)*0.125;
y8=(1-a)*(1+c)*0.125;

u1=(1-a)*(1-b)*(-0.125);
u2=(1+a)*(1-b)*(-0.125);
u3=(1+a)*(1+b)*(-0.125);
u4=(1-a)*(1+b)*(-0.125);
u5=(1-a)*(1-b)*0.125;
u6=(1+a)*(1-b)*0.125;
u7=(1+a)*(1+b)*0.125;
u8=(1-a)*(1+b)*0.125;
    
M[0]=e1*s1+y1*r1+u1*t1;
M[1]=e2*s1+y2*r1+u2*t1;
M[2]=e3*s1+y3*r1+u3*t1;
M[3]=e4*s1+y4*r1+u4*t1;
M[4]=e5*s1+y5*r1+u5*t1;
M[5]=e6*s1+y6*r1+u6*t1;
M[6]=e7*s1+y7*r1+u7*t1;
M[7]=e8*s1+y8*r1+u8*t1;

M[8]=e1*s2+y1*r2+u1*t2;
M[9]=e2*s2+y2*r2+u2*t2;
M[10]=e3*s2+y3*r2+u3*t2;
M[11]=e4*s2+y4*r2+u4*t2;
M[12]=e5*s2+y5*r2+u5*t2;
M[13]=e6*s2+y6*r2+u6*t2;
M[14]=e7*s2+y7*r2+u7*t2;
M[15]=e8*s2+y8*r2+u8*t2;    

M[16]=e1*s3+y1*r3+u1*t3;
M[17]=e2*s3+y2*r3+u2*t3;
M[18]=e3*s3+y3*r3+u3*t3;
M[19]=e4*s3+y4*r3+u4*t3;
M[20]=e5*s3+y5*r3+u5*t3;
M[21]=e6*s3+y6*r3+u6*t3;
M[22]=e7*s3+y7*r3+u7*t3;
M[23]=e8*s3+y8*r3+u8*t3;
}
__device__ void FEMPRE(double *x,double *y, double *z, cuDoubleComplex *K)
{
double x1=z[0]-z[1];
double x2=z[2]-z[3];
double x3=z[4]-z[5];
double x4=z[6]-z[7];

double y1=z[0]-z[3];
double y2=z[4]-z[7];
double y3=z[2]-z[1];        
double y4=z[6]-z[5]; 
    int ii,jj;

for (ii=0;ii<4;ii++){
    for (jj=0;jj<4;jj++){
    K[ii+(jj+4)*20]=make_cuDoubleComplex(0.0,0.0);
    K[ii+4+(jj)*20]=make_cuDoubleComplex(0.0,0.0);
    }
}

for (ii=0;ii<12;ii++){
    for (jj=12;jj<20;jj++){
    K[ii+(jj)*20]=make_cuDoubleComplex(0.0,0.0);
    K[jj+(ii)*20]=make_cuDoubleComplex(0.0,0.0);
    }
}    


if (y1!=0.0 || y2!=0.0 || y3!=0.0 || y4!=0.0 || x1!=0 || x2!=0 || x3!=0 || x4!=0){
    for (ii=0;ii<4;ii++){
        for (jj=0;jj<4;jj++){
        K[ii+(jj+8)*20]=make_cuDoubleComplex(0.0,cuCimag(K[ii+(jj+8)*20]));
        K[ii+8+(jj)*20]=make_cuDoubleComplex(0.0,cuCimag(K[ii+8+(jj)*20]));
        K[ii+4+(jj+8)*20]=make_cuDoubleComplex(0.0,cuCimag(K[ii+4+(jj+8)*20]));
        K[ii+8+(jj+4)*20]=make_cuDoubleComplex(0.0,cuCimag(K[ii+8+(jj+4)*20]));      
        }
    }
}else{
    for (ii=0;ii<4;ii++){
        for (jj=0;jj<4;jj++){
        K[ii+(jj+8)*20]=make_cuDoubleComplex(0.0,0.0);
        K[ii+8+(jj)*20]=make_cuDoubleComplex(0.0,0.0);
        K[ii+4+(jj+8)*20]=make_cuDoubleComplex(0.0,0.0);
        K[ii+8+(jj+4)*20]=make_cuDoubleComplex(0.0,0.0);        
        }
    }
}
}


__device__ void baseKedgeUNF(double *lx, double *ly, double *lz, double *x,double *y, double *z, cuDoubleComplex *K, double sigma, double f){
double xx[3]={0,-sqrt(0.6),sqrt(0.6)};
double ww[3]={0.888888888888889, 0.555555555555556, 0.555555555555556};
int i,j,k,ii,jj;
double R[36]={0};
double N[36]={0};
double M[24]={0};
double JJ[9]={0};
double w1,w2,w3,a1,b1,c1,det1=(-1.0),w123,say;
double mu0=4.0*CUDART_PI*0.0000001;    

double Mkat=2.0*CUDART_PI*mu0*sigma*f;
double Ukat=mu0*sigma;
double Lkat=mu0*(-0.5)/CUDART_PI/f*sigma;

for (i=0;i<3;i++){
    w1=ww[i];
    a1=xx[i];    
    for (j=0;j<3;j++){
        w2=ww[j];
        b1=xx[j];     
        for (k=0;k<3;k++){
            w3=ww[k];
            c1=xx[k];  
               
            det1=Jabc3DMTF(x, y, z, a1, b1, c1, JJ);
            rotNUnF(lx, ly, lz, JJ, a1, b1, c1, R, N, M);
            w123=w1*w2*w3*det1;
            
            for (ii=0;ii<12;ii++){
                for (jj=0;jj<12;jj++){
                    say=R[ii]*R[jj]+R[12+ii]*R[12+jj]+R[24+ii]*R[24+jj];
                    K[ii+20*jj]=cuCadd(K[ii+20*jj],make_cuDoubleComplex(w123*say,0.0));
                }
            }
            
            for (ii=0;ii<12;ii++){
                for (jj=0;jj<12;jj++){
                    say=N[ii]*N[jj]+N[12+ii]*N[12+jj]+N[24+ii]*N[24+jj];
                    K[ii+20*jj]=cuCadd(K[ii+20*jj],make_cuDoubleComplex(0.0,w123*say*Mkat));
                }
            }                               
            
            for (ii=0;ii<12;ii++){
                for (jj=0;jj<8;jj++){
                    say=N[ii]*M[jj]+N[12+ii]*M[8+jj]+N[24+ii]*M[16+jj];
                    K[ii+(jj+12)*20]=cuCadd(K[ii+(jj+12)*20],make_cuDoubleComplex(say*w123*Ukat,0.0));
                    K[jj+12+(ii)*20]=cuCadd(K[jj+12+(ii)*20],make_cuDoubleComplex(say*w123*Ukat,0.0));            
                }
            }
            
            for (ii=0;ii<8;ii++){
                for (jj=0;jj<8;jj++){
                    say=M[ii]*M[jj]+M[8+ii]*M[8+jj]+M[16+ii]*M[16+jj];
                    K[ii+12+(jj+12)*20]=cuCadd(K[ii+12+(jj+12)*20],make_cuDoubleComplex(0.0,say*w123*Lkat));                
                }
            }                        
        }
    }
}
}

__device__ void getnoEL(int ii, int *no, int *EL, int N){
no[0]=EL[ii+N*0];    
no[1]=EL[ii+N*1];    
no[2]=EL[ii+N*2];    
no[3]=EL[ii+N*3];    
no[4]=EL[ii+N*4];    
no[5]=EL[ii+N*5];    
no[6]=EL[ii+N*6];    
no[7]=EL[ii+N*7];  
no[8]=EL[ii+N*8];  
no[9]=EL[ii+N*9];  
no[10]=EL[ii+N*10];  
no[11]=EL[ii+N*11];  
no[12]=EL[ii+N*12];  
no[13]=EL[ii+N*13];  
no[14]=EL[ii+N*14];  
no[15]=EL[ii+N*15];  
no[16]=EL[ii+N*16];  
no[17]=EL[ii+N*17];  
no[18]=EL[ii+N*18];  
no[19]=EL[ii+N*19];  
}


__global__ void FEpart(int N, int nx, int ny, int nz, double *NK, int *EL, int *FE, double *Bbg, double *m, double f,
                     cuDoubleComplex *BKv, int *BKc, int *BKr, cuDoubleComplex *BMv)
{
    int ii = blockIdx.x*blockDim.x + threadIdx.x;  
    if (ii < N){
        double b[40]={0.0};  
        double sinx[20]={0.0};
        double siny[20]={0.0};      
        int i,j,k,pp,kk,jj;
        int no[20];
        int ind[20]={0};
        int ind2[20]={0};
        int c,c2,al;
        int ek=0;
        double x[8]={0};
        double y[8]={0};
        double z[8]={0};
        double lx[4]={0};
        double ly[4]={0};
        double lz[4]={0};
        double dd;     
        int totV=(nx)*(ny-1)*(nz-1)+(nx-1)*(ny)*(nz-1)+(nx-1)*(ny-1)*(nz)+(nx-1)*(ny-1)*(nz-1);   
        int FEind;
        double sigma;
        
        i=EL[ii+N*20];
        j=EL[ii+N*21];
        k=EL[ii+N*22];
        pp=EL[ii+N*23];
        
        if(pp==(-1)){
        sigma=0.00000001;
        }else{
        sigma=m[pp-1];
        }
        
        FEind=ny*nx*(k-1)+ny*(i-1)+(j-1);    
        getnoEL(ii, no, EL, N);
        xyzlxlylz(NK, x, y, z, lx, ly, lz, i, j, k, nx, ny, nz);
        cuDoubleComplex K[400]={0};
        
        if(FE[FEind]>0){
        baseKedgeUNF(lx, ly, lz, x, y, z, K, sigma, f);
        }
        
        // b hazırlanır    
        c=0;c2=0;
        for (kk=0;kk<20;kk++){
        if(no[kk]>0){
          c2=c2+1;
          ind2[c2-1]=kk+1;
        }else{
          c=c+1;
           ind[c-1]=kk+1;
           if(kk+1<=4){
           sinx[c-1]=1.0;
           siny[c-1]=0.0;
           }else if(kk+1>4 && kk+1<=8){
           sinx[c-1]=0.0;
           siny[c-1]=1.0;
           }else{
           sinx[c-1]=0.0;
           siny[c-1]=0.0;
           }
        }
        }
        
        if(c>0){
            for(kk=0;kk<c2;kk++){
                for (jj=0;jj<c;jj++){
                if(ind2[kk]<=12){
                dd=cuCreal(K[ind2[kk]-1 + 20*(ind[jj]-1)]);
                b[kk]=b[kk]-sinx[jj]*dd;
                b[kk+20]=b[kk+20]-siny[jj]*dd; 
                }
                }
            }
        }
        
        // b yerine yazılır    
        if(c>0){
            for(kk=0;kk<c2;kk++){
            al=no[ind2[kk]-1];
            atomicAdd(&Bbg[(al-1)],b[kk]);
            atomicAdd(&Bbg[(al-1)+totV],b[kk+20]);
            }
        }
        
        BKr[ii]=160*ii;
        if(ii==0){
        BKr[N]=160*N;
        }
        
        for(kk=0;kk<20;kk++){
            if(no[kk]<0){
                continue;
            }
            for(jj=0;jj<20;jj++){ 
                if(no[jj]<0){
                continue;
                }
                
                if(kk<4){
                ek=kk;
                }else if(kk>=4 && kk<8){
                ek=kk-4;
                }else if(kk>=8 && kk<12){
                ek=kk-8;
                }else{
                ek=kk-12;
                }
                BKv[(no[kk]-1)*160+jj+ek*20]=cuCadd(BKv[(no[kk]-1)*160+jj+ek*20],K[jj + 20*kk]);
                BKc[(no[kk]-1)*160+jj+ek*20]=no[jj];
            }
        }
        
        if(FE[FEind]>0){    
        FEMPRE(x,y,z,K);
        }
        
        for(kk=0;kk<20;kk++){
            if(no[kk]<0){
                continue;
            }
            for(jj=0;jj<20;jj++){ 
                if(no[jj]<0){
                continue;
                }
                
                if(kk<4){
                ek=kk;
                }else if(kk>=4 && kk<8){
                ek=kk-4;
                }else if(kk>=8 && kk<12){
                ek=kk-8;
                }else{
                ek=kk-12;
                }
                BMv[(no[kk]-1)*160+jj+ek*20]=cuCadd(BMv[(no[kk]-1)*160+jj+ek*20],K[jj + 20*kk]);
            }
        }    
    }
}

__device__ int neredeAVF(int i, int j, int k, int nx, int ny, int nz, int ok)
{
int xyon=0,yyon=0, noAx=0, noAy=0, noAz=0;
noAx=nx*(ny-1)*(nz-1);
noAy=(nx-1)*ny*(nz-1);
noAz=(nx-1)*(ny-1)*nz;
int say;
if(ok==1){
    if(i>nx || i<1 || j>ny || j<2 || k>nz || k<2){
    say=(-1);   
    }else{
    xyon=(nz-1)*(ny-1);
    yyon=(nz-1);   
    say=(i-1)*xyon+(j-2)*yyon+(k-1);
    }
}else if(ok==2){
    if(i>nx || i<2 || j>ny || j<1 || k>nz || k<2){
    say=(-1);   
    }else{    
    xyon=(nz-1)*ny;
    yyon=(nz-1);
    say=noAx+(i-2)*xyon+(j-1)*yyon+(k-1);    
    }
}else if(ok==3){
    if(i>nx || i<2 || j>ny || j<2 || k>nz || k<1){
    say=(-1);   
    }else{    
    xyon=nz*(ny-1);
    yyon=nz;
    say=noAx+noAy+(i-2)*xyon+(j-2)*yyon+k;
    }
}else if(ok==4){
    if(i>nx || i<2 || j>ny || j<2 || k>nz || k<2){
    say=(-1);   
    }else{    
    xyon=(nz-1)*(ny-1);
    yyon=(nz-1);
    say=noAx+noAy+noAz+(i-2)*xyon+(j-2)*yyon+(k-1);
    }
}
return say;    
}

__device__ void getnos(int i, int j, int k,int nx, int ny, int nz, int *no){
    no[0]=neredeAVF(i,j,k,nx,ny,nz,1); 
    no[1]=neredeAVF(i,j+1,k,nx,ny,nz,1); 
    no[2]=neredeAVF(i,j,k+1,nx,ny,nz,1);     
    no[3]=neredeAVF(i,j+1,k+1,nx,ny,nz,1); 
    no[4]=neredeAVF(i,j,k,nx,ny,nz,2); 
    no[5]=neredeAVF(i,j,k+1,nx,ny,nz,2); 
    no[6]=neredeAVF(i+1,j,k,nx,ny,nz,2); 
    no[7]=neredeAVF(i+1,j,k+1,nx,ny,nz,2); 
    no[8]=neredeAVF(i,j,k,nx,ny,nz,3); 
    no[9]=neredeAVF(i+1,j,k,nx,ny,nz,3); 
    no[10]=neredeAVF(i,j+1,k,nx,ny,nz,3); 
    no[11]=neredeAVF(i+1,j+1,k,nx,ny,nz,3); 
    no[12]=neredeAVF(i,j,k,nx,ny,nz,4); 
    no[13]=neredeAVF(i+1,j,k,nx,ny,nz,4); 
    no[14]=neredeAVF(i+1,j+1,k,nx,ny,nz,4);     
    no[15]=neredeAVF(i,j+1,k,nx,ny,nz,4); 
    no[16]=neredeAVF(i,j,k+1,nx,ny,nz,4); 
    no[17]=neredeAVF(i+1,j,k+1,nx,ny,nz,4); 
    no[18]=neredeAVF(i+1,j+1,k+1,nx,ny,nz,4);     
    no[19]=neredeAVF(i,j+1,k+1,nx,ny,nz,4);  
}

__device__ void getdxyz(int i, int j, int k, int nx, int ny, int nz, double *NK, double *dxo, double *dyo, double *dzo){
int ii1,ii2;
ii1=NKindex(i+1,j,k,nx,ny,nz,1);
ii2=NKindex(i,j,k,nx,ny,nz,1);
*dxo=NK[ii1-1]-NK[ii2-1];

ii1=NKindex(i,j+1,k,nx,ny,nz,2);
ii2=NKindex(i,j,k,nx,ny,nz,2);
*dyo=NK[ii1-1]-NK[ii2-1];    

ii1=NKindex(i,j,k+1,nx,ny,nz,3);
ii2=NKindex(i,j,k,nx,ny,nz,3);
*dzo=NK[ii1-1]-NK[ii2-1];    
}

__device__ void katsayiFDMPRE(cuDoubleComplex *K1)
{
int i,j;

for (i=0;i<4;i++){
for (j=4;j<20;j++){
K1[i+j*20]=make_cuDoubleComplex(0.0,0.0);
K1[j+i*20]=make_cuDoubleComplex(0.0,0.0);
}
}

for (i=4;i<8;i++){
for (j=8;j<20;j++){
K1[i+j*20]=make_cuDoubleComplex(0.0,0.0);
K1[j+i*20]=make_cuDoubleComplex(0.0,0.0);
}
}

for (i=8;i<12;i++){
for (j=12;j<20;j++){
K1[i+j*20]=make_cuDoubleComplex(0.0,0.0);
K1[j+i*20]=make_cuDoubleComplex(0.0,0.0);
}
} 

K1[0+20*3]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);    
K1[1+20*2]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);        
K1[2+20*1]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);        
K1[3+20*0]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);            
K1[4+20*7]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                
K1[5+20*6]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                    
K1[6+20*5]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                    
K1[7+20*4]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                    
K1[8+20*11]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                        
K1[9+20*10]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                            
K1[10+20*9]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[11+20*8]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[12+20*14]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[12+20*17]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[12+20*18]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[12+20*19]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);  
K1[13+20*15]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[13+20*16]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[13+20*18]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[13+20*19]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);   
K1[14+20*12]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[14+20*16]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[14+20*17]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[14+20*19]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);
K1[15+20*13]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[15+20*16]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[15+20*17]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[15+20*18]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);
K1[16+20*13]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[16+20*14]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[16+20*15]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[16+20*18]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);
K1[17+20*12]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[17+20*14]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[17+20*15]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[17+20*19]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);
K1[18+20*12]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[18+20*13]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[18+20*15]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[18+20*16]=make_cuDoubleComplex(0.0000000000000000000000000000001,0); 
K1[19+20*12]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[19+20*13]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[19+20*14]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);                                
K1[19+20*17]=make_cuDoubleComplex(0.0000000000000000000000000000001,0);  


}

__device__ void katsayiFD(double dxo, double dyo, double dzo, cuDoubleComplex *K1, double sigma, double f)
{

double mu0=4*CUDART_PI*0.0000001;    
double kzi,kyi,qy1,qz1,kf1;
double kxi,qx1,kax,kay,kaz,vx1,vy1,vz1;      

kzi=dxo*dyo/(-2.0*dzo);
kyi=dxo*dzo/(-2.0*dyo);
qy1=dzo*0.5;
qz1=dyo*0.5;
kf1=mu0*dyo*dzo*0.25;   

//Ex i,j,k 
K1[0]=make_cuDoubleComplex((kzi+kyi)*(-1.0),0.0);
K1[20*1]=make_cuDoubleComplex(kyi,0.0);
K1[20*2]=make_cuDoubleComplex(kzi,0.0);
K1[20*4]=make_cuDoubleComplex(qy1*(-1.0),0.0);
K1[20*6]=make_cuDoubleComplex(qy1,0.0);
K1[20*8]=make_cuDoubleComplex(qz1*(-1.0),0.0);
K1[20*9]=make_cuDoubleComplex(qz1,0.0);
K1[20*12]=make_cuDoubleComplex(kf1*(-1.0)*sigma,0.0);
K1[20*13]=make_cuDoubleComplex(kf1*sigma,0.0);
K1[0]=cuCadd(K1[0],make_cuDoubleComplex(0.0,CUDART_PI*mu0*0.5*dxo*dyo*dzo*sigma*f));

//Ex i,j+1,k 
K1[1+1*20]=make_cuDoubleComplex((kzi+kyi)*(-1.0),0.0);
K1[1]=make_cuDoubleComplex(kyi,0.0);
K1[1+20*3]=make_cuDoubleComplex(kzi,0.0);
K1[1+20*4]=make_cuDoubleComplex(qy1,0.0);    
K1[1+20*6]=make_cuDoubleComplex(qy1*(-1.0),0.0);
K1[1+20*10]=make_cuDoubleComplex(qz1*(-1.0),0.0);
K1[1+20*11]=make_cuDoubleComplex(qz1,0.0);    
K1[1+20*14]=make_cuDoubleComplex(kf1*sigma,0.0);    
K1[1+20*15]=make_cuDoubleComplex(kf1*(-1.0)*sigma,0.0);
K1[1+1*20]=cuCadd(K1[1+1*20],make_cuDoubleComplex(0.0,CUDART_PI*mu0*0.5*dxo*dyo*dzo*sigma*f));    

//Ex i,j,k+1 
K1[2+2*20]=make_cuDoubleComplex((kzi+kyi)*(-1.0),0.0);
K1[2]=make_cuDoubleComplex(kzi,0.0);
K1[2+20*3]=make_cuDoubleComplex(kyi,0.0);
K1[2+20*5]=make_cuDoubleComplex(qy1*(-1.0),0.0);
K1[2+20*7]=make_cuDoubleComplex(qy1,0.0);
K1[2+20*8]=make_cuDoubleComplex(qz1,0.0);
K1[2+20*9]=make_cuDoubleComplex(qz1*(-1.0),0.0);
K1[2+20*16]=make_cuDoubleComplex(kf1*(-1.0)*sigma,0.0);
K1[2+20*17]=make_cuDoubleComplex(kf1*sigma,0.0);
K1[2+2*20]=cuCadd(K1[2+2*20],make_cuDoubleComplex(0.0,CUDART_PI*mu0*0.5*dxo*dyo*dzo*sigma*f));        

//Ex i,j+1,k+1 
K1[3+3*20]=make_cuDoubleComplex((kzi+kyi)*(-1.0),0.0);
K1[3+20*1]=make_cuDoubleComplex(kzi,0.0);    
K1[3+2*20]=make_cuDoubleComplex(kyi,0.0);
K1[3+20*5]=make_cuDoubleComplex(qy1,0.0);
K1[3+20*7]=make_cuDoubleComplex(qy1*(-1.0),0.0);
K1[3+20*10]=make_cuDoubleComplex(qz1,0.0);
K1[3+20*11]=make_cuDoubleComplex(qz1*(-1.0),0.0);
K1[3+20*18]=make_cuDoubleComplex(kf1*sigma,0.0);
K1[3+20*19]=make_cuDoubleComplex(kf1*(-1.0)*sigma,0.0);
K1[3+3*20]=cuCadd(K1[3+3*20],make_cuDoubleComplex(0.0,CUDART_PI*mu0*0.5*dxo*dyo*dzo*sigma*f));        

//EY  
kzi=dyo*dxo/(-2.0*dzo);
kxi=dyo*dzo/(-2.0*dxo);
qx1=dzo*0.5;
qz1=dxo*0.5;
kf1=mu0*dxo*dzo*0.25; 

//Ey i,j,k 
K1[4+4*20]=make_cuDoubleComplex((kzi+kxi)*(-1.0),0.0);
K1[4+5*20]=make_cuDoubleComplex(kzi,0.0);
K1[4+6*20]=make_cuDoubleComplex(kxi,0.0);
K1[4]=make_cuDoubleComplex(qx1*(-1.0),0.0);
K1[4+1*20]=make_cuDoubleComplex(qx1,0.0);
K1[4+8*20]=make_cuDoubleComplex(qz1*(-1.0),0.0);
K1[4+10*20]=make_cuDoubleComplex(qz1,0.0);
K1[4+20*12]=make_cuDoubleComplex(kf1*(-1.0)*sigma,0.0);
K1[4+20*15]=make_cuDoubleComplex(kf1*sigma,0.0);
K1[4+4*20]=cuCadd(K1[4+4*20],make_cuDoubleComplex(0.0,CUDART_PI*mu0*0.5*dxo*dyo*dzo*sigma*f)); 

//Ey i,j,k+1
K1[5+5*20]=make_cuDoubleComplex((kzi+kxi)*(-1.0),0.0);
K1[5+4*20]=make_cuDoubleComplex(kzi,0.0);
K1[5+7*20]=make_cuDoubleComplex(kxi,0.0);
K1[5+20*2]=make_cuDoubleComplex(qx1*(-1.0),0.0);
K1[5+3*20]=make_cuDoubleComplex(qx1,0.0);
K1[5+8*20]=make_cuDoubleComplex(qz1,0.0);
K1[5+10*20]=make_cuDoubleComplex(qz1*(-1.0),0.0);    
K1[5+20*16]=make_cuDoubleComplex(kf1*(-1.0)*sigma,0.0);
K1[5+20*19]=make_cuDoubleComplex(kf1*sigma,0.0);
K1[5+5*20]=cuCadd(K1[5+5*20],make_cuDoubleComplex(0.0,CUDART_PI*mu0*0.5*dxo*dyo*dzo*sigma*f));     

//Ey i+1,j,k
K1[6+6*20]=make_cuDoubleComplex((kzi+kxi)*(-1.0),0.0);
K1[6+4*20]=make_cuDoubleComplex(kxi,0.0);
K1[6+7*20]=make_cuDoubleComplex(kzi,0.0);
K1[6]=make_cuDoubleComplex(qx1,0.0);
K1[6+1*20]=make_cuDoubleComplex(qx1*(-1.0),0.0);
K1[6+9*20]=make_cuDoubleComplex(qz1*(-1.0),0.0);
K1[6+11*20]=make_cuDoubleComplex(qz1,0.0); 
K1[6+20*13]=make_cuDoubleComplex(kf1*(-1.0)*sigma,0.0);
K1[6+20*14]=make_cuDoubleComplex(kf1*sigma,0.0);
K1[6+6*20]=cuCadd(K1[6+6*20],make_cuDoubleComplex(0.0,CUDART_PI*mu0*0.5*dxo*dyo*dzo*sigma*f));       

//Ey i+1,j,k+1 
K1[7+7*20]=make_cuDoubleComplex((kzi+kxi)*(-1.0),0.0);
K1[7+5*20]=make_cuDoubleComplex(kxi,0.0);
K1[7+6*20]=make_cuDoubleComplex(kzi,0.0);
K1[7+20*2]=make_cuDoubleComplex(qx1,0.0);
K1[7+3*20]=make_cuDoubleComplex(qx1*(-1.0),0.0);
K1[7+9*20]=make_cuDoubleComplex(qz1,0.0);
K1[7+11*20]=make_cuDoubleComplex(qz1*(-1.0),0.0);
K1[7+20*17]=make_cuDoubleComplex(kf1*(-1.0)*sigma,0.0);
K1[7+20*18]=make_cuDoubleComplex(kf1*sigma,0.0);
K1[7+7*20]=cuCadd(K1[7+7*20],make_cuDoubleComplex(0.0,CUDART_PI*mu0*0.5*dxo*dyo*dzo*sigma*f));      

//EZ ler
kyi=dzo*dxo/(-2.0*dyo);
kxi=dzo*dyo/(-2.0*dxo);
qx1=dyo*0.5;
qy1=dxo*0.5;
kf1=mu0*dxo*dyo*0.25;       
  
//Ez i,j,k 
K1[8+8*20]=make_cuDoubleComplex((kyi+kxi)*(-1.0),0.0);
K1[8+9*20]=make_cuDoubleComplex(kxi,0.0);
K1[8+10*20]=make_cuDoubleComplex(kyi,0.0);
K1[8]=make_cuDoubleComplex(qx1*(-1.0),0.0);
K1[8+2*20]=make_cuDoubleComplex(qx1,0.0);
K1[8+4*20]=make_cuDoubleComplex(qy1*(-1.0),0.0);
K1[8+5*20]=make_cuDoubleComplex(qy1,0.0);
K1[8+20*12]=make_cuDoubleComplex(kf1*(-1.0)*sigma,0.0);
K1[8+20*16]=make_cuDoubleComplex(kf1*sigma,0.0);
K1[8+8*20]=cuCadd(K1[8+8*20],make_cuDoubleComplex(0.0,CUDART_PI*mu0*0.5*dxo*dyo*dzo*sigma*f));      

//Ez i+1,j,k 
K1[9+9*20]=make_cuDoubleComplex((kyi+kxi)*(-1.0),0.0);
K1[9+8*20]=make_cuDoubleComplex(kxi,0.0);
K1[9+11*20]=make_cuDoubleComplex(kyi,0.0);
K1[9]=make_cuDoubleComplex(qx1,0.0);
K1[9+2*20]=make_cuDoubleComplex(qx1*(-1.0),0.0);
K1[9+6*20]=make_cuDoubleComplex(qy1*(-1.0),0.0);
K1[9+7*20]=make_cuDoubleComplex(qy1,0.0);
K1[9+20*13]=make_cuDoubleComplex(kf1*(-1.0)*sigma,0.0);
K1[9+20*17]=make_cuDoubleComplex(kf1*sigma,0.0);
K1[9+9*20]=cuCadd(K1[9+9*20],make_cuDoubleComplex(0.0,CUDART_PI*mu0*0.5*dxo*dyo*dzo*sigma*f));      

//Ez i,j+1,k 
K1[10+10*20]=make_cuDoubleComplex((kyi+kxi)*(-1.0),0.0);
K1[10+8*20]=make_cuDoubleComplex(kyi,0.0);
K1[10+11*20]=make_cuDoubleComplex(kxi,0.0);
K1[10+1*20]=make_cuDoubleComplex(qx1*(-1.0),0.0);
K1[10+3*20]=make_cuDoubleComplex(qx1,0.0);
K1[10+4*20]=make_cuDoubleComplex(qy1,0.0);
K1[10+5*20]=make_cuDoubleComplex(qy1*(-1.0),0.0);    
K1[10+20*15]=make_cuDoubleComplex(kf1*(-1.0)*sigma,0.0);
K1[10+20*19]=make_cuDoubleComplex(kf1*sigma,0.0);
K1[10+10*20]=cuCadd(K1[10+10*20],make_cuDoubleComplex(0.0,CUDART_PI*mu0*0.5*dxo*dyo*dzo*sigma*f));     

//Ez i+1,j+1,k 
K1[11+11*20]=make_cuDoubleComplex((kyi+kxi)*(-1.0),0.0);
K1[11+9*20]=make_cuDoubleComplex(kyi,0.0);
K1[11+10*20]=make_cuDoubleComplex(kxi,0.0);
K1[11+1*20]=make_cuDoubleComplex(qx1,0.0);
K1[11+3*20]=make_cuDoubleComplex(qx1*(-1.0),0.0);
K1[11+6*20]=make_cuDoubleComplex(qy1,0.0);
K1[11+7*20]=make_cuDoubleComplex(qy1*(-1.0),0.0);
K1[11+20*14]=make_cuDoubleComplex(kf1*(-1.0)*sigma,0.0);
K1[11+20*18]=make_cuDoubleComplex(kf1*sigma,0.0);
K1[11+11*20]=cuCadd(K1[11+11*20],make_cuDoubleComplex(0.0,CUDART_PI*mu0*0.5*dxo*dyo*dzo*sigma*f));

kax=mu0*0.25*dyo*dzo;
kay=mu0*0.25*dxo*dzo;
kaz=mu0*0.25*dxo*dyo;
vx1=mu0/CUDART_PI*0.125*dyo*dzo/dxo; 
vy1=mu0/CUDART_PI*0.125*dxo*dzo/dyo;    
vz1=mu0/CUDART_PI*0.125*dxo*dyo/dzo; 
double fd=1.0/f;

//V i,j,k
K1[12+12*20]=make_cuDoubleComplex(0.0,(-1.0)*(vx1+vy1+vz1)*sigma*fd);    
K1[12+13*20]=make_cuDoubleComplex(0.0,vx1*sigma*fd);    
K1[12+15*20]=make_cuDoubleComplex(0.0,vy1*sigma*fd);    
K1[12+16*20]=make_cuDoubleComplex(0.0,vz1*sigma*fd);                               
K1[12+0*20]=make_cuDoubleComplex(kax*(-1.0)*sigma,0.0);
K1[12+4*20]=make_cuDoubleComplex(kay*(-1.0)*sigma,0.0);
K1[12+8*20]=make_cuDoubleComplex(kaz*(-1.0)*sigma,0.0);    

//V i+1,j,k
K1[13+13*20]=make_cuDoubleComplex(0.0,(-1.0)*(vx1+vy1+vz1)*sigma*fd);    
K1[13+12*20]=make_cuDoubleComplex(0.0,vx1*sigma*fd);    
K1[13+14*20]=make_cuDoubleComplex(0.0,vy1*sigma*fd);    
K1[13+17*20]=make_cuDoubleComplex(0.0,vz1*sigma*fd); 
K1[13+0*20]=make_cuDoubleComplex(kax*sigma,0.0);
K1[13+6*20]=make_cuDoubleComplex(kay*(-1.0)*sigma,0.0);
K1[13+9*20]=make_cuDoubleComplex(kaz*(-1.0)*sigma,0.0);

//V i+1,j+1,k
K1[14+14*20]=make_cuDoubleComplex(0.0,(-1.0)*(vx1+vy1+vz1)*sigma*fd);    
K1[14+15*20]=make_cuDoubleComplex(0.0,vx1*sigma*fd);    
K1[14+13*20]=make_cuDoubleComplex(0.0,vy1*sigma*fd);    
K1[14+18*20]=make_cuDoubleComplex(0.0,vz1*sigma*fd);     
K1[14+1*20]=make_cuDoubleComplex(kax*sigma,0.0);
K1[14+6*20]=make_cuDoubleComplex(kay*sigma,0.0);
K1[14+11*20]=make_cuDoubleComplex(kaz*(-1.0)*sigma,0.0); 

//V i,j+1,k
K1[15+15*20]=make_cuDoubleComplex(0.0,(-1.0)*(vx1+vy1+vz1)*sigma*fd);    
K1[15+14*20]=make_cuDoubleComplex(0.0,vx1*sigma*fd);    
K1[15+12*20]=make_cuDoubleComplex(0.0,vy1*sigma*fd);    
K1[15+19*20]=make_cuDoubleComplex(0.0,vz1*sigma*fd);
K1[15+1*20]=make_cuDoubleComplex(kax*(-1.0)*sigma,0.0);
K1[15+4*20]=make_cuDoubleComplex(kay*sigma,0.0);
K1[15+10*20]=make_cuDoubleComplex(kaz*(-1.0)*sigma,0.0); 

//V i,j,k+1
K1[16+16*20]=make_cuDoubleComplex(0.0,(-1.0)*(vx1+vy1+vz1)*sigma*fd);    
K1[16+17*20]=make_cuDoubleComplex(0.0,vx1*sigma*fd);    
K1[16+19*20]=make_cuDoubleComplex(0.0,vy1*sigma*fd);    
K1[16+12*20]=make_cuDoubleComplex(0.0,vz1*sigma*fd);     
K1[16+2*20]=make_cuDoubleComplex(kax*(-1.0)*sigma,0.0);
K1[16+5*20]=make_cuDoubleComplex(kay*(-1.0)*sigma,0.0);
K1[16+8*20]=make_cuDoubleComplex(kaz*sigma,0.0);

//V i+1,j,k+1
K1[17+17*20]=make_cuDoubleComplex(0.0,(-1.0)*(vx1+vy1+vz1)*sigma*fd);    
K1[17+16*20]=make_cuDoubleComplex(0.0,vx1*sigma*fd);    
K1[17+18*20]=make_cuDoubleComplex(0.0,vy1*sigma*fd);    
K1[17+13*20]=make_cuDoubleComplex(0.0,vz1*sigma*fd);      
K1[17+2*20]=make_cuDoubleComplex(kax*sigma,0.0);
K1[17+7*20]=make_cuDoubleComplex(kay*(-1.0)*sigma,0.0);
K1[17+9*20]=make_cuDoubleComplex(kaz*sigma,0.0); 

//V i+1,j+1,k+1
K1[18+18*20]=make_cuDoubleComplex(0.0,(-1.0)*(vx1+vy1+vz1)*sigma*fd);    
K1[18+19*20]=make_cuDoubleComplex(0.0,vx1*sigma*fd);    
K1[18+17*20]=make_cuDoubleComplex(0.0,vy1*sigma*fd);    
K1[18+14*20]=make_cuDoubleComplex(0.0,vz1*sigma*fd);     
K1[18+3*20]=make_cuDoubleComplex(kax*sigma,0.0);
K1[18+7*20]=make_cuDoubleComplex(kay*sigma,0.0);
K1[18+11*20]=make_cuDoubleComplex(kaz*sigma,0.0);

//V i,j+1,k+1
K1[19+19*20]=make_cuDoubleComplex(0.0,(-1.0)*(vx1+vy1+vz1)*sigma*fd);    
K1[19+18*20]=make_cuDoubleComplex(0.0,vx1*sigma*fd);    
K1[19+16*20]=make_cuDoubleComplex(0.0,vy1*sigma*fd);    
K1[19+15*20]=make_cuDoubleComplex(0.0,vz1*sigma*fd);    
K1[19+3*20]=make_cuDoubleComplex(kax*(-1.0)*sigma,0.0);
K1[19+5*20]=make_cuDoubleComplex(kay*sigma,0.0);
K1[19+10*20]=make_cuDoubleComplex(kaz*sigma,0.0); 
}


__global__ void FDpart(int N, int nx, int ny, int nz, double *NK, int *EL, int *FD, double *Abg, double *m, double f,
                     cuDoubleComplex *AKv, int *AKc, int *AKr, cuDoubleComplex *AMv)
{
    int ii = blockIdx.x*blockDim.x + threadIdx.x;
    if (ii < N){
        int i,j,k,pp,kk,jj;
        double dxo,dyo,dzo;
        int no[20];
        int FDind;
        int totV=(nx)*(ny-1)*(nz-1)+(nx-1)*(ny)*(nz-1)+(nx-1)*(ny-1)*(nz)+(nx-1)*(ny-1)*(nz-1);   
        double b[40]={0.0};  
        double sinx[20]={0.0};
        double siny[20]={0.0};      
        int ind[20]={0};
        int ind2[20]={0};
        int c,c2;
        double sigma;
        
        i=EL[ii+N*20];
        j=EL[ii+N*21];
        k=EL[ii+N*22];
        pp=EL[ii+N*23];
        
        if(pp==(-1)){
        sigma=0.00000001;
        }else{
        sigma=m[pp-1];
        }
        
        FDind=(ny*nx)*(k-1)+ny*(i-1)+(j-1);
        getnos( i,  j,  k, nx,  ny,  nz, no);
        getdxyz(i, j, k, nx, ny, nz, NK, &dxo, &dyo, &dzo);
        
        cuDoubleComplex K[400]={make_cuDoubleComplex(0.0,0.0)};
        
        if(FD[FDind]==1){
        katsayiFD(dxo, dyo, dzo, K, sigma, f);
        }
        
        // b hazırlanır    
        c=0;c2=0;
        for (kk=0;kk<20;kk++){
        if(no[kk]>0){
          c2=c2+1;
          ind2[c2-1]=kk+1;
        }else{
          c=c+1;
           ind[c-1]=kk+1;
           if(kk+1<=4){
           sinx[c-1]=1.0;
           siny[c-1]=0.0;
           }else if(kk+1>4 && kk+1<=8){
           sinx[c-1]=0.0;
           siny[c-1]=1.0;
           }else{
           sinx[c-1]=0.0;
           siny[c-1]=0.0;
           }
        }
        }
        
        double dd;      
        if(c>0){
            for(kk=0;kk<c2;kk++){
                for (jj=0;jj<c;jj++){
                if(ind2[kk]<=12){
                dd=cuCreal(K[ind2[kk]-1 + 20*(ind[jj]-1)]);
                b[kk]=b[kk]-sinx[jj]*dd;
                b[kk+20]=b[kk+20]-siny[jj]*dd; 
                }
                }
            }
        }
        
        // b yerine yazılır    
        int al,ek;
        if(c>0){
            for(kk=0;kk<c2;kk++){
            al=no[ind2[kk]-1];
            atomicAdd(&Abg[(al-1)],b[kk]);
            atomicAdd(&Abg[(al-1)+totV],b[kk+20]);
            }
        }
        
        
        AKr[ii]=160*ii;
        if(ii==0){
        AKr[N]=160*N;
        }
        
        for(kk=0;kk<20;kk++){
            if(no[kk]<0){
                continue;
            }
            for(jj=0;jj<20;jj++){   
                if(no[jj]<0){
                continue;
                }
                
                 if(kk<4){
                 ek=kk;
                 }else if(kk>=4 && kk<8){
                    ek=kk-4;
                 }else if(kk>=8 && kk<12){
                    ek=kk-8;
                 }else{
                    ek=kk-12;
                 }
                
                AKv[(no[kk]-1)*160+jj+ek*20]=K[jj + 20*kk];
                AKc[(no[kk]-1)*160+jj+ek*20]=no[jj];
            }
        }
        
        if(FD[FDind]==1){
        katsayiFDMPRE(K);
        }    
       
        
        for(kk=0;kk<20;kk++){
            if(no[kk]<0){
                continue;
            }
            for(jj=0;jj<20;jj++){
                
            if(no[jj]<0){
            continue;
            }
            
             if(kk<4){
             ek=kk;
             }else if(kk>=4 && kk<8){
                ek=kk-4;
             }else if(kk>=8 && kk<12){
                ek=kk-8;
             }else{
                ek=kk-12;
             }
            AMv[(no[kk]-1)*160+jj+ek*20]=K[jj + 20*kk];
        }
        }    
    }
}

__global__ void mergesame2(int N, cuDoubleComplex *K0v, cuDoubleComplex *K1v,int *K0c,int *K0r)
{
    int ii = blockIdx.x*blockDim.x + threadIdx.x;
    if (ii < N){
        int kk,jj,c=0,al;
        int lst[160]={0};
        int lstc[160]={0};
            
        for (kk=0;kk<160;kk++){
            al=K0c[kk+ii*160];
            
            if(al==0 || al<0){
            K0v[kk+ii*160]=make_cuDoubleComplex(0.0,0.0);
            K1v[kk+ii*160]=make_cuDoubleComplex(0.0,0.0);                
            K0c[kk+ii*160]=1;
            continue;
            }
            
            if(c==0){
            c=c+1;
            lst[c-1]=al;
            lstc[c-1]=kk;
            continue;
            }
                
            for (jj=0;jj<c;jj++){
                if(al==lst[jj]){
                K0v[lstc[jj]+ii*160]=cuCadd(K0v[kk+ii*160],K0v[lstc[jj]+ii*160]);
                K0v[kk+ii*160]=make_cuDoubleComplex(0.0,0.0);
                K1v[lstc[jj]+ii*160]=cuCadd(K1v[kk+ii*160],K1v[lstc[jj]+ii*160]);
                K1v[kk+ii*160]=make_cuDoubleComplex(0.0,0.0);                    
                K0c[kk+ii*160]=1;
                continue;
                }
            }
            c=c+1;
            lst[c-1]=al;
            lstc[c-1]=kk;
        }
        K0r[ii]=160*ii;
        if(ii==0){
        K0r[N]=160*N;
        }
        
        for (kk=0;kk<160;kk++){
        K0c[kk+ii*160]=K0c[kk+ii*160]-1;
        }
    }
}

void mexFunction(int nleft, mxArray * plhs[], int nright, const mxArray * prhs[])
{

mxGPUArray const *NK= mxGPUCreateFromMxArray(prhs[0]);
double* d_NK = (double*)mxGPUGetDataReadOnly(NK);

mxGPUArray const *EL= mxGPUCreateFromMxArray(prhs[1]);
int* d_EL = (int*)mxGPUGetDataReadOnly(EL);

mxGPUArray const *FD= mxGPUCreateFromMxArray(prhs[2]);
int* d_FD = (int*)mxGPUGetDataReadOnly(FD);
mwSize nEls=mxGPUGetNumberOfElements(FD);

mxGPUArray const *FE= mxGPUCreateFromMxArray(prhs[3]);
int* d_FE = (int*)mxGPUGetDataReadOnly(FE);
    
int* nx = (int*)mxGetPr(prhs[4]);
int* ny =(int*) mxGetPr(prhs[5]);
int* nz = (int*)mxGetPr(prhs[6]); 

double* f = (double*)mxGetPr(prhs[7]); 
   
mxGPUArray const *m= mxGPUCreateFromMxArray(prhs[8]);
double* d_m = (double*)mxGPUGetDataReadOnly(m);
const int blocksize=256;

cuDoubleComplex tol = make_cuDoubleComplex(0.0,0.0);
int nzn;    
size_t lworkInBytes = 0;
char *d_work = NULL;
int *nnzPerRowY;
cuDoubleComplex onec=make_cuDoubleComplex(1.0,0.0);    
    
cusparseHandle_t cusparsehandle = NULL;
cusparseCreate(&cusparsehandle);
cusparseMatDescr_t descrA;
cusparseCreateMatDescr(&descrA);    
cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);

mwSize N=(*nx-1)*(*ny-1)*(*nz-1)+(*nx)*(*ny-1)*(*nz-1)+(*nx-1)*(*ny)*(*nz-1)+(*nx-1)*(*ny-1)*(*nz);
    
//col ve val'ler    
mwSize dims[2]={160,N};
mxGPUArray *tempv1=mxGPUCreateGPUArray(2,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);
cuDoubleComplex* d_tempv1 = (cuDoubleComplex*)mxGPUGetData(tempv1);
mxGPUArray *tempv2=mxGPUCreateGPUArray(2,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);
cuDoubleComplex* d_tempv2 = (cuDoubleComplex*)mxGPUGetData(tempv2);    
mxGPUArray *tempc1=mxGPUCreateGPUArray(2,dims,mxINT32_CLASS,mxREAL,MX_GPU_INITIALIZE_VALUES);
int* d_tempc1= (int*)mxGPUGetData(tempc1);    
dims[0]=1; dims[1]=N+1;    
mxGPUArray *tempr1=mxGPUCreateGPUArray(2,dims,mxINT32_CLASS,mxREAL,MX_GPU_INITIALIZE_VALUES);
int* d_tempr1 = (int*)mxGPUGetData(tempr1);


dims[0]=N;dims[1]=2;
mxGPUArray *b=mxGPUCreateGPUArray(2,dims,mxDOUBLE_CLASS,mxREAL,MX_GPU_INITIALIZE_VALUES);
double* d_b = (double*)mxGPUGetData(b);

// FD PART
FDpart<<<(nEls+(blocksize-1))/blocksize, blocksize>>>(nEls,  *nx,  *ny,  *nz, d_NK, d_EL, d_FD, d_b, d_m, *f,
                     d_tempv1, d_tempc1, d_tempr1, d_tempv2);

FEpart<<<(nEls+(blocksize-1))/blocksize, blocksize>>>(nEls,  *nx,  *ny,  *nz, d_NK, d_EL, d_FE, d_b, d_m, *f,
                     d_tempv1, d_tempc1, d_tempr1, d_tempv2);


mergesame2<<<(N+(blocksize-1))/blocksize, blocksize>>>( N, d_tempv1, d_tempv2, d_tempc1, d_tempr1);

csru2csrInfo_t infoc;
cusparseCreateCsru2csrInfo(&infoc);

//STIFFNESS ALL
cudaMallocManaged(&nnzPerRowY, sizeof(int) * N ); 
cusparseZnnz_compress(cusparsehandle, N, descrA, d_tempv1, d_tempr1, nnzPerRowY, &nzn, tol);
dims[0]=nzn; dims[1]=1;        
mxGPUArray *Kc=mxGPUCreateGPUArray(2,dims,mxINT32_CLASS,mxREAL,MX_GPU_INITIALIZE_VALUES);
int* d_Kc = (int*)mxGPUGetDataReadOnly(Kc);
dims[0]=N+1; dims[1]=1;        
mxGPUArray *Kr=mxGPUCreateGPUArray(2,dims,mxINT32_CLASS,mxREAL,MX_GPU_INITIALIZE_VALUES);
int* d_Kr = (int*)mxGPUGetDataReadOnly(Kr);    
dims[0]=nzn;dims[1]=1;        
mxGPUArray *Kv=mxGPUCreateGPUArray(2,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES); 
cuDoubleComplex* d_Kv = (cuDoubleComplex*)mxGPUGetDataReadOnly(Kv);    
cusparseZcsr2csr_compress( cusparsehandle, N, N, descrA, d_tempv1, d_tempc1, d_tempr1, 160*N, nnzPerRowY, d_Kv, d_Kc, d_Kr, tol);
cusparseZcsru2csr_bufferSizeExt(cusparsehandle, N, N, nzn, d_Kv, d_Kr, d_Kc, infoc, &lworkInBytes);
cudaMalloc(&d_work, sizeof(char)* lworkInBytes);
cusparseZcsru2csr(cusparsehandle, N, N, nzn, descrA, d_Kv, d_Kr, d_Kc, infoc, d_work);
cudaFree(nnzPerRowY);cudaFree(d_work);

mxGPUDestroyGPUArray(tempv1);
    
//PRECONDITIONER
cudaMallocManaged(&nnzPerRowY, sizeof(int) * N ); 
cusparseZnnz_compress(cusparsehandle, N, descrA, d_tempv2, d_tempr1, nnzPerRowY, &nzn, tol);
dims[0]=nzn; dims[1]=1;        
mxGPUArray *Mc=mxGPUCreateGPUArray(2,dims,mxINT32_CLASS,mxREAL,MX_GPU_INITIALIZE_VALUES);
int* d_Mc = (int*)mxGPUGetDataReadOnly(Mc);
dims[0]=N+1; dims[1]=1;        
mxGPUArray *Mr=mxGPUCreateGPUArray(2,dims,mxINT32_CLASS,mxREAL,MX_GPU_INITIALIZE_VALUES);
int* d_Mr = (int*)mxGPUGetDataReadOnly(Mr);    
dims[0]=nzn;dims[1]=1;        
mxGPUArray *Mv=mxGPUCreateGPUArray(2,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES); 
cuDoubleComplex* d_Mv = (cuDoubleComplex*)mxGPUGetDataReadOnly(Mv);    
cusparseZcsr2csr_compress( cusparsehandle, N, N, descrA, d_tempv2, d_tempc1, d_tempr1, 160*N, nnzPerRowY, d_Mv, d_Mc, d_Mr, tol);
cusparseZcsru2csr_bufferSizeExt(cusparsehandle, N, N, nzn, d_Mv, d_Mr, d_Mc, infoc, &lworkInBytes);
cudaMalloc(&d_work, sizeof(char)* lworkInBytes);
cusparseZcsru2csr(cusparsehandle, N, N, nzn, descrA, d_Mv, d_Mr, d_Mc, infoc, d_work);
cudaFree(nnzPerRowY);cudaFree(d_work);

mxGPUDestroyGPUArray(tempv2);mxGPUDestroyGPUArray(tempc1);mxGPUDestroyGPUArray(tempr1);

plhs[0] = mxGPUCreateMxArrayOnGPU(Kv);
plhs[1] = mxGPUCreateMxArrayOnGPU(Kc);
plhs[2] = mxGPUCreateMxArrayOnGPU(Kr);   
plhs[3] = mxGPUCreateMxArrayOnGPU(Mv);
plhs[4] = mxGPUCreateMxArrayOnGPU(Mc);
plhs[5] = mxGPUCreateMxArrayOnGPU(Mr);     
plhs[6] = mxGPUCreateMxArrayOnGPU(b);     

mxGPUDestroyGPUArray(b);

cusparseDestroy(cusparsehandle);
cusparseDestroyMatDescr(descrA);
mxGPUDestroyGPUArray(Kv);mxGPUDestroyGPUArray(Kr);mxGPUDestroyGPUArray(Kc);
mxGPUDestroyGPUArray(Mv);mxGPUDestroyGPUArray(Mr);mxGPUDestroyGPUArray(Mc);

mxGPUDestroyGPUArray(NK);mxGPUDestroyGPUArray(FE);mxGPUDestroyGPUArray(FD);
mxGPUDestroyGPUArray(m);mxGPUDestroyGPUArray(EL);
cusparseDestroyCsru2csrInfo(infoc);

cudaDeviceSynchronize();


return;


}


