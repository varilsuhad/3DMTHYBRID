//Created by Deniz Varilsuha
//email: deniz.varilsuha@itu.edu.tr

// To compile the code use the following line in Matlab's command line (change the paths if necessary)
// mexcuda -R2018a BlockGPBiCG.cu -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\include" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\lib\x64" NVCCFLAGS='"-Wno-deprecated-gpu-targets --gpu-architecture=compute_61  --gpu-code=sm_61,sm_86,sm_89 -use_fast_math -extra-device-vectorization"' -lcusparse -lcublas

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"

void mexFunction(int nleft, mxArray * plhs[], int nright, const mxArray * prhs[])
{

if(nright!=11){
    printf("right hand side is incorrect\n");
    return;
}

mxGPUArray const *row_csr = mxGPUCreateFromMxArray(prhs[0]);
mxGPUArray const *col = mxGPUCreateFromMxArray(prhs[1]);
mxGPUArray const *val = mxGPUCreateFromMxArray(prhs[2]);
mxGPUArray const *b = mxGPUCreateFromMxArray(prhs[3]);
mxGPUArray const *row_csrM = mxGPUCreateFromMxArray(prhs[4]);
mxGPUArray const *colM = mxGPUCreateFromMxArray(prhs[5]);
mxGPUArray const *valM = mxGPUCreateFromMxArray(prhs[6]);
double* tol = mxGetPr(prhs[7]);
double* maxiter = mxGetPr(prhs[8]);
double* stagdetect = mxGetPr(prhs[9]);
mxGPUArray *xi = (mxGPUArray*)mxGPUCreateFromMxArray(prhs[10]);

mwSize N=mxGPUGetNumberOfElements(row_csr)-1;
mwSize ndim = 2;
mwSize dims[2] = {2*N,1};
mwSize nnzM = mxGPUGetNumberOfElements(valM);
mwSize nnz = mxGPUGetNumberOfElements(val);
mwSize dims2[2] = {nnzM,1};

int i,c=0,is;
double res[2000];
double nb,nAx;   
const cuDoubleComplex* d_b = (cuDoubleComplex*)mxGPUGetDataReadOnly(b);   
double resl=1000;

const int* d_row_csrM = (int*)mxGPUGetDataReadOnly(row_csrM);
const int* d_colM = (int*)mxGPUGetDataReadOnly(colM);
const int* d_row_csr = (int*)mxGPUGetDataReadOnly(row_csr);
const int* d_col = (int*)mxGPUGetDataReadOnly(col);

mxGPUArray *y=mxGPUCreateGPUArray(ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);
mxGPUArray *xr=mxGPUCreateGPUArray(ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES); //return x values
mxGPUArray *x0=mxGPUCreateGPUArray(ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);
mxGPUArray *r0=mxGPUCreateGPUArray(ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);    
mxGPUArray *z=mxGPUCreateGPUArray(ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);
mxGPUArray *t=mxGPUCreateGPUArray(ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);
mxGPUArray *wp=mxGPUCreateGPUArray(ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);
mxGPUArray *u=mxGPUCreateGPUArray(ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);
mxGPUArray *p=mxGPUCreateGPUArray(ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);
mxGPUArray *tp=mxGPUCreateGPUArray(ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);
mxGPUArray *Ap=mxGPUCreateGPUArray(ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);
mxGPUArray *AMp=mxGPUCreateGPUArray(ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);
mxGPUArray *tp0=mxGPUCreateGPUArray(ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);
mxGPUArray *At=mxGPUCreateGPUArray(ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);
mxGPUArray *r=mxGPUCreateGPUArray(ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);
mxGPUArray *r0p=mxGPUCreateGPUArray(ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);
mxGPUArray *r0b=mxGPUCreateGPUArray(ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);

cuDoubleComplex* d_r0p = (cuDoubleComplex*)mxGPUGetData(r0p);   
cuDoubleComplex* d_r0b = (cuDoubleComplex*)mxGPUGetData(r0b);
cuDoubleComplex* d_wp = (cuDoubleComplex*)mxGPUGetData(wp);   
cuDoubleComplex* d_u = (cuDoubleComplex*)mxGPUGetData(u);
cuDoubleComplex* d_p = (cuDoubleComplex*)mxGPUGetData(p);   
cuDoubleComplex* d_tp = (cuDoubleComplex*)mxGPUGetData(tp);
cuDoubleComplex* d_Ap = (cuDoubleComplex*)mxGPUGetData(Ap);    
cuDoubleComplex* d_AMp = (cuDoubleComplex*)mxGPUGetData(AMp);
cuDoubleComplex* d_tp0 = (cuDoubleComplex*)mxGPUGetData(tp0);
cuDoubleComplex* d_At = (cuDoubleComplex*)mxGPUGetData(At);
cuDoubleComplex* d_r = (cuDoubleComplex*)mxGPUGetData(r);
cuDoubleComplex* d_r0 = (cuDoubleComplex*)mxGPUGetData(r0);   
cuDoubleComplex* d_y = (cuDoubleComplex*)mxGPUGetData(y);   
cuDoubleComplex* d_valM = (cuDoubleComplex*)mxGPUGetDataReadOnly(valM);
cuDoubleComplex* d_val = (cuDoubleComplex*)mxGPUGetDataReadOnly(val);    
cuDoubleComplex* d_x0 = (cuDoubleComplex*)mxGPUGetData(x0); 
cuDoubleComplex* d_xi = (cuDoubleComplex*)mxGPUGetData(xi);     
cuDoubleComplex* d_xr = (cuDoubleComplex*)mxGPUGetData(xr);        
cuDoubleComplex* d_z = (cuDoubleComplex*)mxGPUGetData(z);   
cuDoubleComplex* d_t = (cuDoubleComplex*)mxGPUGetData(t);

cuDoubleComplex beta = make_cuDoubleComplex(0.0f, 0.0f);
cuDoubleComplex ro0 = make_cuDoubleComplex(1.0f, 0.0f);
cuDoubleComplex ro1 = make_cuDoubleComplex(1.0f, 0.0f);
cuDoubleComplex ro2 = make_cuDoubleComplex(1.0f, 0.0f);
cuDoubleComplex ro3 = make_cuDoubleComplex(1.0f, 0.0f);
cuDoubleComplex ro4 = make_cuDoubleComplex(1.0f, 0.0f);
cuDoubleComplex ro5 = make_cuDoubleComplex(1.0f, 0.0f);
cuDoubleComplex alt = make_cuDoubleComplex(1.0f, 0.0f);
cuDoubleComplex ust1 = make_cuDoubleComplex(1.0f, 0.0f);
cuDoubleComplex ust2 = make_cuDoubleComplex(1.0f, 0.0f);

cuDoubleComplex alfa = make_cuDoubleComplex(1.0f, 0.0f);
cuDoubleComplex minusone = make_cuDoubleComplex(-1.0f, 0.0f);    
cuDoubleComplex w = make_cuDoubleComplex(0.0f, 0.0f);
cuDoubleComplex nu = make_cuDoubleComplex(0.0f, 0.0f);
cuDoubleComplex one = make_cuDoubleComplex(1.0f, 0.0f);
cuDoubleComplex zero = make_cuDoubleComplex(0.0f, 0.0f);
cuDoubleComplex nalfa = make_cuDoubleComplex(1.0f, 0.0f);
   
mxGPUArray *zara=mxGPUCreateGPUArray(ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);     
mxGPUArray *valM2s=mxGPUCreateGPUArray(ndim,dims2,mxDOUBLE_CLASS,mxCOMPLEX,MX_GPU_INITIALIZE_VALUES);    
cuDoubleComplex* d_zara = (cuDoubleComplex*)mxGPUGetData(zara);
cuDoubleComplex* d_valM2s = (cuDoubleComplex*)mxGPUGetData(valM2s);        

// CUBLAS APIs
cublasHandle_t cublashandle=NULL;
cublasStatus_t status;
status = cublasCreate(&cublashandle);
if (status != CUBLAS_STATUS_SUCCESS) {printf("!!!! CUBLAS initialization error\n");return;}

// CUSPARSE APIs
cusparseHandle_t cusparsehandle = NULL;
cusparseStatus_t status2;
status2=cusparseCreate(&cusparsehandle);
if (status2 != CUSPARSE_STATUS_SUCCESS) {printf("cusparse initialization error\n");return;}
cusparseMatDescr_t descr_M = 0;
csrilu02Info_t info_M  = 0;
int pBufferSize_M;
int pBufferSize_A;
size_t bufferSizeL,bufferSizeU;    
void *pBuffer = 0,*d_bufferLU,*d_bufferLs,*d_bufferUs;
// descriptor'lar
status2=cusparseCreateMatDescr(&descr_M);
status2=cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
status2=cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);
status2=cusparseCreateCsrilu02Info(&info_M);
   
cusparseFillMode_t fill_lower    = CUSPARSE_FILL_MODE_LOWER;
cusparseDiagType_t diag_unit     = CUSPARSE_DIAG_TYPE_UNIT;
cusparseFillMode_t fill_upper    = CUSPARSE_FILL_MODE_UPPER;
cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

cusparseSpSMDescr_t spsvDescrLs, spsvDescrUs;
    
cusparseSpMatDescr_t matA;
cusparseDnMatDescr_t matp;
cusparseDnMatDescr_t mattp, matAt;
cusparseDnMatDescr_t matx0, matr0,matr0p,matAMp;
cusparseSpMatDescr_t matM_lowers, matM_uppers;
cusparseDnMatDescr_t matAp,matzara;
    
status2=cusparseCreateCsr(&matA, N, N, nnz, (void*)d_row_csr, (void*)d_col, d_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F) ;
status2=cusparseCreateDnMat(&matp, N,2,N, d_p, CUDA_C_64F,CUSPARSE_ORDER_COL);
status2=cusparseCreateDnMat(&matAp, N,2,N, d_Ap, CUDA_C_64F,CUSPARSE_ORDER_COL);
status2=cusparseCreateDnMat(&mattp, N,2,N, d_tp, CUDA_C_64F,CUSPARSE_ORDER_COL);
status2=cusparseCreateDnMat(&matAt, N,2,N, d_At, CUDA_C_64F,CUSPARSE_ORDER_COL);
status2=cusparseCreateDnMat(&matx0, N,2,N, d_x0, CUDA_C_64F,CUSPARSE_ORDER_COL);
status2=cusparseCreateDnMat(&matr0, N,2,N, d_r0, CUDA_C_64F,CUSPARSE_ORDER_COL);
status2=cusparseCreateDnMat(&matr0p, N,2,N, d_r0p, CUDA_C_64F,CUSPARSE_ORDER_COL);
status2=cusparseCreateDnMat(&matAMp, N,2,N, d_AMp, CUDA_C_64F,CUSPARSE_ORDER_COL);
status2=cusparseCreateDnMat(&matzara, N,2,N, d_zara, CUDA_C_64F,CUSPARSE_ORDER_COL);

cusparseSpMMAlg_t alg=CUSPARSE_SPMM_CSR_ALG1;  
cusparseSpSMAlg_t alg2=CUSPARSE_SPSM_ALG_DEFAULT;

status2=cusparseSpMM_bufferSize(cusparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, matp, &zero, matAp, CUDA_C_64F, alg , (size_t*)&pBufferSize_A);
if (status2 != CUSPARSE_STATUS_SUCCESS) {printf("cusparse Ax buffer error\n");return;}
cudaMalloc((void**)&pBuffer, pBufferSize_A);

status2=cusparseSpMM_preprocess(cusparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, mattp, &zero, matAt, CUDA_C_64F, alg , pBuffer);

cusparseMatDescr_t matLU;    
cusparseCreateMatDescr(&matLU);
cusparseSetMatType(matLU, CUSPARSE_MATRIX_TYPE_GENERAL);
cusparseSetMatIndexBase(matLU, CUSPARSE_INDEX_BASE_ZERO);

status=cublasZcopy(cublashandle, nnzM, d_valM, 1, d_valM2s, 1);

status2=cusparseZcsrilu02_bufferSize(cusparsehandle, N, nnzM, matLU, d_valM2s, d_row_csrM, d_colM, info_M, &pBufferSize_M);
if (status2 != CUSPARSE_STATUS_SUCCESS) { printf("cusparse ilu error\n");return;}
cudaMalloc((void**)&d_bufferLU, pBufferSize_M);

cusparseZcsrilu02_analysis( cusparsehandle, N, nnzM, descr_M, d_valM2s, d_row_csrM, d_colM, info_M, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU);
cusparseZcsrilu02( cusparsehandle, N, nnzM, matLU, d_valM2s, d_row_csrM, d_colM, info_M, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU);

cusparseSpSM_createDescr(&spsvDescrLs);
cusparseSpSM_createDescr(&spsvDescrUs);    

///////////////////////////////

cusparseCreateCsr(&matM_lowers, N, N, nnzM, (void*)d_row_csrM, (void*)d_colM, d_valM2s, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);
cusparseSpMatSetAttribute(matM_lowers,CUSPARSE_SPMAT_FILL_MODE, &fill_lower, sizeof(fill_lower));
cusparseSpMatSetAttribute(matM_lowers, CUSPARSE_SPMAT_DIAG_TYPE, &diag_unit, sizeof(diag_unit));

cusparseCreateCsr(&matM_uppers, N, N, nnzM, (void*)d_row_csrM, (void*)d_colM, d_valM2s, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);
cusparseSpMatSetAttribute(matM_uppers, CUSPARSE_SPMAT_FILL_MODE, &fill_upper, sizeof(fill_upper));
cusparseSpMatSetAttribute(matM_uppers, CUSPARSE_SPMAT_DIAG_TYPE, &diag_non_unit,  sizeof(diag_non_unit)); 

cusparseSpSM_bufferSize( cusparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matM_lowers, matr0, matzara, CUDA_C_64F, alg2, spsvDescrLs, &bufferSizeL);
cudaMalloc(&d_bufferLs, bufferSizeL);
cusparseSpSM_bufferSize(cusparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matM_uppers, matr0, matzara, CUDA_C_64F, alg2, spsvDescrUs, &bufferSizeU);
cudaMalloc(&d_bufferUs, bufferSizeU);   

cusparseSpSM_analysis(cusparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matM_lowers, matr0, matzara, CUDA_C_64F, alg2, spsvDescrLs, d_bufferLs);
cusparseSpSM_analysis(cusparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matM_uppers, matr0, matzara, CUDA_C_64F, alg2, spsvDescrUs, d_bufferUs);
    
/////////////////////////////////

// xi ->x0 kopyala
status=cublasZcopy(cublashandle, 2*N, d_xi, 1, d_x0, 1);

// nb=norm(b);
status=cublasDznrm2(cublashandle,2*N, d_b, 1, &nb);

// r0=(b-A*x0);
status=cublasZcopy(cublashandle, 2*N, d_b, 1, d_r0, 1);
status2=cusparseSpMM(cusparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE, &minusone, matA, matx0, &one, matr0, CUDA_C_64F, alg , pBuffer);    

cusparseSpSM_solve(cusparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matM_lowers, matr0, matzara, CUDA_C_64F, alg2, spsvDescrLs);
cusparseSpSM_solve(cusparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matM_uppers, matzara, matr0p, CUDA_C_64F, alg2, spsvDescrUs);

//r0b=M2\(M1\r0p); //modladım //r0b=r0p
status=cublasZcopy(cublashandle, 2*N, d_r0p, 1, d_r0b, 1);

// FOR LOOP
for (i=0;i<(int)(*maxiter);++i)
{
//p=r0p+beta*(p-u);
status=cublasZaxpy(cublashandle, 2*N, &minusone, d_u, 1, d_p, 1);
status=cublasZscal(cublashandle, 2*N, &beta, d_p, 1);
status=cublasZaxpy(cublashandle, 2*N, &one, d_r0p, 1, d_p, 1);
//Ap=A*p;
status2=cusparseSpMM(cusparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, matp, &zero, matAp, CUDA_C_64F, alg , pBuffer);
  
cusparseSpSM_solve(cusparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matM_lowers, matAp, matzara, CUDA_C_64F, alg2, spsvDescrLs);
cusparseSpSM_solve(cusparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matM_uppers, matzara, matAMp, CUDA_C_64F, alg2, spsvDescrUs);

// alfa=(r0b'*r0)/(r0b'*Ap);
status=cublasZdotc(cublashandle,2*N, d_r0b, 1, d_r0, 1, &ro0);    
status=cublasZdotc(cublashandle,2*N, d_r0b, 1, d_Ap, 1, &ro1);    
alfa=cuCdiv(ro0,ro1);

//y=t-r0+alfa*(Ap-wp);
status=cublasZcopy(cublashandle, 2*N, d_Ap, 1, d_y, 1);
status=cublasZaxpy(cublashandle, 2*N, &minusone, d_wp, 1, d_y, 1);
status=cublasZscal(cublashandle, 2*N, &alfa, d_y, 1);
status=cublasZaxpy(cublashandle, 2*N, &minusone, d_r0, 1, d_y, 1);
status=cublasZaxpy(cublashandle, 2*N, &one, d_t, 1, d_y, 1);

//t=r0-alfa*Ap;
status=cublasZcopy(cublashandle, 2*N, d_r0, 1, d_t, 1);
nalfa=cuCmul(minusone,alfa);
status=cublasZaxpy(cublashandle, 2*N, &nalfa, d_Ap, 1, d_t, 1);

//tp0=tp;
status=cublasZcopy(cublashandle, 2*N, d_tp, 1, d_tp0, 1);

//tp=r0p-alfa*AMp;
status=cublasZcopy(cublashandle, 2*N, d_r0p, 1, d_tp, 1);
status=cublasZaxpy(cublashandle, 2*N, &nalfa, d_AMp, 1, d_tp, 1);
    
//At=A*tp;
status2=cusparseSpMM(cusparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, mattp, &zero, matAt, CUDA_C_64F, alg , pBuffer);
// if (status2 != CUSPARSE_STATUS_SUCCESS) {printf("cusparse Ax carpim  error\n");return;}

if(i==0){
//w=(At'*t)/(At'*At);   n=0;    
status=cublasZdotc(cublashandle,2*N, d_At, 1, d_t, 1, &ro0);    
status=cublasZdotc(cublashandle,2*N, d_At, 1, d_At, 1, &ro1);   
w=cuCdiv(ro0,ro1);
nu = make_cuDoubleComplex(0.0f, 0.0f);

}else{
//w=((y'*y)*(At'*t)-(y'*t)*(At'*y))/((At'*At)*(y'*y)-(y'*At)*(At'*y));
//n=((At'*At)*(y'*t)-(y'*At)*(At'*t))/((At'*At)*(y'*y)-(y'*At)*(At'*y));    
status=cublasZdotc(cublashandle,2*N, d_y, 1, d_y, 1, &ro0);    
status=cublasZdotc(cublashandle,2*N, d_At, 1, d_t, 1, &ro1);    
status=cublasZdotc(cublashandle,2*N, d_y, 1, d_t, 1, &ro2);    
status=cublasZdotc(cublashandle,2*N, d_At, 1, d_y, 1, &ro3);    
status=cublasZdotc(cublashandle,2*N, d_At, 1, d_At, 1, &ro4);    
status=cublasZdotc(cublashandle,2*N, d_y, 1, d_At, 1, &ro5);    

alt=cuCsub(cuCmul(ro4,ro0),cuCmul(ro5,ro3));
ust1=cuCsub(cuCmul(ro0,ro1),cuCmul(ro2,ro3));
ust2=cuCsub(cuCmul(ro4,ro2),cuCmul(ro5,ro1));

w=cuCdiv(ust1,alt);  
nu=cuCdiv(ust2,alt);    
}

//u=w*AMp+n*(tp0-r0p+beta*u);
status=cublasZscal(cublashandle, 2*N, &beta, d_u, 1);
status=cublasZaxpy(cublashandle, 2*N, &minusone, d_r0p, 1, d_u, 1);
status=cublasZaxpy(cublashandle, 2*N, &one, d_tp0, 1, d_u, 1);
status=cublasZscal(cublashandle, 2*N, &nu, d_u, 1);
status=cublasZaxpy(cublashandle, 2*N, &w, d_AMp, 1, d_u, 1);

//z=w*r0p+n*z-alfa*u;
status=cublasZscal(cublashandle, 2*N, &nu, d_z, 1);
status=cublasZaxpy(cublashandle, 2*N, &w, d_r0p, 1, d_z, 1);
status=cublasZaxpy(cublashandle, 2*N, &nalfa, d_u, 1, d_z, 1);

//x0=x0+alfa*p+z;
status=cublasZaxpy(cublashandle, 2*N, &alfa, d_p, 1, d_x0, 1);
status=cublasZaxpy(cublashandle, 2*N, &one, d_z, 1, d_x0, 1);

//r=r0;
status=cublasZcopy(cublashandle, 2*N, d_r0, 1, d_r, 1);

//r0=t-n*y-w*At;
status=cublasZcopy(cublashandle, 2*N, d_t, 1, d_r0, 1);
ro0=cuCmul(minusone,nu);
ro1=cuCmul(minusone,w);
status=cublasZaxpy(cublashandle, 2*N, &ro0, d_y, 1, d_r0, 1);
status=cublasZaxpy(cublashandle, 2*N, &ro1, d_At, 1, d_r0, 1);

// //r0p=M2\(M1\r0);
cusparseSpSM_solve(cusparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matM_lowers, matr0, matzara, CUDA_C_64F, alg2, spsvDescrLs);
cusparseSpSM_solve(cusparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matM_uppers, matzara, matr0p, CUDA_C_64F, alg2, spsvDescrUs);

//beta=alfa/w*(r0b'*r0)/(r0b'*r);
status=cublasZdotc(cublashandle, 2*N, d_r0b, 1, d_r0, 1, &ro0);    
status=cublasZdotc(cublashandle, 2*N, d_r0b, 1, d_r, 1, &ro1);    
ro2=cuCdiv(alfa,w);
ro3=cuCdiv(ro0,ro1);
beta=cuCmul(ro2,ro3);    

//wp=At+beta*Ap;
status=cublasZcopy(cublashandle, 2*N, d_At, 1, d_wp, 1);
status=cublasZaxpy(cublashandle, 2*N, &beta, d_Ap, 1, d_wp, 1);

// relres=norm(r0)/nb;
status=cublasDznrm2(cublashandle, 2*N, d_r0, 1, &nAx);
res[i]=nAx/nb;


if(res[i]<=*tol){
    status=cublasZcopy(cublashandle, 2*N, d_x0, 1, d_xr, 1);
    is=i;
    break;
}

// x0->xr    
if(resl>res[i]){
status=cublasZcopy(cublashandle, 2*N, d_x0, 1, d_xr, 1);
resl=res[i];
c=0;
is=i;    
}else{
c=c+1;
  if(c>(int)(*stagdetect)){
      break;
    }
   
}
}

plhs[0] = mxGPUCreateMxArrayOnGPU(xr);

int nn;
if(i==(int)(*maxiter)){    
nn=i;
}else{
nn=i+1;
}

plhs[1] = mxCreateDoubleMatrix(nn, 1, mxREAL);
double* py = mxGetPr(plhs[1]);
for (i=0;i<nn;++i){
py[i]=res[i];
}

plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
double* py2 = mxGetPr(plhs[2]);
py2[0]=res[is];

cudaFree(d_bufferLs);
cudaFree(d_bufferUs);
cudaFree(d_bufferLU);

cusparseSpSM_destroyDescr(spsvDescrUs);   
cusparseSpSM_destroyDescr(spsvDescrLs);    
cusparseDestroyMatDescr(matLU);

mxGPUDestroyGPUArray(xi);    
mxGPUDestroyGPUArray(row_csr);
mxGPUDestroyGPUArray(col);
mxGPUDestroyGPUArray(val);
mxGPUDestroyGPUArray(b);
mxGPUDestroyGPUArray(row_csrM);
mxGPUDestroyGPUArray(colM);
mxGPUDestroyGPUArray(valM);

mxGPUDestroyGPUArray(zara);
mxGPUDestroyGPUArray(valM2s); 
mxGPUDestroyGPUArray(y);
mxGPUDestroyGPUArray(xr);    
mxGPUDestroyGPUArray(x0);
mxGPUDestroyGPUArray(r0);
mxGPUDestroyGPUArray(z);
mxGPUDestroyGPUArray(t);
mxGPUDestroyGPUArray(wp);
mxGPUDestroyGPUArray(u);
mxGPUDestroyGPUArray(p);
mxGPUDestroyGPUArray(tp);
mxGPUDestroyGPUArray(Ap);
mxGPUDestroyGPUArray(AMp);
mxGPUDestroyGPUArray(tp0);
mxGPUDestroyGPUArray(At);
mxGPUDestroyGPUArray(r);
mxGPUDestroyGPUArray(r0p);
mxGPUDestroyGPUArray(r0b);

cusparseDestroySpMat(matA);
cusparseDestroySpMat(matM_lowers);
cusparseDestroySpMat(matM_uppers);    
cusparseDestroyDnMat(matp);    
cusparseDestroyDnMat(matAp);    
cusparseDestroyDnMat(mattp);    
cusparseDestroyDnMat(matAt);    
cusparseDestroyDnMat(matx0);    
cusparseDestroyDnMat(matr0);    
cusparseDestroyDnMat(matr0p);    
cusparseDestroyDnMat(matAMp);    
cusparseDestroyDnMat(matzara);    
    
cusparseDestroyMatDescr(descr_M);
cusparseDestroyCsrilu02Info(info_M);

status=cublasDestroy(cublashandle);
if (status != CUBLAS_STATUS_SUCCESS) {printf("!!!! cublasDestroy error\n");return;}
status2=cusparseDestroy(cusparsehandle);
if (status2 != CUSPARSE_STATUS_SUCCESS) {printf("!!!! cusparse destroy error\n");return;}
cudaFree(pBuffer);

cudaDeviceSynchronize();
return;
}


