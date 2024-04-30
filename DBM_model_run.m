% Author: Deniz Varilsuha
% Date of creation: 29/02/2024
% 

clear all;clc;close all;



sw=0; % pure FD
% sw=1; % pure FE
% sw=2; % hybrid


load('DBMmodel.mat');

d=gpuDevice(1);
d.CachePolicy='minimum';


% Switch for the 3 cases
if (sw==0)  % pure FD
    FD=FD0;
    FE=FE0;
elseif( sw==1) % pure FE
    FD=FD1;
    FE=FE1;    
elseif(sw==2) % hybrid
    FD=FD2;
    FE=FE2;
end

FDr=nnz(FD)/numel(FD)*100;
FEr=nnz(FE)/numel(FE)*100;
fprintf('FD ratio=%f and FE ratio=%f\n',FDr,FEr);


% Load the variables into the GPU
NKg=gpuArray(NK);
ELg=gpuArray(int32(EL));
FDg=gpuArray(int32(FD));
FEg=gpuArray(int32(FE));
[ny,nx,nz]=size(FE);
mg=gpuArray(m);

maxit=400; %maximum iteration
stagdetect=400; 
stopcrit=10^-9;

% return
tot=0;
tots=0;
for i=1:length(f)
%Create the matrices to solve Ax=b for 2 polarizations with the preconditioner matrix M    
[Aval,Acol,Arow,Mval,Mcol,Mrow,bg]=createStiffnessMatrix(NKg,(ELg),(FDg),(FEg),int32(nx),int32(ny),int32(nz),double(f(i)),mg); 
fprintf('Matrices are assembled\n');

%Starting initial x vector for iterative solution
xinit=gpuArray(zeros(size(bg)));

%The iterative solver solves both polarizations
st=tic;
[x,r,relres]=BlockGPBiCG(Arow,Acol,Aval,complex(bg),Mrow,Mcol,Mval,stopcrit,maxit,stagdetect,complex(xinit)); 
en=toc(st);

fprintf('frequency=%fHz and relative residual=%e\n',f(i),relres);
%Save the the iteration count length(r) and the solution time
tot=tot+length(r);
tots=tots+en;

clear Arow Acol Aval Mrow Mcol Mval xinit bg

end 

fprintf('Total time for iterative solution=%f s\n Total iteration count=%d\n',tots,tot);  


[Aval,Acol,Arow,Mval,Mcol,Mrow,bg]=createStiffnessMatrix(NKg,(ELg),(FDg),(FEg),int32(nx),int32(ny),int32(nz),double(f(i)),mg); 
Memoryreq=whos('Arow').bytes+whos('Acol').bytes+whos('Aval').bytes+whos('Mrow').bytes+whos('Mcol').bytes+whos('Mval').bytes;
fprintf('Memory consumption for A and M matrices=%f MB\n',Memoryreq/1024/1024);



