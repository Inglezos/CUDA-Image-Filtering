%% SCRIPT: PIPELINE_NON_LOCAL_MEANS
%
% Pipeline for non local means algorithm as described in [1].
%
% The code thus far is implemented in CPU.
%
% DEPENDENCIES
%
%     A non-local algorithm for image denoising.
%
  
  clear all %#ok
  close all

  %% PARAMETERS
  
  
  % input image

  image_temp = double(imread('../images/black_hole_64x64.jpg'));
  image_temp = image_temp./256;

  imageLength = 64;
  for i=1:imageLength
      for j=1:imageLength
          image(i,j) = image_temp(i,j);
      end
  end

  save ../images/black_hole_64x64.mat image

  clear all
  
  pathImg   = '../images/black_hole_64x64.mat';
  strImgVar = 'image';


  % noise

  noiseParams = {'gaussian', ...
                 0,...
                 0.001};
  
  % filter sigma value
  filtSigma = 0.02;
  patchSize = [3 3];
  patchSize_x=floor(patchSize(1)./2);
  patchSize_y=floor(patchSize(2)./2);
  patchSigma = 5/3;
  
  %% USEFUL FUNCTIONS

  % image normalizer
  normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));
  
  %% (BEGIN)

  fprintf('...begin %s...\n',mfilename);  
  
  %% INPUT DATA
  
  fprintf('...loading input data...\n')
  
  ioImg = matfile( pathImg );
  I     = ioImg.(strImgVar);
  [N M] = size(I);

  
  %% PREPROCESS
  
  fprintf(' - normalizing image...\n')
  I = normImg( I );
  
  figure('Name','Original Image');
  imagesc(I); axis image;
  colormap gray;
  saveas(gcf,'Original image.jpg');
  
  %% NOISE
  
  fprintf(' - applying noise...\n')
  J = imnoise( I, noiseParams{:} );
  figure('Name','Noisy-Input Image');
  imagesc(J); axis image;
  colormap gray;
  saveas(gcf,'Original image with noise.jpg');
  
  %% NON LOCAL MEANS

  %% SET INITIAL PARAMETERS FOR GPU PARALLEL COMPUTING WITH CUDA
  
  %% BLOCK DATA
  
  threads_In_Block = [16 16];
  
  array_padding = padarray(J,floor(patchSize./2),'symmetric');
  [N_new, M_new] = size(array_padding);

  G = fspecial('gaussian',patchSize, patchSigma);
  G = G(:) ./ max(G(:)); 
 
  KERNEL = parallel.gpu.CUDAKernel( '../cuda/Babis_Kernel.ptx', '../cuda/Babis_Kernel.cu');

  N_block = ceil(N_new/threads_In_Block(1));
  M_block = ceil(M_new/threads_In_Block(2)); 
  Blocks = [N_block M_block];    
  KERNEL.ThreadBlockSize = threads_In_Block;
  KERNEL.GridSize = Blocks;
     
  % Convert arrays to single precision
  
  B = zeros(N,M);
  
  % Send the above data from CPU to GPU
  
  tic
  
  A_GPU = gpuArray(array_padding);
  B_GPU = gpuArray(B);
  G_GPU = gpuArray(G);  
  
  toc
  

  % Get the results from the GPU back to CPU

  tic

  temp_final = feval(KERNEL, A_GPU, B_GPU,G_GPU, N_new, M_new, patchSize_x, patchSize_y, filtSigma);

  wait(gpuDevice);

  toc

  tic
  
  FINAL = gather( temp_final );
  
  toc

  fprintf('Error: %e\n', norm( B - (J+1), 'fro' ) );
  
  figure('Name', 'Filtered image');
  imagesc(FINAL); axis image;
  colormap gray;
  saveas(gcf,'Filtered image.jpg');
  
  figure('Name', 'Residual');
  imagesc(FINAL-J); axis image;
  colormap gray;
  saveas(gcf,'Residual.jpg');
  
  %% (END)

  fprintf('...end %s...\n',mfilename);

