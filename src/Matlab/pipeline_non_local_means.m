%% SCRIPT: PIPELINE_NON_LOCAL_MEANS
%
% Pipeline for non local means algorithm as described in [1].
%
% The code thus far is implemented in CPU.
%
% DEPENDENCIES
%
% [1] Antoni Buades, Bartomeu Coll, and J-M Morel. A non-local
%     algorithm for image denoising. In 2005 IEEE Computer Society
%     Conference on Computer Vision and Pattern Recognition (CVPR’05),
%      volume 2, pages 60–65. IEEE, 2005.
%
  
  clear all %#ok
  close all

  %% PARAMETERS
  
  % input image
  
  image_temp = double(imread('../images/me.jpg'));
  image_temp = image_temp./256;

  imageLength = 64;
  for i=1:imageLength
      for j=1:imageLength
          image(i,j) = image_temp(i,j);
      end
  end

  save ../images/image.mat image

  clear all
  
  pathImg   = '../data/house.mat';
  strImgVar = 'house';
  
  % noise
  noiseParams = {'gaussian', ...
                 0,...
                 0.001};
  
  % filter sigma value
  n_patch_filter = 3;
  m_patch_filter = 3;
  filtSigma = 0.02;
  patchSize = [n_patch_filter m_patch_filter];
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
  
  N_block = 64;
  M_block = 64;
  threads_in_block = [16 16];
  
  %% GRID DATA
  
  N_grid = 1;
  M_grid = 1;
  grid_size = [N_grid M_grid];
  
  %% CREATE KERNEL OBJECT
  
  KERNEL = parallel.gpu.CUDAKernel( '../cuda/Babis_Kernel_Jan_2017.ptx', '../cuda/Babis_Kernel_Jan_2017.cu');
  KERNEL.ThreadBlockSize = threads_in_block;
  KERNEL.GridSize = grid_size;
  
  
  %% MAIN DATA FOR CUDA
  
  threads = threads_in_block(1);
  
  % Patch for gaussian noise
  
  G = fspecial('gaussian',patchSize, patchSigma);
  G = G(:) ./ max(G(:));
  
  % Convert arrays to single precision
  
  J_single = single(J);
  Zero1 = zeros(N_block,M_block);
  Zero_single1 = single(Zero1);
  Zero2 = zeros(N_block,M_block);
  Zero_single2 = single(Zero2);
  Zero3 = zeros(N_block,M_block);
  Zero_single3 = single(Zero3);
  G_single = single(G);
  
  % Send the above data from CPU to GPU
  
  tic
  
  J_single = gpuArray(J_single);
  Zero_single1 = gpuArray(Zero_single1);
  G_single = gpuArray(G_single);
  Zero_single2 = gpuArray(Zero_single2);
  Zero_single3 = gpuArray(Zero_single3);
  
  
  toc
  
  % Get the results from the GPU back to CPU

  tic

  temp_final = feval(KERNEL, J_single, Zero_single1, G_single, Zero_single2, Zero_single3);

  wait(gpuDevice);

  toc

  tic
  
  FINAL = gather( temp_final );
  
  toc
  
  
  
  %% VISUALIZE RESULT
  
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

%%------------------------------------------------------------
%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%
% VERSION
%
%   0.1 - December 28, 2016
%
% CHANGELOG
%
%   0.1 (Dec 28, 2016) - Dimitris
%       * initial implementation
%
% ------------------------------------------------------------
