#include <math.h>
#include <stdio.h>
#include <stdlib.h>


__global__ void Babis_Kernel(double const *A, double  *B, double const *G, int n, int m, int patchSize_x, int patchSize_y, double filtSigma)
{
  int x,y,area_x,area_y;
  double norm,w_temp,diff=0,W=0,Products=0;
 

  // Set pixel coordinates

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  
  // For every pixel we check every other pixel neighbor, if we are inside the table limits

  if( ( i>=patchSize_x ) && ( j>=patchSize_y ) && ( i<=(n-patchSize_x) -1 ) && ( j<=(m-patchSize_x) -1 ) )
  { 
    for(x=patchSize_x; x< (n-patchSize_x) ; x++)
    {
	   for(y=patchSize_y; y< (m-patchSize_y) ; y++)
        {
		   norm=0; 
		   w_temp=0;
		   
		// i,j indicate the coordinates of the current thread, while x,y indicate the coordinates of every other pixel-neighbor in the table/image and x_area,y_area indicate 
		// the area around the neighbor, for example 3x3, 5x5, 7x7.

		   for(area_x=-patchSize_x; area_x<=patchSize_x; area_x++)
             {
		      for(area_y=-patchSize_y; area_y<=patchSize_y; area_y++)
                {

				  diff=abs(A[(i+area_x)*m+(j+area_y)]-A[(x+area_x)*m+(y+area_y)]);

				  diff=diff*G[(area_x+patchSize_x)*((patchSize_y*2)+1)+(area_y+patchSize_y)];

				  diff=diff*diff;

				  norm+=diff;  
			 }
		    }

		    w_temp=exp(-norm/filtSigma);
		    W+=w_temp;
		    Products+=w_temp*A[x*m+y];
	    }
	}
  
     B[(i-patchSize_x)*(m-(2*patchSize_y)) + (j-patchSize_y)] = Products/W;
  }
}

