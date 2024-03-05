
// Local Histogram Creation
kernel void hist_local_simple(global const uchar* A, global uint* H, local uint* LH, int num_bins, int is_colour) {
	int id = get_global_id(0); 
	int lid = get_local_id(0);
	int size_of_image = get_global_size(0);
	if (is_colour ==1){
		size_of_image = size_of_image/3;
	}
	int bin_index = (A[id] * num_bins)/256;
	int Y_Cb_Cr = id / size_of_image; // 0 - Y, 1 - Cb, 2 - Cr
	//clear the scratch bins

	LH[lid] = 0;
	
	// Checks if reading either intensity value of colour image, or greyscale image
	// Doesn't run for Cb or Cr of colour image
	if (((is_colour == 1) && (Y_Cb_Cr == 0)) || !(is_colour == 1)){
		barrier(CLK_LOCAL_MEM_FENCE);
		atomic_inc(&LH[bin_index]);
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid < num_bins){ //combine all local hist into a global one
			atomic_add(&H[lid], LH[lid]);
		}
	}
}

// Global Histogram Creation
kernel void hist_global(global const uchar* A, global uint* H, int num_bins, int is_colour) {
	int id = get_global_id(0);
	int size_of_image = get_global_size(0);
	if (is_colour ==1){
		size_of_image = size_of_image/3;
	}
	int bin_index = (A[id] * num_bins)/256;
	int Y_Cb_Cr = id / size_of_image; // 0 - Y, 1 - Cb, 2 - Cr
	
	// Checks if reading either intensity value of colour image, or greyscale image
	// Doesn't run for Cb or Cr of colour image
	if (((is_colour == 1) && (Y_Cb_Cr == 0)) || !(is_colour == 1)){
		atomic_inc(&H[bin_index]);
	}
}

//Hillis-Steele basic inclusive scan
//requires additional buffer B to avoid data overwrite 
kernel void scan_hs(global uint* A, global uint* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;

	for (int stride = 1; stride <= N; stride *= 2) {
		B[id] = A[id];
		if (id >= stride)
			B[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

		C = A; A = B; B = C; //swap A & B between steps
	}
}
//Blelloch basic exclusive scan
kernel void scan_bl(global uint* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;

	//up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride*2)) == 0)
			A[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	//down-sweep
	if (id == 0)
		A[N-1] = 0;//exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	for (int stride = N/2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride*2)) == 0) {
			t = A[id];
			A[id] += A[id - stride]; //reduce 
			A[id - stride] = t;		 //move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
}

//Normalize histogram
kernel void normalize_histo(global uint* A, global uint* B, float scaler){
	int id = get_global_id(0);
	B[id] = A[id] * scaler;
}

// Backpropogation
kernel void equalise_image(global uchar* A, global uchar* B, global uint* NH, int num_bins, int is_colour){
	int id = get_global_id(0);
	int size_of_image = get_global_size(0);
	if (is_colour == 1){
		size_of_image = size_of_image/3;
	}
	int bin_index = (A[id] * num_bins)/256;
	int Y_Cb_Cr = id / size_of_image; // 0 - Y, 1 - Cb, 2 - Cr
	
	// Changes intensity of intensity values of colour image, or intensity of greyscale image
	if (((is_colour == 1) && (Y_Cb_Cr == 0)) || !(is_colour == 1)){
		B[id] = NH[bin_index];
	}
	// Just copies colour image Cb and Cr
	else{
		B[id] = A[id];
	}
}


