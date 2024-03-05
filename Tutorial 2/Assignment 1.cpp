#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;


void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	//std::cerr << "  -f : input image file (default: test.pgm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		//else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	string user_input;
	bool do_loop = true;
	int int_user_input;



	// User Selects what Image to run code on
	// While loop + try/catch for error handling
	string image_filename;
	bool is_colour_image;
	while (do_loop) {
		cout << "What image would you like to run code on?" << endl << "[1] Greyscale" << endl << "[2] Colour" << endl << "[3] Greyscale (Large)" << endl << "[4] Colour (Large)" << endl << "[5] Custom PPM File" << endl;
		cin >> user_input;
		try {
			int_user_input = stoi(user_input);
			switch (int_user_input) {
			case 1:
				image_filename = "test.pgm";
				is_colour_image = false;

				do_loop = false;
				break;
			case 2:
				image_filename = "test.ppm";
				is_colour_image = true;

				do_loop = false;
				break;
			case 3:
				image_filename = "test_large.pgm";
				is_colour_image = false;

				do_loop = false;
				break;
			case 4:
				image_filename = "test_large.ppm";
				is_colour_image = true;

				do_loop = false;
				break;
			case 5:
				cout << "Input file location of image" << endl;
				cin >> user_input;
				image_filename = user_input;
				is_colour_image = true;

				do_loop = false;
				break;
			default:
				cout << "Invalid Integer" << endl;
			}
		}
		catch (...) {
			cout << "Please input a valid integer" << endl;
		}
	}

	// Cant pass bool to kernels so create int of same value to check
	int int_is_colour_image = (is_colour_image) ? 1 : 0;

	// User Selects a number of bins between 1 and 255
	// While loop + try/catch for error handling
	int number_of_bins;
	do_loop = true;
	while (do_loop) {
		cout << "How many bins would you like to run code for?" << endl << "[1] 2" << endl << "[2] 4" << endl << "[3] 8" << endl << "[4] 16" << endl << "[5] 32" << endl << "[6] 64" << endl << "[7] 128" << endl << "[8] 256" << endl;
		cin >> user_input;
		try {
			int_user_input = stoi(user_input);
			if ((int_user_input >= 1) && (int_user_input <= 9)) {
				number_of_bins = pow(2, int_user_input);
				do_loop = false;
			}
			else {
				cout << "Invalid Integer" << endl;
			}
		}
		catch (...) {
			cout << "Please input a valid integer" << endl;
		}
	}


	// User Selects whether to create histogram using local or global memory
	// While loop + try/catch for error handling
	bool local_hist;
	do_loop = true;
	while (do_loop) {
		cout << "Would you like to create histogram using Global or Local memory?" << endl << "[1] Global" << endl << "[2] Local" << endl;
		cin >> user_input;
		try {
			int_user_input = stoi(user_input);
			switch (int_user_input) {
			case 1:
				local_hist = false;
				do_loop = false;
				break;
			case 2:
				local_hist = true;
				do_loop = false;
				break;
			default:
				cout << "Invalid Integer" << endl;
			}
		}
		catch (...) {
			cout << "Please input a valid integer" << endl;
		}
	}


	// User Selects whether to use Blelloch or Hillis-Steele cumulation
	// While loop + try/catch for error handling
	string cumulation;
	do_loop = true;
	while (do_loop) {
		cout << "Would you like to use Blelloch or Hillis-Steele cumulation?" << endl << "[1] Blelloch" << endl << "[2] Hillis-Steele" << endl;
		cin >> user_input;
		try {
			int_user_input = stoi(user_input);
			switch (int_user_input) {
			case 1:
				cumulation = "bl";
				do_loop = false;
				break;
			case 2:
				cumulation = "hs";
				do_loop = false;
				break;
			default:
				cout << "Invalid Integer" << endl;
			}
		}
		catch (...) {
			cout << "Please input a valid integer" << endl;
		}
	}


	vector<unsigned int> histogram(number_of_bins, 0);
	size_t total_memory_of_histogram = number_of_bins * sizeof(unsigned int);

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		// Setup image to be displayed and display input image
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input,"input");
		// Get image size
		float image_size = image_input.size();
		//if it's a colour image, change to YCbCr
		if (is_colour_image) image_input.RGBtoYCbCr();
		
		//host operations
		//Select computing devices
		cl::Context context = GetContext(platform_id, device_id);
		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
		//create a queue to which we will push commands for the device
		// Enable queue profiling to get kernel execution times
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//device operations


		//Create buffers
		//Image buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); 
		//Histogram buffers
		cl::Buffer dev_histogram(context, CL_MEM_READ_WRITE, total_memory_of_histogram);
		cl::Buffer dev_c_histogram(context, CL_MEM_READ_WRITE, total_memory_of_histogram);
		cl::Buffer dev_n_histogram(context, CL_MEM_READ_WRITE, total_memory_of_histogram);
		
		//copy image to dev_image_input
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);



		//Setup and execute histogram kernel 
		cl::Kernel histogram_kernel;

		//Create event to get execution time
		cl::Event histogram_timer;
		// If the histogram is local, use local kernel and set args for it
		if (local_hist) {

			histogram_kernel = cl::Kernel(program, "hist_local_simple");
			histogram_kernel.setArg(0, dev_image_input); // Input Image
			histogram_kernel.setArg(1, dev_histogram); // Histogram
			histogram_kernel.setArg(2, cl::Local(total_memory_of_histogram)); // Memory for local histogram
			histogram_kernel.setArg(3, number_of_bins); // Number of bins
			histogram_kernel.setArg(4, int_is_colour_image); // If image is colour or not
		}
		// If the histogram is global, use global kernel and set args for it
		else{
			histogram_kernel = cl::Kernel(program, "hist_global");
			histogram_kernel.setArg(0, dev_image_input); // Input Image
			histogram_kernel.setArg(1, dev_histogram); // Histogram
			histogram_kernel.setArg(2, number_of_bins); // Number of bins
			histogram_kernel.setArg(3, int_is_colour_image); // If image is colour or not
		}
		// Execute whichever histogram kernel has been set up
		queue.enqueueNDRangeKernel(histogram_kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NDRange(number_of_bins), NULL, &histogram_timer);
		queue.enqueueReadBuffer(dev_histogram, CL_TRUE, 0, total_memory_of_histogram, &histogram[0], NULL);

		// Output kernel execution time and add to running total
		int histogram_time_ns = histogram_timer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - histogram_timer.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		int total_timer = histogram_time_ns;
		std::cout << "Histogram kernel execution time [ns]:" << histogram_time_ns << std::endl;

		// Output simple histogram
		std::cout << "Simple Histogram:" << endl;
		std::cout << histogram << endl;



		//Setup and execute cumulative histogram kernel 
		cl::Kernel cumulative_kernel;

		//Create event to get execution time
		cl::Event c_histogram_timer;
		// If using Hillis-Steele cumulation, use that kernel and set args for it
		if (cumulation == "hs") {
			cumulative_kernel = cl::Kernel(program, "scan_hs");
			cumulative_kernel.setArg(0, dev_histogram); // Histogram
			cumulative_kernel.setArg(1, dev_c_histogram); // Cumulative Histogram
		}
		// If using Blelloch cumulation, use that kernel and set args for it
		else if (cumulation == "bl") {
			cumulative_kernel = cl::Kernel(program, "scan_bl");
			cumulative_kernel.setArg(0, dev_histogram); // Histogram
			dev_c_histogram = dev_histogram; // Set c histogram to histogram because Blelloch only takes one arguement
		}
		// Execute whichever histogram scanning kernel has been set up
		queue.enqueueNDRangeKernel(cumulative_kernel, cl::NullRange, cl::NDRange(number_of_bins), cl::NDRange(number_of_bins), NULL, &c_histogram_timer);
		queue.enqueueReadBuffer(dev_c_histogram, CL_TRUE, 0, total_memory_of_histogram, &histogram[0], NULL);

		// Output kernel execution time and add to running total
		int c_histogram_time_ns = c_histogram_timer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - c_histogram_timer.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		total_timer += c_histogram_time_ns;
		std::cout << "Cummulative Histogram kernel execution time [ns]:" << c_histogram_time_ns << std::endl;

		// Output cumulitive histogram
		std::cout << "Cumulative Histogram:" << endl;
		std::cout << histogram << endl;
		
		
		//Setup and execute normalisation kernel 
		// Get scalar to normalise cumulative histogram with
		float scale = 255.0f / image_size;
		// If it's a colour image, scaler is 3 times beigger because image size is 3 times bigger
		if (is_colour_image) {
			scale = scale * 3;
		}
		//Create event to get execution time
		cl::Event n_histogram_timer;
		// Get kernel and set args
		cl::Kernel normalize_kernel = cl::Kernel(program, "normalize_histo");
		normalize_kernel.setArg(0, dev_c_histogram); // Cumulative Histogram
		normalize_kernel.setArg(1, dev_n_histogram); // Normalised Histogram 
		normalize_kernel.setArg(2, scale); // What to scale

		// Execute histogram normalisation kernel has been set up
		queue.enqueueNDRangeKernel(normalize_kernel, cl::NullRange, cl::NDRange(number_of_bins), cl::NDRange(number_of_bins), NULL, &n_histogram_timer);
		queue.enqueueReadBuffer(dev_n_histogram, CL_TRUE, 0, total_memory_of_histogram, &histogram[0], NULL);

		// Output kernel execution time and add to running total
		int n_histogram_time_ns = n_histogram_timer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - n_histogram_timer.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		total_timer += n_histogram_time_ns;
		std::cout << "Normalize Histogram kernel execution time [ns]:" << n_histogram_time_ns << std::endl;

		// Output normalised histogram
		std::cout << "Normalised Histogram:" << endl;
		std::cout << histogram << endl;



		//Setup and execute image equalisation kernel 
		//Create event to get execution time
		cl::Event e_image_timer;
		// Get kernel and set args
		cl::Kernel equalise_kernel = cl::Kernel(program, "equalise_image");
		equalise_kernel.setArg(0, dev_image_input); // Input image 
		equalise_kernel.setArg(1, dev_image_output); // Output image 
		equalise_kernel.setArg(2, dev_n_histogram); // Normalised histogram
		equalise_kernel.setArg(3, number_of_bins); // Number of bins
		equalise_kernel.setArg(4, int_is_colour_image); // If image is colour or not

		//Execute image equalisation kernel
		queue.enqueueNDRangeKernel(equalise_kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NDRange(number_of_bins), NULL, &e_image_timer);
		vector<unsigned char> output_buffer(image_input.size());
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0], NULL);

		// Output kernel execution time and add to running total
		int e_image_time_ns = e_image_timer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - e_image_timer.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		total_timer += e_image_time_ns;
		std::cout << "Equalise Image kernel execution time [ns]:" << e_image_time_ns << std::endl;

		// Output total run time
		std::cout << "Total Program execution time [ns]:" << total_timer << std::endl;

		// Get data for output image
		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		// If it's a colour image, convert back from YCbCr to RGB
		if (is_colour_image == 1) output_image.YCbCrtoRGB();
		// Output image
		CImgDisplay disp_output(output_image, "output");

		// Keep input and output images displayed
		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}
	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
