#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tiffio.h>
#include "perlin.h"

// Simple RGB TIFF creation
int make_tiff(const char* filename, int width, int height, int channels, size_t channel_depth, unsigned char* data) {
	size_t bytes_per_line = (size_t)(width * channels) * channel_depth; 


	TIFF* tif = TIFFOpen(filename, "w");
	if (!tif) {
		fprintf(stderr, "Could not open %s for writing\n", filename);
		return -1;
	}

	// Set required tags
	TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
	TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
	TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, channels);          // RGB = 3 samples
	TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, channel_depth);            // 8 bits per channel
	TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
	TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
	TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE); // No compression

	// Optional: Set resolution
	TIFFSetField(tif, TIFFTAG_XRESOLUTION, 72.0);
	TIFFSetField(tif, TIFFTAG_YRESOLUTION, 72.0);
	TIFFSetField(tif, TIFFTAG_RESOLUTIONUNIT, RESUNIT_INCH);


	for (int row = 0; row < height; row++) {
		if (TIFFWriteScanline(tif, data + (row * bytes_per_line), row, 0) < 0) {
			fprintf(stderr, "Could not write scanline %d\n", row);
			TIFFClose(tif);
			return -1;
		}
	}

	TIFFClose(tif);
	return 0;
}

// Example usage with procedurally generated data
int main() {
	int width = 256;
	int height = 256;

	float** noise_grid = generate_perlin_grid(width, height, .1f, 42); 

	// Allocate memory for RGB image data
	unsigned char* rgb_data = malloc(width * height * 3);
	if (!rgb_data) {
		fprintf(stderr, "Memory allocation failed\n");
		return -1;
	}

	// Generate a simple gradient pattern
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int idx = (y * width + x) * 3;
			char noise = (char)(noise_grid[x][y] * 256);
			rgb_data[idx + 0] = noise;
			rgb_data[idx + 1] = noise;
			rgb_data[idx + 2] = noise;
		}
	}

	// Create the TIFF file
	if (basic_tiff("gradient.tif", width, height, width*3, rgb_data) == 0) {
		printf("Successfully created gradient.tif\n");
	}

	free_perlin(noise_grid, width, height);
	free(rgb_data);
	return 0;
}
