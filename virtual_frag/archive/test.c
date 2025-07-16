#include "stdlib.h"
#include "stdio.h"

#include "bmp.h"

// Example usage
int test() {
    // Create a simple test pattern (checkerboard)
    int width = 37, height = 49;
    uint8_t* test_data = malloc(width * height);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Create checkerboard pattern
            test_data[y * width + x] = ((x + y) % 2);
        }
    }
    
    // Write 1-bit BMP file
    if (write_bmp_1bit("test_1bit.bmp", test_data, width, height) == 0) {
        printf("Successfully wrote test_1bit.bmp\n");
    }
    
    // Read it back
    uint8_t* read_data;
    int read_width, read_height;
    if (read_bmp_1bit("test_1bit.bmp", &read_data, &read_width, &read_height) == 0) {
        printf("Successfully read 1-bit BMP: %dx%d\n", read_width, read_height);
        
        // Print a small portion to verify
        printf("First 10x10 pixels:\n");
        for (int y = 0; y < 10 && y < read_height; y++) {
            for (int x = 0; x < 10 && x < read_width; x++) {
                printf("%d", read_data[y * read_width + x]);
            }
            printf("\n");
        }
        
        free(read_data);
    }
    
    free(test_data);
    return 0;
}

int main() {
	const char* origin = "../hand_writing_dataset/LETT_CAP_NORM.FI/form_1.bmp";
	const char* dest = "./form_1.bmp";

	uint8_t* data;
	int width, height;

	if (read_bmp_1bit(origin, &data, &width, &height) == 0) {
		printf("Successfully read file\n");
	}

	write_bmp_1bit(dest, data, width, height);
	free(data);
}
