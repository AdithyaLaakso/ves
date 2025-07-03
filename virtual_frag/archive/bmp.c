#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "bmp.h"

// BMP file header structures
#pragma pack(push, 1)
typedef struct {
    uint16_t type;        // "BM"
    uint32_t size;        // File size
    uint16_t reserved1;   // Reserved
    uint16_t reserved2;   // Reserved
    uint32_t offset;      // Offset to pixel data
} BMPFileHeader;

typedef struct {
    uint32_t size;        // Header size
    int32_t width;        // Image width
    int32_t height;       // Image height
    uint16_t planes;      // Color planes
    uint16_t bits;        // Bits per pixel
    uint32_t compression; // Compression type
    uint32_t imagesize;   // Image size
    int32_t xresolution;  // X resolution
    int32_t yresolution;  // Y resolution
    uint32_t ncolors;     // Number of colors
    uint32_t importantcolors; // Important colors
} BMPInfoHeader;

typedef struct {
    uint8_t blue;
    uint8_t green;
    uint8_t red;
    uint8_t reserved;
} BMPColorTableEntry;
#pragma pack(pop)

/**
 * Write a 1-bit BMP file from a binary array
 * 
 * @param filename: Output filename
 * @param data: Binary pixel data array (0 = black, 1 = white)
 * @param width: Image width in pixels
 * @param height: Image height in pixels
 * @return: 0 on success, -1 on error
 */
int write_bmp_1bit(const char* filename, const uint8_t* data, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Cannot create file %s\n", filename);
        return -1;
    }
    
    // Calculate bytes per row (8 pixels per byte, padded to 4-byte boundary)
    int bytes_per_row = (width + 7) / 8;  // Round up to nearest byte
    int row_padded = (bytes_per_row + 3) & (~3);  // Pad to 4-byte boundary
    int image_size = row_padded * height;
    
    // File header
    BMPFileHeader file_header = {0};
    file_header.type = 0x4D42;  // "BM"
    file_header.size = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + 8 + image_size; // +8 for color table
    file_header.offset = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + 8;
    
    // Info header
    BMPInfoHeader info_header = {0};
    info_header.size = sizeof(BMPInfoHeader);
    info_header.width = width;
    info_header.height = height;
    info_header.planes = 1;
    info_header.bits = 1;
    info_header.compression = 0;
    info_header.imagesize = image_size;
    info_header.ncolors = 2;
    info_header.importantcolors = 2;
    
    // Color table (black and white)
    BMPColorTableEntry color_table[2];
    // Color 0: Black
    color_table[0].blue = 0;
    color_table[0].green = 0;
    color_table[0].red = 0;
    color_table[0].reserved = 0;
    // Color 1: White
    color_table[1].blue = 255;
    color_table[1].green = 255;
    color_table[1].red = 255;
    color_table[1].reserved = 0;
    
    // Write headers and color table
    fwrite(&file_header, sizeof(BMPFileHeader), 1, file);
    fwrite(&info_header, sizeof(BMPInfoHeader), 1, file);
    fwrite(color_table, sizeof(BMPColorTableEntry), 2, file);
    
    // Write pixel data (BMP stores pixels bottom-to-top)
    uint8_t* row_buffer = calloc(row_padded, 1);
    if (!row_buffer) {
        fclose(file);
        return -1;
    }
    
    for (int y = height - 1; y >= 0; y--) {
        // Clear row buffer
        for (int i = 0; i < row_padded; i++) {
            row_buffer[i] = 0;
        }
        
        // Pack 8 pixels into each byte
        for (int x = 0; x < width; x++) {
            int pixel_value = data[y * width + x];
            int byte_idx = x / 8;
            int bit_idx = 7 - (x % 8);  // MSB first
            
            if (pixel_value) {
                row_buffer[byte_idx] |= (1 << bit_idx);
            }
        }
        
        fwrite(row_buffer, row_padded, 1, file);
    }
    
    free(row_buffer);
    fclose(file);
    return 0;
}

/**
 * Read a 1-bit BMP file into a binary array
 * 
 * @param filename: Input filename
 * @param data: Pointer to store allocated binary array (caller must free)
 * @param width: Pointer to store image width
 * @param height: Pointer to store image height
 * @return: 0 on success, -1 on error
 */
int read_bmp_1bit(const char* filename, uint8_t** data, int* width, int* height) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return -1;
    }
    
    BMPFileHeader file_header;
    BMPInfoHeader info_header;
    
    // Read headers
    if (fread(&file_header, sizeof(BMPFileHeader), 1, file) != 1 ||
        fread(&info_header, sizeof(BMPInfoHeader), 1, file) != 1) {
        printf("Error: Failed to read BMP headers\n");
        fclose(file);
        return -1;
    }
    
    // Validate BMP signature
    if (file_header.type != 0x4D42) {
        printf("Error: Not a valid BMP file\n");
        fclose(file);
        return -1;
    }
    
    // Only support 1-bit BMPs
    if (info_header.bits != 1) {
        printf("Error: Only 1-bit BMPs are supported (found %d-bit)\n", info_header.bits);
        fclose(file);
        return -1;
    }
    
    *width = info_header.width;
    *height = abs(info_header.height);
    
    // Allocate memory for binary data (one byte per pixel for simplicity)
    *data = malloc(*width * *height);
    if (!*data) {
        printf("Error: Memory allocation failed\n");
        fclose(file);
        return -1;
    }
    
    // Skip color table (8 bytes for 2 colors)
    fseek(file, file_header.offset, SEEK_SET);
    
    // Calculate row padding
    int bytes_per_row = (*width + 7) / 8;
    int row_padded = (bytes_per_row + 3) & (~3);
    uint8_t* row_buffer = malloc(row_padded);
    if (!row_buffer) {
        free(*data);
        fclose(file);
        return -1;
    }
    
    // Read pixel data (unpack bits, handle bottom-to-top storage)
    for (int y = 0; y < *height; y++) {
        if (fread(row_buffer, row_padded, 1, file) != 1) {
            printf("Error: Failed to read pixel data\n");
            free(*data);
            free(row_buffer);
            fclose(file);
            return -1;
        }
        
        // Determine actual row in output (handle bottom-up storage)
        int output_row = (info_header.height > 0) ? (*height - 1 - y) : y;
        
        // Unpack bits from bytes
        for (int x = 0; x < *width; x++) {
            int byte_idx = x / 8;
            int bit_idx = 7 - (x % 8);  // MSB first
            int pixel_value = (row_buffer[byte_idx] >> bit_idx) & 1;
            
            (*data)[output_row * *width + x] = pixel_value;
        }
    }
    
    free(row_buffer);
    fclose(file);
    return 0;
}
