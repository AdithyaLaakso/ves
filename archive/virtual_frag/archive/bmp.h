#ifndef BMP
#define BMP

#include <stdint.h>

int write_bmp_1bit(const char* filename, const uint8_t* data, int width, int height);

int read_bmp_1bit(const char* filename, uint8_t** data, int* width, int* height);

#endif /* BMP_1BIT_H */
