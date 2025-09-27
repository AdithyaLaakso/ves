#include <stdlib.h>
#include <math.h>

#define PERMUTATION_SIZE 256

// Permutation array (doubled for overflow)
int p[PERMUTATION_SIZE * 2];

void init_permutation(int seed) {
    int i;
    srand((unsigned int) seed);
    for (i = 0; i < PERMUTATION_SIZE; i++) {
        p[i] = i;
    }

    for (i = 0; i < PERMUTATION_SIZE; i++) {
        int j = rand() % PERMUTATION_SIZE;
        int tmp = p[i];
        p[i] = p[j];
        p[j] = tmp;
    }

    for (i = 0; i < PERMUTATION_SIZE; i++) {
        p[PERMUTATION_SIZE + i] = p[i];
    }
}

// Gradient function
float grad(int hash, float x, float y) {
    switch(hash & 0x3) {
        case 0: return  x + y;
        case 1: return -x + y;
        case 2: return  x - y;
        case 3: return -x - y;
        default: return 0; // never happens
    }
}

// Fade function (smootherstep)
float fade(float t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

// Linear interpolation
float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// 2D Perlin Noise
float perlin2D(float x, float y) {
    int xi = (int)(x) & 255;
    int yi = (int)(y) & 255;

    float xf = x - (int)(x);
    float yf = y - (int)(y);

    float u = fade(xf);
    float v = fade(yf);

    int aa = p[p[xi] + yi];
    int ab = p[p[xi] + yi + 1];
    int ba = p[p[xi + 1] + yi];
    int bb = p[p[xi + 1] + yi + 1];

    float x1 = lerp(grad(aa, xf, yf), grad(ba, xf - 1, yf), u);
    float x2 = lerp(grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1), u);
    float result = lerp(x1, x2, v);

    return result;
}

// Generates a 2D grid of Perlin noise
float** generate_perlin_grid(int width, int height, float scale, int seed) {
    init_permutation(seed);

    float** grid = malloc((size_t)height * sizeof(float*));
    for (int y = 0; y < height; y++) {
        grid[y] = malloc((size_t)width * sizeof(float));
        for (int x = 0; x < width; x++) {
            float nx = (float)x * scale;
            float ny = (float)y * scale;
            grid[y][x] = perlin2D(nx, ny);  // value in [-1, 1]
        }
    }
    return grid;
}

void free_perlin(float** perlin, int width, int height) {
	for (int y = 0; y < height; y++) {
		free(perlin[y]);
	}
}
