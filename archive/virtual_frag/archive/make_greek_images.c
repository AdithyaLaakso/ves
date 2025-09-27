#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "bmp.h"
#define origin "../hand_writing_dataset/LETT_"

typedef enum {
	ALPHA,
	BETA,
	DELTA,
	EPSILON,
	FI,
	GAMMA,
	HETA,
	IOTA,
	KAPA,
	KSI,
	LAMDA,
	MI,
	NI,
	OMEGA,
	OMIKRON,
	PII,
	PSI,
	RO,
	SIGMA,
	TAU,
	THETA,
	XI,
	YPSILON,
	ZETA,
	ANY_LETTER,
	NONE
} GreekLetter;

typedef enum {
	UPPER,
	LOWER,
	ANY_CAPS
} Caps;

typedef enum {
	END,
		NORM,
		ANY_POS
} Pos;

typedef struct {
	GreekLetter letter;
	Caps caps;
	Pos pos;
} LetterData;

typedef struct {
	uint8_t* data;
	int height, width;
} BMPImage;

int rand_in_range(int minimum_number, int max_number) {
	if (max_number < minimum_number) {
		fprintf(stderr, "Invalid range: max_number (%d) < minimum_number (%d)\n", max_number, minimum_number);
		exit(EXIT_FAILURE);
	}
	if (max_number == minimum_number) {
		return max_number;
	}
	return rand() % (max_number + 1 - minimum_number) + minimum_number;
}

const char* greekLetterToString(GreekLetter letter) {
	switch (letter) {
		case ALPHA:    return "ALPHA";
		case BETA:     return "BETA";
		case DELTA:    return "DELTA";
		case EPSILON:  return "EPSILON";
		case FI:       return "FI";
		case GAMMA:    return "GAMMA";
		case HETA:     return "HETA";
		case IOTA:     return "IOTA";
		case KAPA:     return "KAPA";
		case KSI:      return "KSI";
		case LAMDA:    return "LAMDA";
		case MI:       return "MI";
		case NI:       return "NI";
		case OMEGA:    return "OMEGA";
		case OMIKRON:  return "OMIKRON";
		case PII:      return "PII";
		case PSI:      return "PSI";
		case RO:       return "RO";
		case SIGMA:    return "SIGMA";
		case TAU:      return "TAU";
		case THETA:    return "THETA";
		case XI:       return "XI";
		case YPSILON:  return "YPSILON";
		case ZETA:     return "ZETA";
		case ANY_LETTER:
		case NONE:
		default:       return NULL;
	}
}

const char* get_path_for_letter(LetterData* letter) {
	const char* sep1 = "_";
	const char* sep2 = ".";
	const char* sep3 = "/";

	static char path[256];

	if (letter->letter == NONE ||
			letter->letter == ANY_LETTER ||
			letter->pos == ANY_POS ||
			letter->caps == ANY_CAPS)
		return NULL;

	const char* letter_name = greekLetterToString(letter->letter);
	if (letter_name == NULL) return NULL;

	const char* caps_name = (letter->caps == UPPER) ? "CAP" : "SML";
	const char* pos_name = (letter->pos == END) ? "SUFF" : "NORM";

	strcat(path, origin);
	strcat(path, caps_name);
	strcat(path, sep1);
	strcat(path, pos_name);
	strcat(path, sep2);
	strcat(path, letter_name);
	strcat(path, sep3);

	printf("\nAttempting to read path: %s\n", path);

	return path;
}

void resolve_anys(LetterData* letter) {
	printf("resolve");
	if (letter->letter == ANY_LETTER) letter->letter = rand_in_range(0,23);
	if (letter->caps == ANY_CAPS) letter->caps = rand_in_range(0,1);
	if (letter->pos == ANY_POS) letter->pos = rand_in_range(0,1);
}

LetterData* get_random_letter_data() {
	LetterData* letter = malloc(sizeof(LetterData));

	letter->letter = ANY_LETTER;
	letter->caps = ANY_CAPS;
	letter->pos = ANY_POS;

	resolve_anys(letter);
	return letter;
}


LetterData* make_letter_data(GreekLetter letter, Caps caps, Pos pos) {
	LetterData* data = malloc(sizeof(LetterData));
	data->letter = letter;
	data->caps = caps;
	data->pos = pos;
	return data;
}

BMPImage* get_letter_image (LetterData* letter) {
	printf("getting image");
	//TODO:
	/*if (letter->letter == NONE)
		return gen_noise;*/
	if (letter->letter == ANY_LETTER ||
			letter->pos == ANY_POS
			|| letter->caps == ANY_CAPS)
		resolve_anys(letter);

	const char* path = get_path_for_letter(letter);
	if (!path) return NULL;

	BMPImage* image = malloc(sizeof(BMPImage));

	if (read_bmp_1bit(path,
				&(image->data),
				&(image->width),
				&(image->height)) != 0) {
		printf("File read failed for path %s", path);
		return NULL;
	}

	return image;
}

void free_image(BMPImage* image) {
	free(image->data);
	free(image);
}

int main() {
	srand(time(NULL));  // Add in main()
	printf("main");
	const char* dest = "./image.bmp";

	LetterData* letter = get_random_letter_data();

	BMPImage* image = get_letter_image(letter);
	printf("writing image");
	write_bmp_1bit(dest, image->data, image->width, image->height);

	printf("freeing");
	free(letter);
	free_image(image);
}
