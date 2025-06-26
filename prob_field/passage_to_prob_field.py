from greek_char_prob_field import greek_char_prob_field
from etc import remove_greek_accents

def char_to_prob_field(char):
    if char == " ":
        return None
    char = remove_greek_accents(char)
    try:
        field = greek_char_prob_field(start_char = char)
    except:
        return None
    field = field.add_noise(uniform_noise = 0.01, gaussian_noise = 0.01, swap_noise = 0.01, dropout_rate = 0.01)
    return field

def line_to_prob_field_list(line):
    prob_field_list = []
    for char in line:
        field = char_to_prob_field(char)
        if field:
            prob_field_list.append(field)
    return prob_field_list

def passage_to_prob_field_matrix(passage):
    prob_field_matrix = []
    passage_arr = passage.split('\n')

    for line in passage_arr:
        prob_field_matrix.append(line_to_prob_field_list(line))

    return prob_field_matrix

def passage_to_prob_field_list(passage):
    passage_arr = list(passage.replace("\n", ""))

    prob_field_matrix = []
    for char in passage_arr:
        field = char_to_prob_field(char)
        if field:
            prob_field_matrix.append(field)

    return prob_field_matrix
