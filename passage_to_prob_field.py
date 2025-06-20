from get_passage import get_passage
from greek_char_prob_field import greek_char_prob_field
from etc import remove_greek_accents

def line_to_prob_field_list(line):
    prob_field_list = []
    for char in line:
        if char == " ":
            continue
        char = remove_greek_accents(char)
        try:
            field = greek_char_prob_field(start_char = char)
        except:
            continue
        field = field.add_noise(uniform_noise = 0.01, gaussian_noise = 0.01, swap_noise = 0.01, dropout_rate = 0.01)
        prob_field_list.append(field)
    return prob_field_list

def passage_to_prob_field_matrix(passage):
    prob_field_matrix = []
    passage_arr = passage.split('\n')

    for line in passage_arr:
        prob_field_matrix.append(line_to_prob_field_list(line))

    return prob_field_matrix

def gen_feild():
    passage = get_passage()
    matrix = passage_to_prob_field_matrix(passage)
    for line in matrix:
        for char in line:
            print(char)

gen_feild()
