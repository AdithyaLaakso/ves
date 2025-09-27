# File paths
input_file = "./pathstmp.json"
output_file = "../single_letter_model/paths.json"

# Read from the input file
with open(input_file, "r") as f:
    input_lines = f.readlines()

output_lines = []

for line in input_lines:
    line = line.strip()
    if not line:
        continue
    # Extract the label (after the last underscore and before .bmp)
    base_name = line.split(".")[0]
    label = base_name.split("_")[-1]
    output_lines.append(f'["training_data/{line}", "{label}"],')

# Write to the output file
with open(output_file, "w") as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"Converted lines written to {output_file}")
