#todo make it actually get a sample
#maybe also make it options just get random words rather than the passage

def get_passage():
    try:
        with open("sample_greek_text/THEOGONY", "r") as file:
            # Read the entire content of the file
            content = file.read()
            return content
    except FileNotFoundError:
        print("Error: The file 'my_file.txt' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
