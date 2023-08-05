import csv
import re
from os import listdir
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

for file in sorted(list(listdir("comments"))):
    if file.endswith("csv"):

        with open(f"comments/{file}", 'r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)

            # Skip the header row if it exists
            next(csv_reader)

            # Iterate through the rows and extract the text from the first column
            text = [row[0] for row in csv_reader]

        filename = file.split(".")[0]
        path = f"comments_txt/{filename}.txt"
        #data = open(f"comments/{file}").read().splitlines()
        #text = [line[0] for line in data]

        with open(path, "w") as file:
            file.write(str(text))

        print(f"Comments saved to {path} successfully.")
