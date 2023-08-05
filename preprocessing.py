import re
import csv
import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from collections import Counter
from wordcloud import WordCloud
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import spacy
from scipy.spatial.distance import cosine
import gensim

emoji_pattern = re.compile(
    r"[\U0001F600-\U0001F64F]|"  # emoticons
    r"[\U0001F300-\U0001F5FF]|"  # symbols & pictographs
    r"[\U0001F680-\U0001F6FF]|"  # transport & map symbols
    r"[\U0001F1E0-\U0001F1FF]|"  # flags (iOS)
    r"[\U00002500-\U00002BEF]|"  # chinese char
    r"[\U00002702-\U000027B0]|"  # Dingbats
    r"[\U000024C2-\U0001F251]"
)


def get_part_of_speech(word):
    probable_part_of_speech = wordnet.synsets(word)

    pos_counts = Counter()

    pos_counts["n"] = len(
        [item for item in probable_part_of_speech if item.pos() == "n"])
    pos_counts["v"] = len(
        [item for item in probable_part_of_speech if item.pos() == "v"])
    pos_counts["a"] = len(
        [item for item in probable_part_of_speech if item.pos() == "a"])
    pos_counts["r"] = len(
        [item for item in probable_part_of_speech if item.pos() == "r"])

    most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
    return most_likely_part_of_speech


lemmatizer = WordNetLemmatizer()
"""
with open("training.1600000.processed.noemoticon.csv", 'r', newline='', encoding='ISO-8859-1') as csvfile:
    csv_reader = csv.reader(csvfile)

    # Iterate through the rows and extract the text from the first column
    training_text = [row[5] for row in csv_reader]


def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|@[^\s]+", "", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)

    # Tokenize and convert to lowercase
    tokens = word_tokenize(text.lower())

    # Remove stopwords and perform stemming
    tokens = [lemmatizer.lemmatize(word, get_part_of_speech(
        word)) for word in tokens]

    return " ".join(tokens)

training_text = [preprocess_text(
    tweet) for tweet in training_text]

with open("training_text_processed.txt", "w", encoding="utf-8") as file:
    file.write(str(training_text))
"""

with open(f"training_text_processed.txt", "r", encoding="utf-8") as file:
    training_text = file.read()
training_text = ast.literal_eval(training_text)
count_vectorizer = CountVectorizer()
count_vectorizer.fit(training_text)
training_counts = count_vectorizer.transform(training_text)
"""
classifier = MultinomialNB()
training_labels = [0] * 800000 + [4] * 800000
classifier.fit(training_counts, training_labels)

joblib.dump(classifier, 'sentiment140_trained_full.pkl')
"""
classifier = joblib.load('sentiment140_trained_full.pkl')

stop_words = stopwords.words('english')

new_stop_words = ("jake", "paul", "nicki", "minaj",
                  "england", "city", "jennie", "lisa", "jisoo", "prime", "logan", "jeenyi", "song", "jeffree", "taco bell", "trish", "trisha", "taco", "bell", "lol", "pewdiepie")
to_remove = ("below", "no", "nor", "not", "only", "too", "very", "can", "don't",
             "aren", "aren't", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't")

for i in new_stop_words:
    stop_words.append(i)
for i in to_remove:
    stop_words.remove(i)

most_common_words = {}

for file in sorted(list(listdir("comments_txt"))):
    if file.endswith("txt"):

        filename = file.split(".")[0]
        name = filename.split("comments_")[1]

        path = f"comments_preprocessed/{filename}_preprocessed_lemmatized_no_stop.txt"

        with open(f"comments_txt/{file}", "r", encoding="utf-8") as file:
            file_content = file.read()
        data_list = ast.literal_eval(file_content)
        data_list = [re.sub(r'[^\w\s,]', lambda x: f' {x.group()} ', comment)
                     for comment in data_list]
        tokenized_data = [word_tokenize(comment.lower().replace(",", "").replace("-", " ").replace(":", "").replace(".", "").replace("?", "").replace("!", "").replace(";", "").replace("@", "").replace("/", "").replace("'", ""))
                          for comment in data_list]
        tokenized_data_no_stop = [
            [token for token in comment if token not in stop_words] for comment in tokenized_data]
        # tokenized_data_stemmed = [
        #     [stemmer.stem(token) for token in comment] for comment in tokenized_data_no_stop]
        tokenized_data_lemmatized = [[lemmatizer.lemmatize(token, get_part_of_speech(
            token)) for token in comment] for comment in tokenized_data_no_stop]

        all_comments_embeddings = gensim.models.Word2Vec(
            tokenized_data_lemmatized, vector_size=96)

        comments_counts = count_vectorizer.transform(
            [" ".join(comment) for comment in tokenized_data_lemmatized])
        comments_predictions = classifier.predict(comments_counts)

        average_comments_prediction = sum(
            comments_predictions) / len(comments_predictions)

        average_comments_prediction_percentage_positive = average_comments_prediction * 25
        average_comments_prediction_percentage_negative = 100 - \
            average_comments_prediction_percentage_positive

        print(
            f"The trained Naive Bayes classifier gives an average score of {average_comments_prediction} to the comments of this video.\nThe scores can go from 0 to 4 with 0 being the most negative & 4 the most positive.")
        if average_comments_prediction > 2:
            print(f"Video {name} has a majority of positive comments")

        else:
            print(f"Video {name} has a majority of negative comments")

        probabilities = classifier.predict_proba(comments_counts)

        # Calculate the average probability for the negative class (class 0)
        average_probability_negative = sum(
            probabilities[:, 0]) / len(probabilities)
        # Calculate the average probability for the positive class (class 1)
        average_probability_positive = sum(
            probabilities[:, 1]) / len(probabilities)

        # Labels for the pie chart segments
        labels = ['Positive', 'Negative']

        # Average probabilities for each class
        average_probabilities = [
            average_comments_prediction_percentage_positive, average_comments_prediction_percentage_negative]

        # Colors for the pie chart segments
        colors = ['blue', 'red']

        # Create the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(average_probabilities, labels=labels,
                colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Average Sentiment Distribution')
        # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.axis('equal')
        plt.savefig(
            f'average_sentiment_pie_charts/average_sentiment_absolute_pie_chart_{name}.svg', format='svg')
        plt.savefig(
            f'average_sentiment_pie_charts/average_sentiment_absolute_pie_chart_{name}.png', format='png', dpi=600)

        concatenated_text = ' '.join(
            [' '.join(comment) for comment in tokenized_data_lemmatized])

        wordcloud = WordCloud(
            width=3840, height=2160, background_color='#f0f0f0', collocations=False).generate(concatenated_text)

        all_words = [
            word for comment in tokenized_data_lemmatized for word in comment]

        most_common_words[name] = Counter(all_words).most_common()[:50]

        print(f"{name} has been added to the dictionary of most common words")

        # Display the word cloud using matplotlib
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(f'wordclouds/wordcloud_{filename}.svg', format="svg")

        with open(path, "w") as file:
            file.write(str(tokenized_data_lemmatized))

        print(f"Comments saved to {path} successfully.")

with open("most_common_words.txt", "w", encoding="utf-8") as file:
    file.write(str(most_common_words))
print(most_common_words)
