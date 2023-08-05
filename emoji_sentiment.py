import spacy
import re
from scipy.spatial.distance import cosine

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("emoji", first=True)

positive_vectors = [nlp("positive").vector]

negative_vectors = [nlp("negative").vector, nlp("sad").vector, nlp("bad").vector, nlp(
    "horrible").vector, nlp("terrible").vector, nlp("dislike").vector, nlp("hate").vector]

to_analyse = [('😅', 232455), ('😂', 205671), ('❤', 93984), ('🎉', 92341), ('😮', 79554), ('🤣', 54294), ('😊', 54057), ('😢', 32181), ('💩', 16266), ('stream', 11853), ('🤢', 10192), ('view', 8620), ('\u200b', 8452), ('️', 6006), ('💜', 5415), ('🐰', 4309), ('😎', 3970), ('love', 3804), ('😍', 3534), ('🤩', 3512), ('\u200d', 3350), ('💓', 3342), ('🌺', 3217), ('😃', 3111),
              ('solo', 3076), ('(', 2944), ('🏻', 2937), (')', 2825), ('🖕', 2811), ('🙄', 2690), ('😡', 2629), ('>', 2607), ('go', 2591), ('<', 2548), ('u', 2521), ('best', 2470), ('😋', 2384), ('💗', 2331), ('ben', 2285), ('🌙', 2256), ('💋', 2248), ('🤗', 2165), ('🖤', 2141), ('let', 2051), ('😵', 2041), ('’', 1930), ('🐿', 1865), ('♡', 1864), ('bu', 1822), ('zeh', 1822)]

for element in to_analyse:
    if re.match(r'[^\w\s,]', element[0]):
        vector = nlp(element[0]).vector

        positive_distances = []
        for pos_vec in positive_vectors:
            positive_distances.append(cosine(vector, pos_vec))
        average_pos_dist = sum(positive_distances) / len(positive_distances)

        negative_distances = []
        for neg_vec in negative_vectors:
            negative_distances.append(cosine(vector, neg_vec))
        average_neg_dist = sum(negative_distances) / len(negative_distances)

        print(f"{element[0]}\nDistance from positive terms : {average_pos_dist}\nDistance from negative terms : {average_neg_dist}\n0 is the closedt and 1 the furthest\n")
