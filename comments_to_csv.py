import os
import csv
import re
from dotenv import load_dotenv
import googleapiclient.discovery

load_dotenv()
yt_data_api_key = os.getenv("API_KEY")

youtube = googleapiclient.discovery.build(
    "youtube", "v3", developerKey=yt_data_api_key)

video_id = ""
csv_filename = f'comments_{video_id}.csv'

with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Text', 'Author', 'Likes', 'Date'])


def append_to_csv(api_response, csv_file, counter):
    # Initialize a list to store the extracted comment data
    comments_data = []

    # Extract information for top-level comments
    for item in api_response['items']:
        top_level_comment = item['snippet']['topLevelComment']['snippet']
        text = top_level_comment['textOriginal']
        text = re.sub("\n", " ", text)
        author = top_level_comment['authorDisplayName']
        likes = top_level_comment['likeCount']
        date = top_level_comment['publishedAt']

        # Append the extracted data to the list
        comments_data.append((text, author, likes, date))

        # Extract information for replies
        if 'replies' in item:
            replies = item['replies']['comments']
            for reply in replies:
                reply_text = reply['snippet']['textOriginal']
                reply_author = reply['snippet']['authorDisplayName']
                reply_likes = reply['snippet']['likeCount']
                reply_date = reply['snippet']['publishedAt']

                # Append the extracted reply data to the list
                comments_data.append(
                    (reply_text, reply_author, reply_likes, reply_date))

    counter += len(comments_data)

    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(comments_data)

    print(f"The CSV file now contains {counter} comments")
    return counter


def get_comments(videoID):
    nb_comments = 0

    request = youtube.commentThreads().list(
        part="replies,snippet",
        # order="relevance",
        videoId=videoID
    )

    response = request.execute()

    nb_comments = append_to_csv(response, csv_filename, nb_comments)

    while response.get('nextPageToken', None) and nb_comments <= 100000:
        request = youtube.commentThreads().list(
            part='replies,snippet',
            # order="relevance",
            videoId=videoID,
            pageToken=response['nextPageToken']
        )
        response = request.execute()

        nb_comments = append_to_csv(response, csv_filename, nb_comments)

    print(
        f"Finished fetching comments for {videoID}. {nb_comments} comments found.")


get_comments(video_id)
