import logging
import threading
from utils.logger import setup_logger
import time
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim import corpora
from gensim.models import LdaModel
import spacy

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

logger = setup_logger()

nlp = spacy.load("en_core_web_sm")

data_lock = threading.Lock()

def validate_data(record):
    required_fields = ["user_id", "video_id", "description", "views", "impressions"]
    for field in required_fields:
        if field not in record or not record[field]:
            logger.warning(f"Missing or empty field {field} in record: {record}")
            return False
    return True

def preprocess_data(data):
    start_time = time.time()
    preprocessed_data = []
    for record in data:
        try:
            if not validate_data(record):
                continue

            logger.info(f"Preprocessing record for user_id: {record['user_id']} and video_id: {record['video_id']}")

            views = record.get("views", 0)
            impressions = record.get("impressions", 1)  
            retention_score = (views / impressions) * 100
            record["retention"] = retention_score
            logger.info(f"Calculated retention score: {retention_score:.2f}%")

            original_interests = record["interests"]
            original_tags = record["tags"]
            record["interests"] = [interest.lower() for interest in record["interests"]]
            record["tags"] = [tag.lower() for tag in record["tags"]]
            logger.info(f"Converted interests from {original_interests} to {record['interests']}")
            logger.info(f"Converted tags from {original_tags} to {record['tags']}")

            doc = nlp(record["description"])
            original_description = record["description"]
            record["description"] = [token.lemma_ for token in doc if not token.is_stop]
            logger.info(f"Processed description from '{original_description}' to '{record['description']}'")

            sentiment = sia.polarity_scores(" ".join(record["description"]))
            record["description_sentiment"] = sentiment
            logger.info(f"Sentiment analysis on description: {sentiment}")

            comments_texts = []
            for comment in record["comments"]:
                comment_doc = nlp(comment["content"])
                comment["content"] = [token.lemma_ for token in comment_doc if not token.is_stop]
                comments_texts.append(" ".join(comment["content"]))
                logger.info(f"Processed comment: {comment['content']}")

            comments_sentiment = [sia.polarity_scores(comment) for comment in comments_texts]
            record["comments_sentiment"] = comments_sentiment
            logger.info(f"Sentiment analysis on comments: {comments_sentiment}")

            dictionary = corpora.Dictionary([comment.split() for comment in comments_texts])
            corpus = [dictionary.doc2bow(comment.split()) for comment in comments_texts]
            lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
            topics = lda_model.print_topics(num_words=4)
            record["comments_topics"] = topics
            logger.info(f"Topic modeling on comments: {topics}")

            watched_views = record["watched_views"]
            record["watched_views"] = [view.rstrip('X') for view in watched_views]
            logger.info(f"Processed watched_views: {record['watched_views']}")

            with data_lock:
                preprocessed_data.append(record)
        except Exception as e:
            logger.error(f"Error preprocessing data for user_id: {record['user_id']} and video_id: {record['video_id']}: {e}")

    end_time = time.time()
    logger.info(f"Preprocessing completed in {end_time - start_time:.2f} seconds.")
    logger.info(f"Preprocessed {len(preprocessed_data)} records.")
    return preprocessed_data
