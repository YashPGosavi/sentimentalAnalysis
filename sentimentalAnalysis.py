from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS  # Import CORS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)
api = Api(app)
CORS(app)  # Add this line to enable CORS
sia = SentimentIntensityAnalyzer()

class SentimentAnalysis(Resource):
    def post(self):
        data = request.get_json()
        reviews_data = data.get('reviews', [])

        grouped_reviews = {}
        total_rating = 0
        rating_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Bot': 0}

        for review in reviews_data:
            user = review.get('user', '')
            review_title = review.get('review_title', '')            
            text = review.get('comment', '')
            rating = int(review.get('rating', 0))
            sentiment_scores = sia.polarity_scores(text)
            length = len(text.split())

            if sentiment_scores['compound'] > 0.1 and length >= 5:
                group, buy, tag = 'Positive', True, 'Positive'
            elif sentiment_scores['compound'] < -0.1 and length >= 5:
                group, buy, tag = 'Negative', False, 'Negative'
            elif sentiment_scores['compound'] == 0 and length < 5:
                group, buy, tag = 'Neutral', False, 'Neutral'
            else:
                group, buy, tag = 'Bot', False, 'Bot'

            if group not in grouped_reviews:
                grouped_reviews[group] = []
            grouped_reviews[group].append({
                'user': user, 
                'review_title': review_title, 
                'text': text, 
                'sentiment': group, 
                'buy': buy, 
                'rating': rating,
                'tag': tag  # Adding the tag to the review data
            })

            total_rating += rating
            rating_counts[rating] += 1
            sentiment_counts[group] += 1

        positive_reviews = [review for review in grouped_reviews.get('Positive', []) if review['buy']]
        positive_percentage = len(positive_reviews) / len(reviews_data) * 100
        negative_percentage = sentiment_counts['Negative'] / len(reviews_data) * 100
        neutral_percentage = sentiment_counts['Neutral'] / len(reviews_data) * 100
        bot_percentage = sentiment_counts['Bot'] / len(reviews_data) * 100

        average_rating = total_rating / len(reviews_data) if len(reviews_data) > 0 else 0
        rating_percentages = {rating: count / sum(rating_counts.values()) * 100 for rating, count in rating_counts.items()}
        sentiment_percentages = {sentiment: count / len(reviews_data) * 100 for sentiment, count in sentiment_counts.items()}

        overall_recommendation = (
            'Based on the overwhelmingly positive reviews, I highly recommend purchasing this product.'
            if positive_percentage >= 50
            else 'While there are some negative reviews, I\'d suggest taking a closer look at the overall feedback. If the product meets your specific needs, it could still be a worthwhile purchase despite a few negative reviews.'
        )

        return {
            'grouped_reviews': grouped_reviews,
            'positive_percentage': positive_percentage,
            'negative_percentage': negative_percentage,
            'neutral_percentage': neutral_percentage,
            'bot_percentage': bot_percentage,
            'average_rating': average_rating,
            'rating_percentages': rating_percentages,
            'sentiment_percentages': sentiment_percentages,
            'overall_recommendation': overall_recommendation
        }

@app.route('/')
def index():
    return 'Hello, World!'

api.add_resource(SentimentAnalysis, '/sentimentalAnalysis')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
