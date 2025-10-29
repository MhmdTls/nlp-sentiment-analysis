
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json



sid = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    if scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return sentiment, scores

def analyze_multiple_texts(texts):
    results = []
    for text in texts:
        sentiment, scores = analyze_sentiment(text)
        results.append({
            'text': text,
            'sentiment': sentiment,
            'scores': scores
        })
    return results

def save_results_to_file(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    print("Welcome to the Sentiment Analyzer!")
    
    texts = []
    while True:
        user_text = input("Please enter the text you want to analyze (or type 'done' to finish): ")
        if user_text.lower() == 'done':
            break
        texts.append(user_text)
    
    if texts:
        results = analyze_multiple_texts(texts)
        
        print("\\nSentiment Analysis Results:")
        for result in results:
            print("\\nText:", result['text'])
            print("Sentiment:", result['sentiment'])
            print("Scores:", result['scores'])
        
        save_choice = input("\\nDo you want to save the results to a file? (yes/no): ")
        if save_choice.lower() == 'yes':
            filename = input("Enter the filename (with .json extension): ")
            save_results_to_file(results, filename)
            print(f"Results saved to {filename}")
        else:
            print("Results not saved.")
    else:
        print("No text entered for analysis.")

if __name__ == "__main__":
    main()