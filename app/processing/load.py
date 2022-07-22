import tweepy
import csv

def hydrated_tweets(credentials, tsv_train_route, tsv_test_route, train_export_route, test_export_route):
    auth = tweepy.OAuthHandler(credentials["CONSUMER_KEY"], credentials["CONSUMER_SECRET"])
    auth.set_access_token(credentials["ACCESS_TOKEN"], credentials["ACCESS_SECRET"])
    api = tweepy.API(auth, wait_on_rate_limit=True)

    train = _tsv_load_data(tsv_train_route, api)
    test = _tsv_load_data(tsv_test_route, api)

    _tsv_export(train, train_export_route)
    _tsv_export(test, test_export_route)


def load_dataset(route_train, route_test):
    train = _load_tweets(route_train)
    test = _load_tweets(route_test)
    return train, test

def _tsv_load_data(route_tsv, api):
    train = []
    with open( route_tsv, 'r', encoding="UTF8") as csv_file:
        rows = csv.reader(csv_file, delimiter="\t")
        for row in rows:
            tweet = api.get_status(row[0], tweet_mode="extended")
            text_tweet = tweet.full_text.replace("\n", " ")
            text_tweet = text_tweet.full_text.replace("\t", " ")
            train.append([text_tweet, row[1]])
    return train

def _tsv_export(data, route_tsv):
    with open(route_tsv, 'w', encoding="UTF8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter="\t")
        for d in data:
            writer.writerow([d[0], d[1]])

def _load_tweets(route):
    data = []
    with open(route, 'r', encoding="UTF8") as csv_file:
        rows = csv.reader(csv_file, delimiter="\t")
        for row in rows:
            data.append(row)

    return data


