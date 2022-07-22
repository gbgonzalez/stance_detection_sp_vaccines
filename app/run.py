import argparse
import torch
import csv
from processing import hydrated_tweets, load_dataset
from model import dataloader_embedding, train_bilstm
from utils.utils import set_seed, model_metrics_evaluate
from pathlib import Path

if __name__ == "__main__":

    #get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", "--batch_size", type=str, help="Batch size")
    parser.add_argument("-lr", "--learning_rate", type=str, help="Learning rate in training model")
    parser.add_argument("-epochs", "--epochs", type=str, help="Number of epochs for train")
    parser.add_argument("-max_len", "--max_len", type=str, help="Maximum length of each of the documents in the corpus")
    parser.add_argument("-d", "--dropout", type=str, help="Maximum length of each of the documents in the corpus")
    parser.add_argument("-ae", "--adam_epsilon", type=str, help="Maximum length of each of the documents in the corpus")
    parser.add_argument("-o", "--option", type=str, help="Select an opton    ")
    args = parser.parse_args()

    path = Path.cwd()

    if args.option == "hydrated_tweets":
        tsv_train_route = f"{str(path)}/data/train_dehydrated.tsv"
        tsv_test_route = f"{str(path)}/data/test_dehydrated.tsv"
        tsv_train_export = f"{str(path)}/data/train.tsv"
        tsv_train_export = f"{str(path)}/data/test.tsv"

        #Credential Twitter API
        credentials = {
            "CONSUMER_KEY": "XXXX",
            "CONSUMER_SECRET": "XXXX",
            "ACCESS_TOKEN": "XXXX",
            "ACCESS_SECRET": "XXXX"
        }

        hydrated_tweets(credentials, tsv_train_route, tsv_test_route, tsv_train_export, tsv_train_export)

    # python app\run.py -o train -max_len 128 -batch_size 16 -lr 3e-5 -epochs 2 -d 0.3 -ae 1e-8
    if args.option == "train":
        route_train = f"{str(path)}/data/train.tsv"
        route_test = f"{str(path)}/data/test.tsv"
        model_route = f"{str(path)}/data/model"
        MAX_LEN = int(args.max_len)
        batch_size = int(args.batch_size)
        learning_rate = float(args.learning_rate)
        adam_ep = float(args.adam_epsilon)
        dropout = float(args.dropout)
        epochs = int(args.epochs)

        name_model = "bertin-project/bertin-roberta-base-spanish"

        set_seed(2021)
        train, test = load_dataset(route_train, route_test)

        X_train = [t[0] for t in train]
        y_train = []
        for t in train:
            if int(t[1]) == -1:
                y_train.append(2)
            else:
                y_train.append(int(t[1]))

        print("Start training.....")
        train_data, train_sampler, train_dataloader = dataloader_embedding(MAX_LEN, batch_size, X_train, y_train,
                                                                           name_model)


        input_dim = 768
        hidden_dim = 384
        output_dim = 3

        model = train_bilstm(epochs, dropout, input_dim, hidden_dim, output_dim, name_model, adam_ep, learning_rate,
                             train_dataloader, f"{model_route}")

    if args.option == "test":
        # python app\run.py -o test -max_len 128 -batch_size 32

        MAX_LEN = int(args.max_len)
        batch_size = int(args.batch_size)
        route_model = f"{str(path)}/data/model"
        route_train = f"{str(path)}/data/train.tsv"
        route_test = f"{str(path)}/data/test.tsv"
        name_model = "bertin-project/bertin-roberta-base-spanish"

        train, test = load_dataset(route_train, route_test)

        X_test = [t[0] for t in test]
        y_test = []
        for t in test:
            if int(t[1]) == -1:
                y_test.append(2)
            else:
                y_test.append(int(t[1]))

        model = torch.load(route_model)

        test_data, test_sampler, test_dataloader = dataloader_embedding(MAX_LEN, batch_size, X_test, y_test, name_model)

        accuracy, precision, recall, f1 = model_metrics_evaluate(model, test_dataloader, route_model)

        with open(f"{route_model}/test.tsv", 'w', newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter="\t")
            writer.writerow(["Accuracy", "Precision", "Recall", "F1 Score"])
            writer.writerow([accuracy, precision, recall, f1])

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1}")