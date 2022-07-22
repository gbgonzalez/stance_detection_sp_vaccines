from transformers import RobertaTokenizer, RobertaModel
import torch

class EmbeddingLSTM(torch.nn.Module):
    def __init__(self, dropout_rate, input_dim, hidden_dim, output_dim, bert_config):

        super(EmbeddingLSTM, self).__init__()

        self.num_class = output_dim
        self.bert_config = bert_config
        self.tokenizer = RobertaTokenizer.from_pretrained(self.bert_config)
        self.model = RobertaModel.from_pretrained(self.bert_config)
        self.dropout_rate = dropout_rate
        self.lstm_input_size = input_dim
        self.lstm_hidden_size = hidden_dim
        self.lstm = torch.nn.LSTM(input_size=self.lstm_input_size,
                                  hidden_size=self.lstm_hidden_size,
                                  bidirectional=True)
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.fc = torch.nn.Linear(in_features=2 * self.lstm_hidden_size,  # LSTM stacked hidden state
                                  out_features=self.num_class)

    def forward(self, sents_tensor, masks_tensor):
        bert_output = self.model(input_ids=sents_tensor, attention_mask=masks_tensor)
        encoded_layers = bert_output[0].permute(1, 0, 2)  # permute dimensions to fit LSTM input
        enc_hiddens, (last_hidden, last_cell) = self.lstm(encoded_layers)
        output_hidden = torch.cat((last_hidden[0, :, :], last_hidden[1, :, :]), dim=1)
        output_hidden = self.dropout(output_hidden)
        output = self.fc(output_hidden)
        return output
