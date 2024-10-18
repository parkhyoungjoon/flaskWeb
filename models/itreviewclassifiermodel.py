import torch.nn as nn
# 모델 정의
class ReviewClassifierModel(nn.Module):
    def __init__(self, n_vocab, hidden_dim, embedding_dim, n_classes,
                 n_layers, dropout=0.3, bidirectional=True) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.multi_classifier = nn.Linear(lstm_output_dim, n_classes)
        self.binary_classifier = nn.Linear(lstm_output_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        lstm_output, _ = self.lstm(embeddings)
        last_output = lstm_output[:, -1, :]  # 마지막 타임스텝의 출력
        last_output = self.dropout(last_output)
        classesd = self.multi_classifier(last_output)
        logits = self.binary_classifier(last_output)
        return classesd, logits