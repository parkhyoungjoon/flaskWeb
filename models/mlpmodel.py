# 5. MLP 모델 정의 (2층 레이어)
import torch
import torch.nn as nn
import torch.nn.functional as F
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPModel, self).__init__()

        # 은닉층들을 nn.ModuleList로 관리
        layers = []
        in_features = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())  # 활성화 함수로 ReLU 추가
            in_features = hidden_size  # 다음 레이어의 입력 크기는 현재 레이어의 출력 크기

        # Dropout과 마지막 출력층 추가
        layers.append(nn.Dropout(0.3))  # Dropout 추가
        layers.append(nn.Linear(in_features, output_size))

        self.network = nn.Sequential(*layers)  # Sequential로 레이어 묶음

    def forward(self, x):
        return self.network(x)