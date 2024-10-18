import torch
import numpy as np
import torch.nn as nn
from torchvision.transforms import v2
import os
import pickle
import string
import re

def wordtoVector(text,loaded_vectorizer):
    return loaded_vectorizer.transform(text).toarray()

def detectMbti(resultWord, snmodel):
    score = 0
    perce = 0
    for word in resultWord:
        dataTS = torch.FloatTensor(word)
        # 새로운 데이터에 대한 예측 즉, predict
        snmodel.eval()
        with torch.no_grad():
        #     # 추론 / 평가
            outputs = snmodel(dataTS).view(-1)
            predicted = torch.round(torch.sigmoid(outputs)).item()
            perced = torch.sigmoid(outputs).item() * 100
            score += int(predicted)
            perce += float(perced)
    return checkMbti(score,perce)

def checkMbti(score,perce):
    r_score = score / 5
    r_perce = round(perce / 5,2)
    if round(r_score): return 1, r_perce
    else : return 0, r_perce

# 이미지 변환 함수
def image_change(img):
    transConvert = v2.Compose([
        v2.Resize(size=(224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
    ])
    img = transConvert(img)
    return img

# 예측 결과에 따른 피부 질환 정보 반환 함수
def get_skin_di(num,target):
    return target.loc[num, target.columns[1]]

def load_vocab(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        vocab = pickle.load(f)
    return {token: idx for idx, token in enumerate(vocab)}

def split_sentences(text,kkma):
    sentences = kkma.sentences(text)
    return sentences

def load_stopwords(DATA_PATH):
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        stop_words = set([line.strip() for line in f])
    return stop_words

# 전처리 함수
def preprocess_text(text, punc):
    for p in punc:
        text = text.replace(p, '')
    text = re.sub('[^ ㄱ-ㅣ가-힣]+', ' ', text)
    return text.strip()

# 토큰화 및 불용어 제거 함수
def tokenize_and_remove_stopwords(tokenizer, texts, stop_words):
    tokens = [tokenizer.morphs(text) for text in texts]
    tokens = [[token for token in doc if token not in stop_words] for doc in tokens]
    return tokens

# 인코딩 함수
def encoding_ids(token_to_id, tokens, unk_id):
    return [[token_to_id.get(token, unk_id) for token in doc] for doc in tokens]

# 패딩 함수
def pad_sequences(sequences, max_length, pad_value):
    padded_seqs = []
    for seq in sequences:
        seq = seq[:max_length]  # 최대 길이에 맞춤
        pad_len = max_length - len(seq)
        padded_seq = seq + [pad_value] * pad_len
        padded_seqs.append(padded_seq)
    return np.array(padded_seqs)

def analyze_review(model, sentence_tensor):
    classesd, logits = model(sentence_tensor)
    return classesd, logits