
# 모델 불러오기 함수
def load_model(model_idx, model):
    filename = os.path.join(MODEL_PATH, model_files[model_idx] + '_model.pt')
    if model_idx == 0 or model_idx == 3:
        filename += 'h'
    if(model_idx != 3):
        model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    model.eval()  # 평가 모드로 전환
    return model

def load_vocab(vocab_idx):
    vocab_path = os.path.join(MODEL_PATH, model_files[vocab_idx] + '_vocab.pkl')
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return {token: idx for idx, token in enumerate(vocab)}

def split_sentences(text):
    kkma = Kkma()
    sentences = kkma.sentences(text)
    return sentences

def load_stopwords():
    STOP_WORD = os.path.join(DATA_PATH, 'stopwords.txt')
    with open(STOP_WORD, 'r', encoding='utf-8') as f:
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
