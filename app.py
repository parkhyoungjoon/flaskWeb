from flask import Flask, render_template, request, jsonify
import pickle
import os
import io
import joblib
import torch
from models.mlpmodel import MLPModel
import numpy as np
from modules.dataChangeLib import wordtoVector, detectMbti, image_change, get_skin_di
import modules.dataChangeLib as dcl
from models.skinkitmodel import SkinKitModel
from models.itreviewclassifiermodel import ReviewClassifierModel
from PIL import Image
import pandas as pd
import string
from konlpy.tag import Okt
from konlpy.tag import Kkma

app = Flask(__name__)

def scale_data(data):
    with open('./models/saved_model', 'rb') as f:
        load_model = pickle.load(f)
    # 모델 불러오기
    mmScaler = pickle.load(open('./models/minmax_scaler.pkl', 'rb'))

    # 불러온 모델 사용 예시
    # 예를 들어, 변환할 데이터를 scaler로 변환해보자
    scaled_data = mmScaler.transform(data)
    pre = load_model.predict(scaled_data)
    return "생존" if pre.tolist()[0] else "사망"
# 기본 라우팅 (홈페이지)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/spaceship', methods=['POST'])
def spaceship():
    # 요청에서 form 데이터를 받음
    data = request.form
    space_feature_list = ['spa','vrdeck','room','foodcort','eTrue','erTrue','shop','sTrue','sFalse','age']
    list_data = [int(data[featrue]) for featrue in space_feature_list]
    return scale_data([list_data])

@app.route('/mbti', methods=['POST'])
def mbtichk():
    mbti_list = ['N','S']
    data = request.form.to_dict()
    MODEL_PATH = os.path.dirname(__file__)+ '/models/sn_model.pth'
    VECTOR_LINK = os.path.dirname(__file__)+ '/models/tfidf_vectorizer.pkl'
    loaded_vectorizer = joblib.load(VECTOR_LINK)
    model = torch.load(MODEL_PATH, weights_only=False,map_location=torch.device('cpu'))
    resultWord = wordtoVector(list(data.values()),loaded_vectorizer)
    mbti, score = detectMbti(resultWord,model)
    return jsonify({"mbti":mbti_list[mbti],
                   "score":score if mbti else 100-score
                   })

@app.route('/skin', methods=['POST'])
def skinchk():
    skin_target = pd.read_csv(os.path.dirname(__file__) + '/csv/skin_diseases.csv', header=None)
    # 'image' 필드에서 전송된 파일을 가져옴
    if 'image' not in request.files:
        return jsonify({'error': '파일이 없습니다.'}), 400
    
    file = request.files['image']  # 파일 가져오기
    
    # 모델 경로
    MODEL_PATH = os.path.dirname(__file__) + '/models/skin-model.pth'
    
    # 모델 로드 (CPU에서 실행)
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.eval()  # 모델을 평가 모드로 전환
    
    try:
        # 이미지를 Pillow로 읽음
        img_file = Image.open(io.BytesIO(file.read())).convert('RGB')  # RGB로 변환
        
        # 이미지 변환
        img = image_change(img_file)
        img = img.unsqueeze(dim=0).float()  # 배치 차원 추가
        
        # 예측
        with torch.no_grad():
            pre = model(img)
        
        # 예측된 클래스 인덱스 추출
        pre_max = int(torch.argmax(torch.softmax(pre, dim=1), dim=1))
        
        # 예측된 클래스에 따른 질환명 반환
        val = get_skin_di(pre_max,skin_target)
        
        # 결과 반환
        return val

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/shop', methods=['POST'])
def shopchk():
    aceptList = ['디자인', '사이즈', '가격', '음량/음질', '화질', '제조일/제조사', '품질', '기능', '조작성',
     '제품구성', '소음', '색상', '편의성', '무게', '시간/속도', '용량', '내구성',
     '전력 및 품질 관련', '소재']
    MODEL_PATH = os.path.dirname(__file__) + '/models/it_model.pth'
    VOCAB_PATH = os.path.dirname(__file__) + '/models/it_vocab.pkl'
    STOPWORDS_PATH = os.path.dirname(__file__) + '/data/stopwords.txt'

    tokenizer = Okt()
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    vocab = dcl.load_vocab(VOCAB_PATH)
    stopwords = dcl.load_stopwords(STOPWORDS_PATH)
    punc = string.punctuation
    MAX_LENGTH = 22
    review = request.form.get('review')
    
    # 리뷰 처리
    sentences = dcl.split_sentences(review,Kkma())
    pre_sentences = [dcl.preprocess_text(sentence, punc) for sentence in sentences]
    sentence_tokens = dcl.tokenize_and_remove_stopwords(tokenizer, pre_sentences, stopwords)
    sentence_ids = dcl.encoding_ids(vocab, sentence_tokens, vocab.get('<unk>'))

    # 입력 데이터 패딩
    input_data = dcl.pad_sequences(sentence_ids, MAX_LENGTH, vocab.get('<pad>'))
    sentence_tensor = torch.tensor(input_data, dtype=torch.long)
    
    # 분석
    classesd, logits = dcl.analyze_review(model, sentence_tensor)
    classesd = torch.argmax(classesd, dim=1)
    logits = torch.sigmoid(logits)
    classesd = classesd.tolist()
    logits = logits.tolist()
    result_dict = {'category':[],'target':[]}
    for i in range(len(classesd)):
        result_dict['category'].append(aceptList[classesd[i]])
        result_dict['target'].append('긍정' if logits[i][0] > 0.5 else '부정')
    try:
       
        return jsonify(result_dict)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)