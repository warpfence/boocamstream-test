import streamlit as st
import argparse
import pandas as pd
import numpy as np
from src.utils import train

if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'books' not in st.session_state:
    st.session_state.book = None

users = pd.read_csv('resource/data/user_id.csv')
books = pd.read_csv('resource/data/isbn.csv')
user2idx = {v:k for k,v in enumerate(users['user_id'].unique())}
isbn2idx = {v:k for k,v in enumerate(books['isbn'].unique())}
users['user_id'] = users['user_id'].map(user2idx)
books['isbn'] = books['isbn'].map(isbn2idx)
users_sample = users.sample(5)['user_id'].unique()
books_sample = books.sample(5)['isbn'].unique()
rating = 0

st.markdown('''
# 4. 모델 구동하기
## 모델명 : FM_DCN
### FM 모델과 DCN 모델을 활용한 새로운 FM + DCN 모델을 만들어 성능 테스트
- feature에 대해 깊게 학습하고 비선형 관계를 학습하기 효율적인 교차 학습 조합으로 좋은 성능을 발휘할 수 있지 않을까 하여 모델 생성해보았다.
---
### 책 평점 예측
''')

col1, col2 = st.columns(2)

with col1:
    sb_user_id = st.selectbox(
        "평점을 예측할 user id를 선택하세요.",
        users_sample,
        placeholder="Select user id"
    )
    st.write('선택한 user id : ', sb_user_id)

with col2:
    sb_book = st.selectbox(
        "평점을 예측할 isbn을 선택하세요.",
        books_sample,
        placeholder="Select isbn"
    )
    book = sb_book
    st.write('선택한 book : ', sb_book)

data = {}
args = {}

parser = argparse.ArgumentParser(description='parser')
arg = parser.add_argument
args = parser.parse_args()
data['test_dataloader'], data['field_dims'] = [int(sb_user_id), int(sb_book), 0, 0, 0, 0, 0], [68069, 149570, 6, 12267, 1609, 11571, 62059]
args.embed_dim, args.num_layers, args.dropout, args.mlp_dims = 16, 3, 0.1, [16, 16]

rating = train(args, data)

st.write(f'선택한 user id와 book의 예측 평점 : ', round(rating[0],2), '입니다.')