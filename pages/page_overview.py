import streamlit as st

st.title('Book Recommendation Wrap-up Report')
st.write('Recsys-3조 ( INT로조 )')
st.write('')
st.write('')
st.markdown('''
# 1. 프로젝트 개요
책과 관련된 정보와 소비자의 정보, 그리고 소비자가 실제로 부여한 평점, 총 3가지의 데이터 셋(users.csv, books.csv, train_ratings.csv)을 활용하여 각 사용자가 주어진 책에 대해 얼마나 평점을 부여할지에 대해 예측하는 것이다.

학습 데이터는 306,795건의 평점 데이터(`train_rating.csv`)이며, 149,570건의 책 정보(`books.csv`) 및 68,092명의 고객 정보(`users.csv`) 또한 주어진다.

각각 데이터는 다음의 형태를 띄고 있다.

### Input

- `train_ratings.csv` : 각 사용자가 책에 대해 평점을 매긴 내역
- `users.csv` : 사용자에 대한 정보
- `books.csv` : 책에 대한 정보
- `Image/` : 책 이미지
''')
st.image('resource/input_1.png')
st.image('resource/input_2.png')
st.image('resource/input_3.png')
st.image('resource/input_4.png')
st.image('resource/input_5.png')
st.markdown('''
### Output

- test_ratings.csv 의 사용자가 주어진 책에 대해 매길 것이라고 예상하는 평점

## 1-1. 평가 데이터

- 26,167명의 사용자(user)가 52,000개의 책에 대해 남길 것으로 기대하는 76,699건의 평점(rating)을 예측한다.

## 1-2. 평가 지표

- 평점 예측에서 자주 사용되는 지표 중 하나인 RMSE (Root Mean Square Error)를 사용한다.
- 모델의 예측이 얼마나 잘못 되었는지를 수치로 나타내고 모델이 예측을 잘할수록 지표는 0 에 수렴한다.
''')
st.image('resource/evaluation.png')
st.markdown('''
### RMSE Formula
''')
st.image('resource/rmse_formula.png')
st.markdown('''
## 1-3. 프로젝트 협업 전략

### Git

- 해당 프로젝트의 협업툴로 Git을 사용한다.
- Git 관리 전략 중 GitHub Flow를 사용하며 Remote 중심 Branch 전략으로 배포의 중심이 되는 main 브랜치와 각각의 feature 브랜치로 구성하고 있다.
''')
st.image('resource/github_flow.png')
st.link_button('level1-bookratingprediction-recsys-03', 'https://github.com/boostcampaitech6/level1-bookratingprediction-recsys-03')
st.markdown('''
### Commit Convention

- **Issue**
    - 기능 추가 또는 테스트를 수행할 Task를 생성하고 수행시 새로운 브랜치를 만든다.
- **Pull requests**
    - 브랜치를 생성하여 작업이 완료되면 main 브랜치로 병합하는 과정에서 다른 팀원의 승인을 받는다.

### Code Commit

- 구현한 기능에 대해 자유롭게 작성하되 머리말과 내용에 아래와 같은 형식을 따른다.
    - 머리말 : [(브랜치명 소문자)]
    - 내용 : #(이슈번호)

### Header / Body / Footer

- 제목 머리말 : [(브랜치 이름 대문자)] (내용)
- 내용 : Issue 또는 Pull requests 템플릿에 따른다.
- 꼬리말 : Issue에 대한 Pull requests 처리시 Issue Tags란에 closed로 이슈 종료를 표시한다.

## 1-4. 서버 구성 및 개발 환경

### 서버 구성

- **AI Stage GPU Cloud 서버**
    - OS 및 버전 : Ubuntu 20.04.6 LTS
    - 성능 : V100 GPU

### 개발 환경

- 사용 언어 : Python
- 버전 정보 : 3.10.13
- 패키지 정보

```
numpy==1.26.2
pandas==2.1.4
matplotlib==3.8.2
scikit-learn==1.3.2
jupyter_client==8.6.0
jupyter_core==5.5.0
seaborn==0.13.0
torch==1.12.1+cu113
torchaudio==0.12.1+cu113
torchvision==0.13.1+cu113
```
''')