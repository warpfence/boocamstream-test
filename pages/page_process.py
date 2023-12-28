import streamlit as st

st.title('Book Recommendation Wrap-up Report')
st.write('Recsys-3조 ( INT로조 )')
st.write('')
st.write('')
st.markdown('''
# 2. 프로젝트 수행 절차 및 방법

## 3-1. 프로젝트 타임라인
''')
st.image('resource/schedule.png')
st.markdown('''
## 3-2. 프로젝트 파이프라인

- 프로젝트 진행은 아래와 같은 파이프라인으로 팀원 모두 end-to-end 방식으로 진행하였다.
''')
st.image('resource/pipeline.png')
st.markdown('''
- EDA → 가설 수립 → 모델링 → 평점 예측 → **결과 분석** → 앙상블 → 제출
    - 결과 분석 과정에서 만족하지 못한 결과가 나오면 EDA 과정부터 다시 진행했다.
    - 모델링 과정에서 하이퍼 파라미터 튜닝을 함께 진행하도록 한다.

## 3-3. 프로젝트 진행

1. 각자 데이터 확인 후 생각하기에 좋은 결과를 낼 것 같은 모델을 사용한다.
2. 데이터 확인 후 효율적인 전처리 방법이 있다면 공유 및 사용한 모델의 점수를 공유한다.
3. 성능 좋은 모델들 위주로 앙상블을 진행하여 리더보드에 기록한다.
4. 앙상블이 끝난 데이터에 대해서도 다른 앙상블 결과와 앙상블 하여 결과를 확인한다.
5. 최대한 가장 좋은 모델 하이퍼 파라미터 튜닝 후 앙상블을 진행하도록 한다.
''')