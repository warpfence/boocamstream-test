import streamlit as st

st.title('Book Recommendation Wrap-up Report')
st.write('Recsys-3조 ( INT로조 )')

from st_pages import Page, add_page_title, show_pages
show_pages(
    [
        Page('pages/page_overview.py', '1. 프로젝트 개요'),
        Page('pages/page_process.py', '2. 프로젝트 수행 절차 및 방법'),
        Page('pages/page_eda.py', '3. EDA (Exploratory Data Analysis)'),
        Page('pages/page_model_interworking.py', '4. 모델 구동하기')
    ]
)