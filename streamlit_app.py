import streamlit as st
from st_pages import Page, add_page_title, show_pages

show_pages(
    [
        Page('pages/page_overview.py', '1. 프로젝트 개요'),
        Page('pages/page_process.py', '2. 프로젝트 수행 절차 및 방법'),
        Page('pages/page_model_interworking.py', '3. 모델 구동하기')
    ]
)