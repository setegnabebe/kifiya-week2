FROM python3.10.2

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY ..

EXPOSE 8501

ENV STREAMLIT_SERVER_HEADLESS true
ENV STREAMLIT_SERVER_PORT 8501
ENV STREAMLIT_SERVER_ENABLECORS false

CMD["streamlit" ,"run" , "app.py"]