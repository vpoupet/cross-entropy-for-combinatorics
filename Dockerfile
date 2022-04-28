FROM sagemath/sagemath:latest

COPY ./requirements.txt .
RUN sage --python -m pip install -r requirements.txt
