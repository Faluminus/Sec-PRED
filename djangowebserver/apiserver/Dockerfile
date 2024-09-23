FROM python:latest

WORKDIR /apiserver

RUN pip install torch;/pip install django;/pip install djangorestframework;/pip install scikit-learn;/pip install transformers;/pip install pandas;/pip install numpy

COPY . .

EXPOSE 80/tcp

CMD ["python manage.py","runserver"]