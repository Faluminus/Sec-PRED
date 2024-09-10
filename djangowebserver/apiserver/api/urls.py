from django.urls import path
from . import views

urlpatterns = [
    path("predict/",views.TransformerPredict, name="transformer-predict")
]