from . import views
from django.urls import path

urlpatterns = [
    path('randomForest', views.randomForest, name='randomForest'),
    path('naiveBayes', views.naiveBayes, name='naiveBayes'),
    path('nearestNeighbour', views.nearestNeighbour, name='nearestNeighbour')
]