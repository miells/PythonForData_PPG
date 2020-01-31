from django.shortcuts import render
from django.http import HttpResponse
import pickle
import pandas as pd

# Dictionnaire comprenant toutes les activités
activities = {'0.0': 'Transient period', '1.0': 'Sitting', '2.0': 'Ascending and descending stairs', '3.0': 'Table soccer', '4.0': 'Cycling', '5.0': 'Driving a car', '6.0': 'Lunch break', '7.0': 'Walking', '8.0': 'Working'}

# Create your views here.
def index(request):
    return HttpResponse("You are on the main page")


def randomForest(request):
    loaded_model = pickle.load(open('predicteur_app/models/rf_model.sav', 'rb'))

    keys = [ 'male','rpeakcount','label','weight','age','height','skin','sport','chestACC0','chestACC1','chestACC2','chestECG0','chestResp0','wristACC0','wristACC1','wristACC2','wristBVP0','wristEDA0','wristTEMP0']
    values = []
    for key in keys:
        values.append(request.GET[key])
    datatest = pd.DataFrame([values], columns = keys)
    prediction = str(loaded_model.predict(datatest)[0])
    display = {k:v for (k,v) in activities.items() if prediction == k}
    return  HttpResponse(str(display))


def naiveBayes(request):
    loaded_model = pickle.load(open('predicteur_app/models/nb_model.sav', 'rb'))

    keys = [ 'male','rpeakcount','label','weight','age','height','skin','sport','chestACC0','chestACC1','chestACC2','chestECG0','chestResp0','wristACC0','wristACC1','wristACC2','wristBVP0','wristEDA0','wristTEMP0']
    values = []
    for key in keys:
        values.append(request.GET[key])
    datatest = pd.DataFrame([values], columns = keys)
    prediction = str(loaded_model.predict(datatest)[0])
    display = {k:v for (k,v) in activities.items() if prediction == k}
    return  HttpResponse(str(display))

def nearestNeighbour(request):
    loaded_model = pickle.load(open('predicteur_app/models/kn_model.sav', 'rb'))

    keys = [ 'male','rpeakcount','label','weight','age','height','skin','sport','chestACC0','chestACC1','chestACC2','chestECG0','chestResp0','wristACC0','wristACC1','wristACC2','wristBVP0','wristEDA0','wristTEMP0']
    values = []
    for key in keys:
        values.append(request.GET[key])
    datatest = pd.DataFrame([values], columns = keys)
    prediction = str(loaded_model.predict(datatest)[0])
    display = {k:v for (k,v) in activities.items() if prediction == k}
    return  HttpResponse(str(display))