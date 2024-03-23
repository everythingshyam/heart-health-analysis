from django.db import IntegrityError

from django.shortcuts import  render, redirect
from .forms import NewUserForm
from django.contrib.auth import login
from django.contrib import messages
from .forms import NewUserForm
from django.contrib.auth import login, logout, authenticate #add this
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm #add this
#from projectApp.models import Register

import os
import time
import random

import requests
from django.shortcuts import render, HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
#from .forms import UserRegisterForm
from django.contrib.auth.forms import UserCreationForm
import numpy as np
import joblib
from django.contrib.auth.models import User
import pandas as pd, numpy as np, re
import os
from sklearn.metrics import classification_report, accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
import tkinter as tk
from sklearn import svm
from PIL import Image, ImageTk
from tkinter import ttk
from joblib import dump , load
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pickle
import nltk
import matplotlib.pyplot as plt

#######################################################################################################
nltk.download('stopwords')
stop = stopwords.words('english')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#######################################################################################################

# Create your views here.
def index(request):
        return render(request, 'index.html')
        
def input(request):
    return render(request, 'input.html')

        
def fakenew(request):
    return render(request, 'fakenew.html')


def result(request):
        #if request.POST.get('action') == 'post':
            lis = []       
            # Receive data from client
            lis.append(request.GET['Age'])
            lis.append(request.GET['Gender'])
            lis.append(request.GET['Chest_Pain'])
            lis.append(request.GET['Restbp'])
            lis.append(request.GET['Chol'])
            lis.append(request.GET['FBS'])
            lis.append(request.GET['Restecg'])
            lis.append(request.GET['thalach'])
            lis.append(request.GET['Exang'])
            lis.append(request.GET['Oldpeak'])
            lis.append(request.GET['slope'])
            lis.append(request.GET['ca'])
            lis.append(request.GET['thal'])
            print(lis) 


            # Traning model
            from joblib import dump , load
            model=load('C:/Users/shyam/Documents/Coding/heart-health-analysis/projectApp/model/heart_model.joblib')
            # model=load('C:/Users/shyam/OneDrive/Desktop/100%heart_disease_detection_web_updated/100%heart_disease_detection_web_updated/100%heart_disease_detection_web_updated/heart_disease_detection_web/Hello/projectApp/model/heart_model.joblib')
            
            # Make prediction
            result = model.predict([lis])

            if result[0]==0:
                print("Heart Disease  ")
                value = f'Heart is healthy.'
                
            else:
                print("Normal Heart")
                value = 'Heart is not healthy.'

            #label4 = tk.Label(root,text ="Normal Speech",width=20,height=2,bg='#FF3C3C',fg='black',font=("Tempus Sanc ITC",25))
            #label4.place(x=450,y=550)
    
            return render(request,'result.html',  {
                      'ans': value,
                      'title': 'Predict Heart Disease ',
                      'active': 'btn btn-success peach-gradient text-white',
                      'result': True,
                      
                  })
    

def result_new(request):
        #if request.POST.get('action') == 'post':
            # Traning model
            from joblib import dump , load
            model=load('C:/Users/shyam/Documents/Coding/heart-health-analysis/projectApp/model/SVM_model.joblib')
            # model=load('C:/Users/shyam/OneDrive/Desktop/100%heart_disease_detection_web_updated/100%heart_disease_detection_web_updated/100%heart_disease_detection_web_updated/heart_disease_detection_web/Hello/projectApp/model/SVM_model.joblib')
            # Receive data from client
            lis = []       
            # Receive data from client
            param_age=int(request.GET['Age'])
            lis.append(param_age)

            param_gender=int(request.GET['Gender'])
            lis.append(param_gender)
            if(param_gender==0):
                param_gender='Male'
            else:
                param_gender='Female'

            param_cp=int(request.GET['Chest_Pain'])
            lis.append(param_cp)

            param_restbp=int(request.GET['Restbp'])
            lis.append(param_restbp)

            param_chol=int(request.GET['Chol'])
            lis.append(param_chol)

            param_fbs=int(request.GET['FBS'])
            lis.append(param_fbs)

            param_restecg=int(request.GET['Restecg'])
            lis.append(param_restecg)

            param_thalach=int(request.GET['thalach'])
            lis.append(param_thalach)

            param_exang=int(request.GET['Exang'])
            lis.append(param_exang)

            param_oldpeak=float(request.GET['Oldpeak'])
            lis.append(param_oldpeak)

            param_slope=int(request.GET['slope'])
            lis.append(param_slope)

            param_ca=int(request.GET['ca'])
            lis.append(param_ca)

            param_thal=int(request.GET['thal'])
            lis.append(param_thal)

            print(lis) 

            # HEALTH ########################################
            result = model.predict([lis])
            pred = 100-np.round(result)[0]
            print("\n",pred)
            print("\n")
            if (pred >= 50):
                print("Good")
                value = 'Your heart is in good state.'
            # elif ((pred > 51) & (pred < 76)):
            #     print("Medium")
            #     value = 'Medium'
            elif ((pred < 50) & (pred > 24)):
                print("Medium")
                value = 'It is good. Can be better though.'
            else:
                print("Bad")
                value = 'Seem bad. You should see a doctor.'

            # price = "Heart Health: "+str(100-pred[0])+ "%" + "\n(" + str(value)+")"
            price = str(value)

            # param_cp ########################################
            cp_head='Chest Pain'
            cp_ref='#'
            if(param_cp==0):
                cp_result='Asymptomatic'
                cp_note='This refers to no chest pain at all. It is on the lower end of the intensity spectrum as there is no discomfort experienced.'
            elif param_cp==1:
                cp_result='Atypical Angina'
                cp_note='This type of chest pain is described as atypical because it might not present with the classic symptoms of angina (crushing chest pain radiating to the arm, jaw, etc.). It can feel like pressure, tightness, or discomfort and may be mistaken for indigestion or other issues.'
            elif param_cp==2:
                cp_result='Non Anginal'
                cp_note='This category encompasses chest pain that is not related to angina or heart problems. It can arise from muscle strain, anxiety, gastrointestinal issues, or other causes. While it might be uncomfortable, it is generally less intense than angina.'
            else:
                cp_result='Typical Angina'
                cp_note='This type of chest pain is the most intense and is a classic symptom of angina pectoris. It often feels like a squeezing, pressure, or burning sensation in the chest that can radiate to the arm, jaw, shoulder, or back. It typically occurs with exertion or stress and improves with rest.'

            # param_restbp ########################################
            restbp_head='Resting Blood Pressure'
            restbp_result=str(param_restbp)+'mm Hg : '
            restbp_ref='#'
            if param_restbp<90 :
                restbp_result+='Hypotension (Low Blood Pressure)'
                restbp_note='This category refers to blood pressure readings that fall below the recommended range.'
            elif  param_restbp<119:
                restbp_result+='Normal Blood Pressure'
                restbp_note='This is the ideal range for resting blood pressure.'
            elif  param_restbp<=129:
                restbp_result+='Elevated Blood Pressure'
                restbp_note='This category indicates blood pressure readings that are higher than normal but not yet considered high blood pressure. It serves as a warning sign to take steps towards lowering blood pressure.'
            elif  param_restbp<=139:
                restbp_result+='Stage 1 Hypertension (High Blood Pressure)'
                restbp_note='This is the first stage of high blood pressure, indicating a mild to moderate elevation.'
            else  :
                restbp_result+='Stage 2 Hypertension (High Blood Pressure)'
                restbp_note='This is the second stage of high blood pressure, indicating a more serious elevation.'

            # param_chol ########################################
            chol_head='Cholesterol'
            chol_result=str(param_chol)+'mg/dL : '
            chol_ref='https://www.nhlbi.nih.gov/resources/cholesterol-your-heart-what-you-need-know-fact-sheet'
            if param_chol<200 :
                chol_result+='Optimal Cholesterol Level'
                chol_note='This is the desirable range associated with a low risk of heart disease. Maintaining healthy lifestyle habits like a balanced diet and regular exercise can help keep your LDL low and HDL high.'
            elif  param_chol<239:
                chol_result+='Borderline-High Cholesterol'
                chol_note='This range indicates borderline high cholesterol levels. It is advisable to consult a healthcare professional for personalized advice. Lifestyle changes like dietary modifications and increased physical activity can often help lower LDL cholesterol and improve your overall cardiovascular health.'
            else  :
                chol_result+='High Cholesterol Level'
                chol_note='Cholesterol levels in this range are considered high and require medical attention. A healthcare professional will recommend a treatment plan that may include medication, lifestyle modifications, or both to manage your cholesterol and reduce your risk of heart disease.'

            # param_fbs ########################################
            fbs_head='Fasting Blood Sugar'
            fbs_result=str(param_fbs)+'mg/dL : '
            fbs_ref='https://diabetes.org/'
            if param_fbs<100 :
                fbs_result+='Normal'
                fbs_note='This is the desirable range for fasting blood sugar. It suggests a healthy balance of blood sugar levels.'
            elif  param_fbs<125:
                fbs_result+='Prediabetes'
                fbs_note='This range indicates prediabetes, a condition where your blood sugar levels are higher than normal but not yet high enough to be diagnosed as diabetes. Lifestyle changes like a balanced diet, regular exercise, and maintaining a healthy weight can often help prevent the progression to type 2 diabetes.'
            else  :
                fbs_result+='Diabetes'
                fbs_note='Fasting blood sugar levels in this range are considered diagnostic of diabetes. There are two main types of diabetes: type 1 and type 2. Depending on the type, a healthcare professional will recommend a treatment plan that may include medication, lifestyle modifications, or both to manage your blood sugar levels and prevent complications.'

            # param_restecg ########################################
            restecg_head='Rest ECG'
            restecg_result=str(param_restecg)+'mm'
            restecg_ref='https://www.heart.org/en/health-topics/heart-attack/diagnosing-a-heart-attack/electrocardiogram-ecg-or-ekg'
            restecg_note='Due to limitations in interpreting Rest ECG results without medical expertise, it is not advisable to categorize them into ranges with specific health implications.'

            # param_thalach ########################################
            thalach_head='Maximum Heart Rate Achieved (Thalach)'
            thalach_result=str(param_thalach)+'bpm'
            thalach_ref='https://newsnetwork.mayoclinic.org/discussion/mayo-clinic-minute-how-to-reach-your-target-heart-rate/'
            thalach_note='Thalach (maximum heart rate achieved) during a test helps assess cardiovascular health, but individual variations are significant. Forget generic MHR ranges based on age - consult a healthcare professional for personalized exercise intensity zones based on your actual Thalach and fitness level.'
            
            # param_exang
            # ########################################
            exang_head='Exercise Induced Angina'
            exang_result=str(param_exang)+'bpm'
            exang_ref='https://www.heart.org/en/healthy-living/fitness/fitness-basics/aha-recs-for-physical-activity-in-adults'
            exang_note='Exercise-induced angina (EIA) pain varies between people due to fitness level and artery blockage. There is no specific "exercise range" to trigger it. Focus on the symptoms: chest pain during exertion that goes away with rest. Do not use exercise to diagnose EIA - see a healthcare professional if you experience chest pain during activity.'
            
            # param_oldpeak ########################################
            oldpeak_head='ST Depression (oldpeak)'
            oldpeak_result=str(param_oldpeak)+'mm'
            oldpeak_ref='https://www.heart.org/en/health-topics/heart-attack/diagnosing-a-heart-attack/electrocardiogram-ecg-or-ekg'
            oldpeak_note='ST depression on an ECG can signal reduced oxygen to the heart muscle, but it is not a definitive diagnosis on its own. Doctors consider the depth, location on the ECG, and your medical history to assess its significance. There is no single range for ST depression - the overall ECG picture matters most for proper interpretation by a healthcare professional.'

            # param_slope ########################################
            slope_head='Slope of the peak exercise'
            slope_result=str(param_slope)
            slope_ref='https://pubmed.ncbi.nlm.nih.gov/479779/'
            slope_note='The slope of your peak exercise ST-segment on an ECG during a stress test reflects blood flow to your heart muscle. Upsloping is normal, flat might be okay, and downsloping raises concerns. There is no specific range for the slope - doctors consider the whole picture for interpretation.'

            # param_ca ########################################
            ca_head='No. of major vessels(CA)'
            ca_result=str(param_ca)
            ca_ref='https://www.betterhealth.vic.gov.au/health/conditionsandtreatments/ecg-test'
            ca_note='The number of major vessels seen using fluoroscopy (CA) during a heart procedure indicates potential blockages. Normally, all 3 major vessels are visualized (CA = 3). Fewer visualized vessels (CA = 2, 1, or 0) suggest blockages. There is no set range for "normal" CA - doctors consider the degree of blockage in each vessel, not just the number. '

            # param_thal ########################################
            thal_head='Thalassemia'
            thal_result=str(param_thal)+'g/dL'
            thal_ref='https://www.ncbi.nlm.nih.gov/books/NBK545151/'
            thal_note='Thalassemia, a genetic blood disorder, affects hemoglobin production. There are alpha and beta types, each with varying severity.  The severity is not based on a specific range but on the number of affected genes and its impact on symptoms and hemoglobin levels. It would be better to go to a doctor who would consider the whole clinical picture for classification.'

            # ########################################

            return render(request,'result_new.html',  {
                        'health':pred,
                        'ans': price,
                        #
                        'param_age': param_age,
                        #
                        'param_gender':param_gender,
                        #
                        'param_cp': param_cp,
                        'cp_head': cp_head,
                        'cp_result': cp_result,
                        'cp_ref': cp_ref,
                        'cp_note':cp_note,
                        #
                        'param_restbp': param_restbp,
                        'restbp_head': restbp_head,
                        'restbp_result': restbp_result,
                        'restbp_ref': restbp_ref,
                        'restbp_note':restbp_note,
                        #
                        'param_chol': param_chol,
                        'chol_head': chol_head,
                        'chol_result': chol_result,
                        'chol_ref': chol_ref,
                        'chol_note':chol_note,
                        #
                        'param_fbs': param_fbs,
                        'fbs_head': fbs_head,
                        'fbs_result': fbs_result,
                        'fbs_ref': fbs_ref,
                        'fbs_note':fbs_note,
                        #
                        'param_restecg': param_restecg,
                        'restecg_head': restecg_head,
                        'restecg_result': restecg_result,
                        'restecg_ref': restecg_ref,
                        'restecg_note':restecg_note,
                        #
                        'param_thalach': param_thalach,
                        'thalach_head': thalach_head,
                        'thalach_result': thalach_result,
                        'thalach_ref': thalach_ref,
                        'thalach_note':thalach_note,
                        #
                        'param_exang': param_exang,
                        'exang_head': exang_head,
                        'exang_result': exang_result,
                        'exang_ref': exang_ref,
                        'exang_note':exang_note,
                        ''
                        #
                        'param_oldpeak': param_oldpeak,
                        'oldpeak_head': oldpeak_head,
                        'oldpeak_result': oldpeak_result,
                        'oldpeak_ref': oldpeak_ref,
                        'oldpeak_note':oldpeak_note,
                        #
                        'param_slope': param_slope,
                        'slope_head': slope_head,
                        'slope_result': slope_result,
                        'slope_ref': slope_ref,
                        'slope_note':slope_note,
                        #
                        'param_ca': param_ca,
                        'ca_head': ca_head,
                        'ca_result': ca_result,
                        'ca_ref': ca_ref,
                        'ca_note':ca_note,
                        #
                        'param_thal': param_thal,
                        'thal_head': thal_head,
                        'thal_result': thal_result,
                        'thal_ref': thal_ref,
                        'thal_note':thal_note,
                        #
                      'title': 'Heart Disease Prediction',
                      'active': 'btn btn-success peach-gradient text-white',
                      'result': True,
                  })

def register(request):
	if request.method == "POST":
		form = NewUserForm(request.POST)
		if form.is_valid():
			user = form.save()
			login(request, user)
			messages.success(request, "Registration successful." )
			return redirect('login1')
		messages.error(request, "Unsuccessful registration. Invalid information.")
	form = NewUserForm()
	return render (request=request, template_name="register.html", context={"register_form":form})



def login1(request):
	if request.method == "POST":
		form = AuthenticationForm(request, data=request.POST)
		if form.is_valid():
			username = form.cleaned_data.get('username')
			password = form.cleaned_data.get('password')
			user = authenticate(username=username, password=password)
			if user is not None:
				login(request, user)
				messages.info(request, f"You are now logged in as {username}.")
				return redirect('index/')
			else:
				messages.error(request,"Invalid username or password.")
		else:
			messages.error(request,"Invalid username or password.")
	form = AuthenticationForm()
	return render(request=request, template_name="login.html", context={"login_form":form})

def logout_request(request):
	logout(request)
	messages.info(request, "You have successfully logged out.") 
	return redirect('login1')