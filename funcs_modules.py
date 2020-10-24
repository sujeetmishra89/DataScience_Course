import os, sys
import platform, multiprocessing
import shutil, subprocess
from io import StringIO
import gc, warnings, glob
warnings.filterwarnings('ignore')
from plyer import notification
# from win10toast import ToastNotifier
# import winsound
# import pkgutil
print("Loaded System Packages")

# dir()
# get_ipython().run_line_magic('reset', '-f') # %reset -f
# get_ipython().run_line_magic('matplotlib', 'inline') # %matplotlib inline
# get_ipython().run_line_magic('history', '-n') # %history -n
# get_ipython().run_line_magic('load_ext', 'autoreload') # %load_ext autoreload
# get_ipython().run_line_magic('autoreload', '2') # %autoreload 2
# from IPython.display import Image, SVG, HTML, display, clear_output
# import ipykernel
# import ipyparallel as ipp
# import ipywidgets as widgets
# from ipywidgets import interact, interact_manual
# from tqdm.keras import TqdmCallback
# from tqdm._tqdm_notebook import tqdm_notebook
# tqdm_notebook.pandas()
print("Loaded Jupyter-Notebook Packages")

import pandas as pd
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 100
pd.options.display.max_rows = 1000
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.float_format', lambda x: '%.3f' % x)
import numpy as np
import math
import random
import re, regex, xlrd, string
import json, pyEX, zipfile
import pickle, joblib
import itertools as it
import more_itertools as mit
from collections import Counter, defaultdict
# import patsy
# import researchpy as rp
# from dask import dataframe as dd
# from pprint import pprint
# from pyxlsb import open_workbook as open_xlsb
# from pandarallel import pandarallel
# from array import *
# from operator import itemgetter
# from numba import njit
# import unittest
print("Loaded Dataframe Packages")

from datetime import datetime, date, time, timedelta
from dateutil.parser import parse as dateparser
from pytz import timezone
import time
print("Loaded Datetime Packages")

import statsmodels.api as sm
# from statsmodels.formula.api import ols
# from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost
from xgboost import XGBClassifier, XGBRegressor
# from lmfit import Model
# import networkx 
# from networkx.algorithms.components.connected import connected_components
from yellowbrick.cluster import KElbowVisualizer
from fastcluster import linkage # You can use SciPy one too
from scipy.stats import pearsonr, spearmanr, rankdata, norm, gaussian_kde, kstest
from scipy.stats.mstats import gmean
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, leaves_list
from scipy.linalg import cholesky, inv
from scipy.spatial import distance
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.signal import argrelextrema
print("Loaded Statistical Packages")

import sklearn
from sklearn import tree, linear_model
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, KFold, StratifiedKFold, GridSearchCV, StratifiedShuffleSplit, RandomizedSearchCV
# from sklearn.externals import joblib
# from sklearn.decomposition import PCA 
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.preprocessing import scale, normalize, StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, ExtraTreesClassifier, RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_curve, auc, roc_auc_score, precision_recall_curve, plot_precision_recall_curve
# from sklearn.metrics import silhouette_samples, silhouette_score, homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score
# from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline, Pipeline
print("Loaded Scikit-learn Packages")

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr # Stops printing "Using Tensorflow Backend"
import keras.backend as K
from keras import utils, optimizers
from keras.models import Sequential, Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.utils import np_utils
# from keras.wrappers.scikit_learn import KerasRegressor
# from keras.preprocessing.text import one_hot, Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.layers import LSTM, Input, Masking, Bidirectional, TimeDistributed, GlobalMaxPooling1D
# from keras.layers.normalization import BatchNormalization
# from keras.layers.core import Dense, Activation, Dropout, Dense
# from keras.layers.embeddings import Embedding
# from keras.layers.merge import Concatenate
# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras.models import model_from_json
print("Loaded Keras-Tensorflow Packages")

import sqlite3
import traceback
# import cx_Oracle
# import pandasql as pdsql
# from sqlalchemy import create_engine, text
# from sqlalchemy.pool import Pool, NullPool, QueuePool
# from sqlalchemy.types import String, Integer, Float, Date, DateTime
print("Loaded SQl Packages")

# import nltk 
# from nltk import FreqDist, ProbDistI
# from nltk.util import ngrams
# from nltk.corpus import stopwords 
# from nltk.tokenize import word_tokenize 
print("Loaded NLP Packages")

# from deap import base, creator, tools, algorithms
print("Loaded Deep Learning Packages")

# import requests
# import getpass
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from email.mime.base import MIMEBase
# from email import encoders
# import urllib.request, urllib.error, urllib.parse
# import http.cookiejar
# from selenium import webdriver
# from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
# from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
print("Loaded Web Packages")

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.pyplot import plot
from matplotlib.widgets import Button
from matplotlib.text import Annotation
# plt.style.use('seaborn')
# plt.ioff()
import seaborn as sns
# import tkinter as tk
# import pandas_profiling as pf
# from mayavi import mlab
# from mpl_toolkits.mplot3d import Axes3D 
# import pylab as pl
# import pydot, pydotplus
# import chart_studio.plotly as py
# import plotly
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots
# from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
# # init_notebook_mode(connected=True)
# import cufflinks as cf
# # cf.go_offline()
# # cf.set_config_file(offline=False, world_readable=True)
# from graphviz import Source
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
print("Loaded Visualisation Packages")

# with open("..\\..\\Personal\\config.json") as conf: config = json.load(conf)
# mail_frm = config["environment"][0]["mail_frm"]
# mail_pwd = config["environment"][0]["mail_pwd"]
# mail_to = config["environment"][0]["mail_to"]
# auth = config["environment"][0]["auth"]
# sms_frm = config["environment"][0]["sms_frm"]
# sms_pwd = config["environment"][0]["sms_pwd"]
# sms_to = config["environment"][0]["sms_to"]
# mail_frm = input("Gmail ID: ")
# mail_pwd = getpass.getpass("Password: ")
# mail_to = input("Send Mail to: ")
# sms_frm = input("Way2SMS Username: ")
# sms_pwd = getpass.getpass("Password: ")
# sms_to = input("Send SMS to: ")

def clear_output(): os.system('cls')

def logs_time(): return(datetime.now(timezone('UTC')).astimezone(timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

def message_log(message, outputfile): open(outputfile, "a").write("[{}] {}\n".format(logs_time(),message))

def input_yes_no(question, val, default="yes"):
	# https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
	valid = ["yes","y","ye"]
	invalid = ["no","n"]

	if default is None: prompt = " [y/n] "
	elif default == "yes": prompt = " [Y/n] "
	elif default == "no": prompt = " [y/N] "
	else: raise ValueError("invalid default answer: '%s'" % default)

	while True:
		sys.stdout.write(question + prompt)
		choice = input().lower()
		if default is not None and choice == '': return val
		elif choice in valid: return val
		elif choice in invalid: return input("Enter Correct Value : ")
		else: sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")

def send_mail(alertmessage, outputfile):
	msg = MIMEMultipart()
	msg['From'] = mail_frm
	msg['To'] = mail_to
	msg['Subject'] = "[script_alert] "+filename

	body = "["+logs_time()+"] "+alertmessage
	msg.attach(MIMEText(body, 'plain', 'utf-8'))
	part = MIMEBase('application', 'octet-stream')
	part.set_payload(open(outputfile, "rb").read())
	encoders.encode_base64(part)
	part.add_header('Content-Disposition', 'attachment', filename = filename)
	msg.attach(part)

	# Go to (https://myaccount.google.com/lesssecureapps) & switch ON less secure apps
	server = smtplib.SMTP_SSL('smtp.gmail.com')
	server.login(mail_frm,pwd)
	server.sendmail(msg['From'], msg['To'], msg.as_string())
	server.quit()
	message_log("Mail has been Sent", outputfile)

	# # https://stackoverflow.com/questions/6332577/send-outlook-email-via-python
	# import win32com.client as win32
	# outlook = win32.Dispatch('outlook.application')
	# mail = outlook.CreateItem(0)
	# mail.To = mail_to
	# mail.Subject = "[script_alert] "+filename
	# mail.Body = "["+logs_time()+"] "+alertmessage
	# mail.HTMLBody = '<h2>Thanks.</h2>'
	# mail.Attachments.Add(os.path.join(outputfile,filename))
	# mail.Display()
	# mail.Send()

def send_sms(approach, message):
	if approach=='requests':
		url = "https://www.fast2sms.com/dev/bulk"
		payload = "sender_id=FSTSMS&message={}&language=english&route=p&numbers={}".format(message,sms_to)
		headers = {'authorization': auth,
				   'Content-Type': "application/x-www-form-urlencoded",
				   'Cache-Control': "no-cache",}
		response = requests.request("POST", url, data=payload, headers=headers)

		# login_data = {'mobileNo': sms_frm, 'password': sms_pwd}
		# post_data = {'toMobile': sms_to, 'message': message}
		# login_response = requests.get('https://www.way2sms.com/', data=login_data)
		# form_response = requests.post('https://www.way2sms.com/send-sms', data=post_data, cookies=login_response.cookies)

	if approach=='urlib':
		message = "+".join(message.split(' '))
		url = 'https://www.way2sms.com/'
		data = 'mobileNo='+sms_frm+'&password='+sms_pwd+'&CatType' #mobile number & password for way2sms
		# data = urllib.parse.urlencode(data).encode("utf-8")

		# # For Cookies:
		cj = http.cookiejar.CookieJar()
		opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

		# # Adding Header detail:
		# opener.addheaders = [('User-Agent','Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.120 Safari/537.36')]
		opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36')]

		# urllib.request.Request('https://www.way2sms.com/send-sms?mobileNo={}&password={}'.format(sms_frm,sms_pwd))
		try: socket = opener.open(url,data.encode('utf-8'))
		except Exception as e:
			print("cannot connect due to",e)
			sys.exit(1)

		session_id = str(cj).split('~')[len(cj)].split(' ')[0]
		# session_id = str(cj).split('~')[1].split(' ')[0]
		# smsurl = 'http://www.way2sms.com/smstoss'
		smsurl = 'https://www.way2sms.com/send-sms'
		# smsurl ='http://site21.way2sms.com/smstoss.action'
		data = 'ssaction=ss&Token='+session_id+'&toMobile='+sms_to+'&message='+message #+'&msgLen='+str(140-len(message))
		opener.addheaders = [('Referrer', smsurl+session_id)]
		# opener.addheaders = [('Referer','http://site21.way2sms.com/sendSms?Token='+session_id)]

		# req = urllib.request.Request(url, data=urllib.parse.urlencode(d))
		# resp = urllib.request.urlopen(req).read()
		# with urllib.request.urlopen(req,data=data) as f: resp = f.read()
		# urllib.request.Request(smsurl, data=urllib.parse.urlencode(data).encode('utf-8'))
		# urllib.request.urlopen(urllib.request.Request(url+'?'+data.encode('utf-8'), headers=headers))
		# p = opener.open('https://www.way2sms.com/send-sms?mobileNo={}&password={}&toMobile={}&message={}'.format(sms_frm,sms_pwd,sms_to,message))
		# p = opener.open('http://site21.way2sms.com/smscofirm.action?SentMessage='+message+'&Token='+session_id+'&status=0')
		# p = opener.open("http://site21.way2sms.com/ebrdg.action?id="+session_id)
		print('http://www.way2sms.com/send-sms?'+'mobileNo='+sms_frm+'&password='+sms_pwd+'&CatType'+'&Token='+session_id+'&toMobile='+sms_to+'&message='+message)
		try: page = opener.open(url,data.encode('utf-8'))
		except Exception as e: print('page:',e)

	if approach=='selenium':
		# cap = DesiredCapabilities().FIREFOX
		# cap["marionette"] = False
		# binary = FirefoxBinary('/path/to/binary')
		# driver = webdriver.Firefox(capabilities=cap, firefox_binary=binary, executable_path='..\\..\\Tools\\geckodriver-v0.26.0-win64\\geckodriver.exe')
		# driver = webdriver.PhantomJS(executable_path='..\\..\\Tools\\phantomjs-2.1.1-windows\\bin\\phantomjs.exe')
		# driver = webdriver.Edge(executable_path='..\\..\\Tools\\edgedriver_win64\\msedgedriver.exe')
		driver.set_window_size(1120, 550)
		driver.get("https://www.way2sms.com/")
		driver.find_element_by_id("mobileNo").send_keys(sms_frm)
		driver.find_element_by_id("password").send_keys(sms_pwd)
		button = driver.find_element_by_xpath("//*[contains(text(), 'Login')]")
		print(button.get_attribute('href'))
		print(len(driver.page_source))
		button.click()
		print(button.get_attribute('href'))
		print(len(driver.page_source))
		driver.get("https://www.way2sms.com/send-sms")
		window_after = driver.window_handles[1]
		driver.switch_to.window(window_after)
		driver.find_element_by_id("mobile").send_keys(sms_to)
		driver.find_element_by_id("message").send_keys(message)
		driver.find_element_by_id("sendButton").click()
		button = driver.find_element_by_id('sendButton')
		button.click()
		print(button.get_attribute('href'))
		print(len(driver.page_source))
		print(driver.window_handles)

def rmse(y_true, y_pred): return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def mse(y_true, y_pred): return K.mean(K.square(y_pred - y_true), axis=-1)

def mqe(y_true, y_pred): return K.mean(K.pow((y_pred - y_true), 4), axis=-1)

def r_square(y_true, y_pred):
	SS_res =  K.sum(K.square(y_true - y_pred))
	SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
	return (1 - SS_res/(SS_tot + K.epsilon()))

def r_square_loss(y_true, y_pred):
	SS_res =  K.sum(K.square(y_true - y_pred))
	SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
	return 1 - (1 - SS_res/(SS_tot + K.epsilon()))

def mean_absolute_percentage_error(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_vs_validation(result):
	plt.figure(figsize=(50,30))

	# plt.subplot(331)
	# # plt.plot(result.history['acc'])
	# # plt.plot(result.history['val_acc'])
	# plt.plot(result.history['accuracy'])
	# plt.plot(result.history['val_accuracy'])
	# plt.title('Model accuracy')
	# plt.ylabel('Accuracy')
	# plt.xlabel('Epoch')
	# plt.legend(['Train', 'Validation'], loc='upper right')
	# # plt.savefig(os.path.join(folder,"accuracy_"+current_time+".png"))
	# # plt.show()

	plt.subplot(331)
	plt.plot(result.history['loss'])
	plt.plot(result.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train','Validation'], loc='upper right')
	# plt.savefig(os.path.join(folder,"loss_"+current_time+".png"))
	# plt.show()

	# plt.subplot(333)
	# plt.plot(result.history['mean_squared_error'])
	# plt.plot(result.history['val_mean_squared_error'])
	# plt.title('Model loss')
	# plt.ylabel('mean_squared_error')
	# plt.xlabel('Epoch')
	# plt.legend(['Train','Validation'], loc='upper right')
	# # plt.savefig(os.path.join(folder,"msePlot_"+current_time+".png"))
	# plt.show()

	plt.subplot(332)
	plt.plot(result.history['rmse'])
	plt.plot(result.history['val_rmse'])
	plt.title('Model rmse')
	plt.ylabel('rmse')
	plt.xlabel('Epoch')
	plt.legend(['Train','Validation'], loc='upper right')
	# plt.savefig(os.path.join(folder,"rmsePlot_"+current_time+".png"))
	# plt.show()

	plt.subplot(333)
	plt.plot(result.history['r_square'])
	plt.plot(result.history['val_r_square'])
	plt.title('Model R^2')
	plt.ylabel('R^2')
	plt.xlabel('Epoch')
	plt.legend(['Train','Validation'], loc='upper right')
	# plt.savefig(os.path.join(folder,"r2Plot_"+current_time+".png"))
	# plt.show()

	plt.subplot(334)
	plt.plot(result.history['mean_absolute_error'])
	plt.plot(result.history['val_mean_absolute_error'])
	plt.title('Model mean_absolute_error')
	plt.ylabel('mean_absolute_error')
	plt.xlabel('Epoch')
	plt.legend(['Train','Validation'], loc='upper right')
	# plt.savefig(os.path.join(folder,"maePlot_"+current_time+".png"))
	# plt.show()

	plt.savefig(os.path.join(folder,"Training_vs_Validation_Plots_"+current_time+".png"))
	plt.show()

def performance_metrics(true, pred):
	# print(len(true), len(pred))
	# print(sc.describe(true))
	# print(sc.describe(pred))
	print("Target Standard Deviation (std):   %f" % true.std())
	print("Coefficient of determination (R^2):%f" % r2_score(true,pred))
	print("Mean absolute error (MAE):         %f" % mean_absolute_error(true,pred))
	print("Mean squared error (MSE):          %f" % mean_squared_error(true,pred))
	print("Root mean squared error (RMSE):    %f" % math.sqrt(mean_squared_error(true,pred)))
	# print("Root mean squared error (RMSE):  %f" % np.sqrt(mean_squared_error(true,pred)))
	# print("Mean absolute Percentage error (MAPE): %f" % tf.keras.losses.MAPE(true,pred))
	# print("Mean absolute Percentage error (MAPE): %f" % (np.mean(np.abs((true - pred) / true)) * 100))
	print("Mean absolute Percentage error (MAPE): %f" % mean_absolute_percentage_error(list(true),pd.DataFrame(pred)[0]))
	# plt.hist(true)
	# plt.hist(pred)

def scatter_plots(true, pred, scatter_plot_name):
	# nx,ny = len(true),len(pred)
	# cx,cy = np.random.rand(nx),np.random.rand(ny)
	# plt.scatter(list(true),pred.flatten().tolist(),c=cx,alpha=0.6)
	# plt.savefig(os.path.join(folder,scatter_plot_name+current_time+".png"))
	# plt.title(scatter_plot_name)
	# plt.show()
	sns_plot = sns.scatterplot(list(true),list(pred))
	fig = sns_plot.get_figure()
	fig.savefig(os.path.join(folder,scatter_plot_name+current_time+".png"))
	fig.show()

def actual_vs_predict(df):
	for tap in df['tappotKey'].unique():
		# i = test_taps.index(tap)
		df1 = df[df['tappotKey']==tap].reset_index(drop=True)
		X, y, y_temp = X_Y_dataset(df1)
		X_ohe = ohe.transform(X.iloc[:,:2])
		X = pd.DataFrame(X_ohe.toarray(), columns=col).join(X.iloc[:,2:])
		X = scaler.transform(X)
		preds = model.predict(X)

		plt.figure(figsize=(50,25))
		plt.plot(list(y), label = "actual")
		plt.plot(preds, label = "pred")
		plt.title("Tap Seq : "+tap)
		plt.xlabel(tap)
		plt.legend()
		plt.savefig(os.path.join(folder,tap+"_"+current_time+".png"))
		# plt.show()
		# plt.clf()

def pred_importance(df,pred_list,op_folder):
	output_result = pd.DataFrame([])
	for i in pred_list:
		df1 = df.copy()
		df1.loc[:,i]=0
		X, y, y_temp = X_Y_dataset(df1)
		X_ohe = ohe.transform(X.iloc[:,:2])
		X = pd.DataFrame(X_ohe.toarray(), columns=col).join(X.iloc[:,2:])
		X = scaler.transform(X)
		preds = model.predict(X)
		A = pd.DataFrame([])
		A.loc[i,0] = math.sqrt(mean_squared_error(y,preds))
		output_result = pd.concat([output_result,A], axis=0)
		print(i)
		clear_output(wait=True)
	output_result = output_result.reset_index().rename(columns={'index':'Features',0:'rmse'}).sort_values('rmse', ascending=False).reset_index(drop=True)
	output_result.to_csv(os.path.join(folder,target+"_"+op_folder+"_"+current_time+".csv"), index=False)
	return output_result
