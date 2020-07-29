from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import joblib
import numpy as np
import pandas as pd
import re
from PIL import Image
from konlpy.tag import Okt
from tensorflow import keras
from keras.models import load_model
from keras.applications.vgg16 import VGG16, decode_predictions
from clu_util import cluster_util

app = Flask(__name__)
app.debug = True

vgg = VGG16()
okt = Okt()
movie_lr = None
movie_lr_dtm = None
def load_movie_lr():
    global movie_lr, movie_lr_dtm
    movie_lr = joblib.load(os.path.join(app.root_path, 'model/movie_lr.pkl'))
    movie_lr_dtm = joblib.load(os.path.join(app.root_path, 'model/movie_lr_dtm.pkl'))

def tw_tokenizer(text):
    # 입력 인자로 들어온 text 를 형태소 단어로 토큰화 하여 list 객체 반환
    tokens_ko = okt.morphs(text)
    return tokens_ko

movie_nb = None
movie_nb_dtm = None
def load_movie_nb():
    global movie_nb, movie_nb_dtm
    movie_nb = joblib.load(os.path.join(app.root_path, 'model/movie_nb.pkl'))
    movie_nb_dtm = joblib.load(os.path.join(app.root_path, 'model/movie_nb_dtm.pkl'))

def nb_transform(review):
    stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    review = review.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    morphs = okt.morphs(review, stem=True)
    temp = ' '.join(morph for morph in morphs if not morph in stopwords)
    return temp



model_iris_lr = None
model_iris_svm = None
model_iris_dt = None
model_iris_deep = None
def load_iris():
    global model_iris_lr, model_iris_svm, model_iris_dt, model_iris_deep
    model_iris_lr = joblib.load(os.path.join(app.root_path, 'model/iris_lr.pkl'))
    model_iris_svm = joblib.load(os.path.join(app.root_path, 'model/iris_svm.pkl'))
    model_iris_dt = joblib.load(os.path.join(app.root_path, 'model/iris_dt.pkl'))
    model_iris_deep = load_model(os.path.join(app.root_path, 'model/iris_deep.hdf5'))

model_blood_lr = None
model_blood_svm = None
model_blood_dt = None
model_blood_deep = None
def load_blood():
    global model_blood_lr, model_blood_svm, model_blood_dt, model_blood_deep
    model_blood_lr = joblib.load(os.path.join(app.root_path, 'model/blood_lr.pkl'))
    model_blood_svm = joblib.load(os.path.join(app.root_path, 'model/blood_svm.pkl'))
    model_blood_dt = joblib.load(os.path.join(app.root_path, 'model/blood_dt.pkl'))
    model_blood_deep = load_model(os.path.join(app.root_path, 'model/blood_deep.hdf5'))

mes_nb = None
mes_nb_dtm = None
def load_mes_nb():
    global mes_nb, mes_nb_dtm
    mes_nb = joblib.load(os.path.join(app.root_path, 'model/spam_nb.pkl'))
    mes_nb_dtm = joblib.load(os.path.join(app.root_path, 'model/spam_nb_dtm.pkl'))


@app.route('/')
def index():
    menu = {'home':True, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':False, 'user':False}
    return render_template('home.html', menu=menu)

@app.route('/regression', methods=['GET', 'POST'])
def regression():
    menu = {'home':False, 'rgrs':True, 'stmt':False, 'clsf':False, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('regression.html', menu=menu)
    else:
        sp_names = ['Setosa', 'Versicolor', 'Virginica']
        slen = float(request.form['slen'])      # Sepal Length
        plen = float(request.form['plen'])      # Petal Length
        pwid = float(request.form['pwid'])      # Petal Width
        sp = int(request.form['species'])       # Species
        species = sp_names[sp]
        swid = 0.63711424 * slen - 0.53485016 * plen + 0.55807355 * pwid - 0.12647156 * sp + 0.78264901
        swid = round(swid, 4)
        iris = {'slen':slen, 'swid':swid, 'plen':plen, 'pwid':pwid, 'species':species}
        return render_template('reg_result.html', menu=menu, iris=iris)

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    menu = {'home':False, 'rgrs':False, 'stmt':True, 'clsf':False, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('sentiment.html', menu=menu)
    else:
        res_str = ['부정', '긍정']
        review = request.form['review']
        # Logistic Regression 처리
        review_lr = re.sub(r"\d+", " ", review)
        review_lr_dtm = movie_lr_dtm.transform([review_lr])
        result_lr = res_str[movie_lr.predict(review_lr_dtm)[0]]
        # Naive Bayes 처리
        review_nb = nb_transform(review)
        review_nb_dtm = movie_nb_dtm.transform([review_nb])
        result_nb = res_str[movie_nb.predict(review_nb_dtm)[0]]
        # 결과 처리
        movie = {'review':review, 'result_lr':result_lr, 'result_nb':result_nb}
        return render_template('senti_result.html', menu=menu, movie=movie)

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':True, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('classification.html', menu=menu)
    else:
        f = request.files['image']
        filename = os.path.join(app.root_path, 'static/images/uploads/') + \
                    secure_filename(f.filename)
        f.save(filename)
        img = np.array(Image.open(filename).resize((224, 224)))
        yhat = vgg.predict(img.reshape(-1, 224, 224, 3))
        label_key = np.argmax(yhat)
        label = decode_predictions(yhat)
        label = label[0][0]
        return render_template('cla_result.html', menu=menu,
                                filename = secure_filename(f.filename),
                                name=label[1], pct='%.2f' % (label[2]*100))

@app.route('/classification_iris', methods=['GET', 'POST'])
def classification_iris():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':True, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('classification_iris.html', menu=menu)
    else:
        sp_names = ['Setosa', 'Versicolor', 'Virginica']
        slen = float(request.form['slen'])      # Sepal Length
        swid = float(request.form['swid'])      # Sepal Width
        plen = float(request.form['plen'])      # Petal Length
        pwid = float(request.form['pwid'])      # Petal Width
        test_data = np.array([slen, swid, plen, pwid]).reshape(1,4)
        species_lr = sp_names[model_iris_lr.predict(test_data)[0]]
        species_svm = sp_names[model_iris_svm.predict(test_data)[0]]
        species_dt = sp_names[model_iris_dt.predict(test_data)[0]]
        species_deep = sp_names[model_iris_deep.predict_classes(test_data)[0]]
        iris = {'slen':slen, 'swid':swid, 'plen':plen, 'pwid':pwid, 
                'species_lr':species_lr, 'species_svm':species_svm,
                'species_dt':species_dt, 'species_deep':species_deep}
        return render_template('cla_iris_result.html', menu=menu, iris=iris)

@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':True, 'user':False}
    if request.method == 'GET':
        return render_template('clustering.html', menu=menu)
    else:
        f = request.files['csv']
        filename = os.path.join(app.root_path, 'static/images/uploads/') + \
                    secure_filename(f.filename)
        f.save(filename)
        ncls = int(request.form['K'])
        cluster_util(app, ncls, secure_filename(f.filename))
        img_file = os.path.join(app.root_path, 'static/images/kmc.png')
        mtime = int(os.stat(img_file).st_mtime)
        return render_template('clu_result.html', menu=menu, K=ncls, mtime=mtime)

@app.route('/spam', methods=['GET', 'POST'])
def spam():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':False, 'mes':True, 'user':False}
    if request.method == 'GET':
        return render_template('spam.html', menu=menu)
    else:
        res_str = ['스팸문자가 아닙니다', '스팸문자 입니다']
        mes = request.form['mes']
        # Naive Bayes 처리

        spam_nb_dtm = mes_nb_dtm.transform([mes])
        result = res_str[mes_nb.predict(spam_nb_dtm)[0]]
        # 결과 처리
        message = {'mes':mes, 'result':result}
        return render_template('spam_result.html', menu=menu, message=message)

@app.route('/blood', methods=['GET', 'POST'])
def classification_disease():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':True, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('blood.html', menu=menu)
    else:
        sp_names = ['X','O']
             
        age = int(request.form['age'])+1     
        hgb = float(request.form['hgb'])      
        tchol = int(request.form['tchol'])     
        tg = int(request.form['tg'])     
        hdl = int(request.form['hdl'])      
               
        test_data = np.array([age, hgb,tchol,tg,hdl]).reshape(1,5)
        blood_lr = sp_names[model_blood_lr.predict(test_data)[0]]
        blood_svm = sp_names[model_blood_svm.predict(test_data)[0]]
        blood_dt = sp_names[model_blood_dt.predict(test_data)[0]]
        blood_deep = sp_names[model_blood_deep.predict_classes(test_data)[0][0]]

        blood = { 'age':age, 'hgb':hgb, 'tchol':tchol,
                'tg':tg, 'hdl':hdl,
                'blood_lr':blood_lr, 'blood_svm':blood_svm,
                'blood_dt':blood_dt, 'blood_deep':blood_deep}
        return render_template('blood_result.html', menu=menu, blood=blood)



@app.route('/member/<name>')
def member(name):
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':False, 'user':True}
    nickname = request.args.get('nickname', '별명: 없음')
    return render_template('user.html', menu=menu, name=name, nickname=nickname)

if __name__ == '__main__':
    load_movie_lr()
    load_movie_nb()
    load_mes_nb()
    load_blood()
    load_iris()
    app.run()  # 외부 접속 허용시 host='0.0.0.0' 추가