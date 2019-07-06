from flask import Flask, request, render_template
from . import app
import os

import pandas as pd
import numpy as np
import pickle
from gensim.models import Word2Vec
from sklearn import preprocessing
import json

#=================================MODELS=================================
with open('flaskexample/models/skills2vec_10D_world.pickle', 'rb') as file:
    skill2vec = pickle.load(file)
    SKILLS_LIST = list(skill2vec.wv.vocab)
    SIZE = 10


with open('flaskexample/models/xgb_world.pickle', 'rb') as file:
    regressor = pickle.load(file)

with open('flaskexample/data/features.pickle', 'rb') as file:
    features = pickle.load(file)

#=================================DICTS=================================
with open('flaskexample/data/median_rate_per_country.pickle', 'rb') as file:
    median_rate_per_country = pickle.load(file)
    COUNTRIES_LIST = list(median_rate_per_country.keys())


#=================================FUNCTIONS=================================
def get_skillset_vector(skillset, normalize=True):
    skillset_vector = []
    for skill in skillset:
        if skill not in skill2vec.wv:
            print('WARNING {} is not in vocab'.format(skill))
            continue

        skill_vector = skill2vec.wv[skill]
        skillset_vector.append(skill_vector)
    if not skillset_vector:
        values = None
    else:
        values = np.mean(skillset_vector, axis=0)
        if normalize:
            values = preprocessing.normalize(values.reshape(1, -1))[0]

    result = pd.Series(data=values, index=[
                       'skill_coord_{}'.format(i) for i in range(SIZE)])
    return result


def get_prediction(skills_selected, country_selected):
    country_feature = pd.Series(
        {'country': median_rate_per_country[country_selected]})

    predictions = []
    for skill in skills_selected:
        skillset_vector = get_skillset_vector([skill], skill2vec)

        entry = pd.Series()
        entry = entry.append(skillset_vector)
        entry = entry.append(country_feature)
        entry = entry.loc[features]

        prediction = regressor.predict(entry.to_frame().T)[0]
        predictions.append(prediction)
    prediction = max(predictions)
    return int(round(prediction))

def get_pred_and_recc(skills_selected, country_selected):
    prediction = get_prediction(skills_selected, country_selected)

    similar_skills = pd.Series(
        {skills[0]: skills[1] for skills in skill2vec.most_similar(skills_selected, topn=50)})
    similar_skills = similar_skills.index

    reccs = pd.Series({skill: get_prediction(
        skills_selected+[skill], country_selected) - prediction for skill in similar_skills})
    reccs = reccs.sort_values(ascending=False)
    skills_recommended = list(reccs[:3].index)

    recommendation_value = 'Consider learning {}, {} and {} to increase your value.'.format(
        *skills_recommended)

    skills_recommended = [tup[0]
                          for tup in skill2vec.most_similar(skills_selected, topn=3)]
    recommendation_proximity = 'People with your skills are also familiar with {}, {} and {}.'.format(
        *skills_recommended)

    return prediction, recommendation_value, recommendation_proximity

#=================================FLASK=================================
@app.route('/')
@app.route('/index')
@app.route('/output', methods=['POST'])
def index():
    return render_template("index.html", 
                           countries=json.dumps(COUNTRIES_LIST),
                           skills=json.dumps(SKILLS_LIST))


@app.route('/', methods=['POST'])
def my_form_post():
   form = request.form
   print(form)
   country_selected = form['country']
   skills_selected = form['skills'].split(',')
   print(skills_selected)
   
   prediction, recommendation_value, recommendation_proximity = get_pred_and_recc(
       skills_selected, country_selected)
   print(prediction, recommendation_value, recommendation_proximity)

   return render_template("output.html", hourly_rate=prediction, 
                                         recommendation_value=recommendation_value,
                                         recommendation_proximity=recommendation_proximity)


   


