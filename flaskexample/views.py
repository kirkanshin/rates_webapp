from flask import Flask, request, render_template
from . import app
import os

import pandas as pd
import numpy as np
import pickle
from gensim.models import Word2Vec
import json

#=================================MODELS=================================
with open('flaskexample/models/skills2vec_10D_world.pickle', 'rb') as file:
    skill2vec = pickle.load(file)

with open('flaskexample/models/xgb_world.pickle', 'rb') as file:
    regressor = pickle.load(file)

with open('flaskexample/data/features.pickle', 'rb') as file:
    features = pickle.load(file)

with open('flaskexample/data/MEAN_SKILL.pickle', 'rb') as file:
    MEAN_SKILL = pickle.load(file)

#=================================DICTS=================================
with open('flaskexample/data/median_rate_per_country.pickle', 'rb') as file:
    median_rate_per_country = pickle.load(file)
    countries_list = list(median_rate_per_country.keys())

with open('flaskexample/data/skills_human_to_model.pickle', 'rb') as file:
    skills_human_to_model = pickle.load(file)
    skills_model_to_human = {v:k for k,v in skills_human_to_model.items()}
    skills_list_human = list(skills_human_to_model.keys())

#=================================FUNCTIONS=================================
def get_skillset_vector(skillset, skill2vec, norm=True):
    skillset_vector = []
    for skill in skillset:
        if skill not in skill2vec.wv:
            print('skill is not in vocab', skill)
            raise KeyError
            continue

        skill_vector = skill2vec.wv[skill]
        skillset_vector.append(skill_vector)

    if not skillset_vector:
        values = MEAN_SKILL
    else:
        skillset_vector = np.array(skillset_vector)
        values = np.sum(skillset_vector, axis=0)

    valnorm = np.linalg.norm(values)
    if norm:
        values = values / np.linalg.norm(values)

    result = pd.Series(data=values, index=[
                       'skill_coord_{}'.format(i) for i in range(len(MEAN_SKILL))])
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

def get_recommendation(skills_selected):
    skills_reccomended = [skills_model_to_human[tup[0]]
                        for tup in skill2vec.most_similar(skills_selected, topn=3)]
    recommendation = 'Consider learning {}, {} and {} to improve your value.'.format(
        *skills_reccomended)
    return recommendation

#=================================FLASK=================================
@app.route('/')
@app.route('/index')
@app.route('/output', methods=['POST'])
def index():
    #pass the list of countries
    countries_list
    skills_list_human
    #list of skills in human form
    return render_template("index.html", 
                            countries=json.dumps(countries_list),
                            skills=json.dumps(skills_list_human))


@app.route('/', methods=['POST'])
def my_form_post():
   form = request.form
   print(form)
   state_selected = form['country']
   skills_selected = [skills_human_to_model[skill]
                      for skill in form['skills'].split(',')]
   print(skills_selected)
   prediction = get_prediction(skills_selected, state_selected)
   recommendation = get_recommendation(skills_selected)

   return render_template("output.html", hourly_rate=prediction, recommendation=recommendation)


   


