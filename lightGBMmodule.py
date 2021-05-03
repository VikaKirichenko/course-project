import pandas as pd
import lightgbm as lgb
import os, json

def func(item):
    alphabet = {"а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м", "н", "о",
                "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я"}

    X_test = pd.DataFrame(columns=['age', 'skills', 'city', 'gender', 'specialty', 'java', 'maven', 'git', 'sql', 'hibernate', 'spring', 'oop',
                                   'student', 'education_level', 'total_experience', 'skill_set', 'rubbish'])
    java = False
    maven = False
    git = False
    english = False
    sql = False
    hibernate = False
    spring = False
    oop = False
    rubbish = False
    if item['skills'] is not None:
        skills = True
    else:
        skills = False

    skill_set = item['skill_set']
    rubbishcount = 0
    d = {}
    for item_in in skill_set:
        if bool(alphabet.intersection(
                set(item_in.lower()))) and not "английский язык" == item_in.lower() \
                and not "ооп" == item_in.lower():
            rubbish = True
            rubbishcount += 1
            continue
        tl = item['title'].lower()
        if 'java' in tl and 'стажер' in tl:
            specialty = 'javaintern'
        elif 'junior' in tl and 'java' in tl:
            specialty = 'javajun'
        elif 'java' in tl:
            specialty = 'javadeveloper'
        elif 'программист' in tl or 'разработчик' in tl:
            specialty = 'developer'
        else:
            specialty = 'other'
        if item_in.lower() in d:
            d[item_in.lower()] += 1
        else:
            d[item_in.lower()] = 1

        if "java" in item_in.lower() and "javascript" not in item_in.lower() and not java:
            java = True
            continue
        if "maven" in item_in.lower() and not maven:
            maven = True
            continue
        if "git" in item_in.lower() and not git:
            git = True
            continue
        if ("english" in item_in.lower() or "aнглийский язык" in item_in.lower()) and not english:
            english = True
            continue
        if "sql" in item_in.lower() and not sql:
            sql = True
            continue
        if "hibernate" in item_in.lower() and not hibernate:
            hibernate = True
            continue
        if "spring" in item_in.lower() and not spring:
            spring = True
            continue
        if "ооп" in item_in.lower() and not oop:
            oop = True
            continue
    skill_set = ",".join(skill_set)
    if item['age'] is not None:
        age = item['age']
    else:
        age = 0
    city = item['area']['name']
    gender = item['gender']['id']
    student = False
    array_of_educations = item['education']['primary']
    for x in array_of_educations:
        if x['year'] > 2020:
            student = True
            break
    if item['education']['level']['id'] is not None:
        education_level = item['education']['level']['id']
    else:
        education_level = "none"
    if item['total_experience'] is not None:
        total_experience = item['total_experience']['months']
    else:
        total_experience = 0

    X_test.loc[0] = [age, skills, city, gender, specialty, java, maven, git, sql, hibernate, spring, oop, student,
                     education_level, total_experience, skill_set, rubbishcount]
    #city,gender,specialty,
    X_test['city'] = pd.Series(X_test['city'], dtype="category")
    X_test['gender'] = pd.Series(X_test['gender'], dtype="category")
    X_test['specialty'] = pd.Series(X_test['specialty'], dtype="category")
    X_test['age'] = X_test['age'].astype(str).astype(int)
    X_test['skills'] = X_test['skills'].astype(str).astype(bool)
    X_test['java'] = X_test['java'].astype(str).astype(bool)
    X_test['maven'] = X_test['maven'].astype(str).astype(bool)
    X_test['git'] = X_test['git'].astype(str).astype(bool)
    X_test['sql'] = X_test['sql'].astype(str).astype(bool)
    X_test['hibernate'] = X_test['hibernate'].astype(str).astype(bool)
    X_test['spring'] = X_test['spring'].astype(str).astype(bool)
    X_test['oop'] = X_test['oop'].astype(str).astype(bool)
    X_test['student'] = X_test['student'].astype(str).astype(bool)
    X_test['total_experience'] = X_test['total_experience'].astype(str).astype(int)
    X_test['education_level'] = pd.Series(X_test['education_level'], dtype="category")
    X_test['skill_set'] = pd.Series(X_test['skill_set'], dtype="category")
    X_test['rubbish'] = X_test['rubbish'].astype(str).astype(int)
    bst = lgb.Booster(model_file='quantile0.95model.txt')
    ypred = bst.predict(X_test)
    return ypred


input_file = open('resume.json', encoding='utf-8')
item = json.load(input_file)
print(func(item))