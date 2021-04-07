import os, json
import pandas as pd
import io
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

alphabet = {"а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м", "н", "о",
            "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я"}

pd.set_option('display.max_columns', None)
input_file = open('resume_ratings_corrected.json', encoding='utf-8')
json_object = json.load(input_file)
json_array = json_object['resumes']
# with open('resume_ratings.json', encoding='utf-8') as data_file:
#     json_array = json.loads(data_file.read())
# this finds our json files
# here I define my pandas Dataframe with the columns I want to get from the json
X_train = pd.DataFrame(columns=['age', 'skills', 'java', 'maven', 'git', 'sql', 'hibernate', 'spring', 'oop',
                                'student', 'education_level', 'total_experience', 'skill_set', 'rubbish'])
X_test = pd.DataFrame(columns=['age', 'skills', 'java', 'maven', 'git', 'sql', 'hibernate', 'spring', 'oop',
                               'student', 'education_level', 'total_experience', 'skill_set', 'rubbish'])
y_train = pd.DataFrame(columns=['rating'])
y_test = pd.DataFrame(columns=['rating'])
i = 0
k = 0
N = len(json_array)
javacount = 0
mavencount = 0
gitcount = 0
englishcount = 0
sqlcount = 0
hibernatecount = 0
springcount = 0
oopcount = 0
rubbishcount = 0

javarating = 0
mavenrating = 0
gitrating = 0
englishrating = 0
sqlrating = 0
hibernaterating = 0
springrating = 0
ooprating = 0
rubbishrating = 0

count = 0
d = {}
newcount = 0
for item in json_array:
    java = False
    maven = False
    git = False
    english = False
    sql = False
    hibernate = False
    spring = False
    oop = False
    rubbish = False
    if item['resume']['skills'] is not None:
        skills = True
    else:
        skills = False
    skill_set = item['resume']['skill_set']
    rubbishcount = 0
    for item_in in skill_set:
        if bool(alphabet.intersection(
                set(item_in.lower()))) and not "английский язык" == item_in.lower() \
                and not "ооп" == item_in.lower():
            rubbish = True
            rubbishcount += 1
            continue

        if item_in.lower() in d:
            d[item_in.lower()] += 1
        else:
            d[item_in.lower()] = 1

        if "java" in item_in.lower() and "javascript" not in item_in.lower() and not java:
            java = True
            javacount += 1
            continue
        if "maven" in item_in.lower() and not maven:
            maven = True
            mavencount += 1
            continue
        if "git" in item_in.lower() and not git:
            git = True
            gitcount += 1
            continue
        if ("english" in item_in.lower() or "aнглийский язык" in item_in.lower()) and not english:
            english = True
            englishcount += 1
            continue
        if "sql" in item_in.lower() and not sql:
            sql = True
            sqlcount += 1
            continue
        if "hibernate" in item_in.lower() and not hibernate:
            hibernate = True
            hibernatecount += 1
            continue
        if "spring" in item_in.lower() and not spring:
            spring = True
            springcount += 1
            continue
        if "ооп" in item_in.lower() and not oop:
            oop = True
            oopcount += 1
            continue
    skill_set = ",".join(skill_set)
    if item['resume']['age'] is not None:
        age = item['resume']['age']
    else:
        age = 0
    student = False
    array_of_educations = item['resume']['education']['primary']
    for x in array_of_educations:
        if x['year'] > 2020:
            student = True
            break
    if item['resume']['education']['level']['id'] is not None:
        education_level = item['resume']['education']['level']['id']
    else:
        education_level = "none"
    if item['resume']['total_experience'] is not None:
        total_experience = item['resume']['total_experience']['months']
    else:
        total_experience = 0
    rating = item['rating']

    if rating >= 0.5:
        count += 1
        if java:
            javarating += 1
        if maven:
            mavenrating += 1
        if git:
            gitrating += 1
        if english:
            englishrating += 1
        if sql:
            sqlrating += 1
        if hibernate:
            hibernaterating += 1
        if spring:
            springrating += 1
        if oop:
            ooprating += 1
        if rubbish:
            rubbishrating += 1
        if count < 9:
            X_train.loc[i] = [age, skills, java, maven, git, sql, hibernate, spring, oop, student,
                              education_level, total_experience, skill_set, rubbishcount]
            y_train.loc[i] = [rating]
            i += 1
        else:
            X_test.loc[k] = [age, skills, java, maven, git, sql, hibernate, spring, oop, student,
                             education_level, total_experience, skill_set, rubbishcount]
            y_test.loc[k] = [rating]
            k += 1
    # rating = json_text['rating']
    # отделить для каждого НУЖНОГО навыка столбец, завести переменную столбца и указывать ее для каждого кантидата
    # например: Java, Maven, Git, Английский язык
    # номер курса (если обучение закончилось, то курс = 0)
    # после чего удалять из списка нужные навыки, если их не было, то ничего и не делать

    # print('{}   {}   {}'.format(i, skills,json_text['skills']))

    if i < 0.7 * N and rating < 0.5:
        X_train.loc[i] = [age, skills, java, maven, git, sql, hibernate, spring, oop, student,
                          education_level, total_experience, skill_set,rubbishcount]
        y_train.loc[i] = [rating]
        i += 1
    elif rating < 0.5:
        X_test.loc[k] = [age, skills, java, maven, git, sql, hibernate, spring, oop, student,
                         education_level, total_experience, skill_set,rubbishcount]
        y_test.loc[k] = [rating]
        k += 1
sorted_values = sorted(d.values())  # Sort the values
sorted_dict = {}

for i in sorted_values:
    for k in d.keys():
        if d[k] == i:
            sorted_dict[k] = d[k]
            break
print(sorted_dict)
print(str(javacount) + " " + str(mavencount) + " " + str(gitcount) + " "
      + str(sqlcount) + " " + str(hibernatecount) + " " + str(springcount) + " " + str(oopcount) + " " + str(rubbishcount))
print(str(javarating) + " " + str(mavenrating) + " " + str(gitrating) + " "
      + str(sqlrating) + " " + str(hibernaterating) + " " + str(springrating) + " " + str(ooprating)+" "+str(rubbishrating))
# obj_feat = list(X_train.loc[:, X_train.dtypes == 'object'].columns.values)
# for feature in obj_feat:
# X_train[feature] = pd.Series(X_train[feature], dtype="category")
X_train['age'] = X_train['age'].astype(str).astype(int)
X_train['skills'] = X_train['skills'].astype(str).astype(bool)
X_train['java'] = X_train['java'].astype(str).astype(bool)
X_train['maven'] = X_train['maven'].astype(str).astype(bool)
X_train['git'] = X_train['git'].astype(str).astype(bool)
X_train['sql'] = X_train['sql'].astype(str).astype(bool)
X_train['hibernate'] = X_train['hibernate'].astype(str).astype(bool)
X_train['spring'] = X_train['spring'].astype(str).astype(bool)
X_train['oop'] = X_train['oop'].astype(str).astype(bool)
X_train['student'] = X_train['student'].astype(str).astype(bool)
X_train['total_experience'] = X_train['total_experience'].astype(str).astype(int)
X_train['rubbish'] = X_train['rubbish'].astype(str).astype(int)
# X_train['education_level'] = X_train['education_level'].astype("|S")
# X_train['skill_set'] = X_train['skill_set'].str.replace(',', '')
# X_train['skill_set'] = X_train['skill_set'].astype("|S")
X_train['education_level'] = pd.Series(X_train['education_level'], dtype="category")
X_train['skill_set'] = pd.Series(X_train['skill_set'], dtype="category")

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

X_train.to_csv('data.csv', header=True)
print(X_train.dtypes)
print(y_train.dtypes)
print(X_test)
print(X_train)
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

categorical_feats = ['skill_set', 'education_level']

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 128,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'force_col_wise': True
}
print('Starting training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=10)
# tree_imp = lgb.importance(gbm, percentage = True)
# lgb.plot.importance(tree_imp, top_n = 5, measure = "Gain")
print('Saving model...')
# save model to file
gbm.save_model('model.txt')

print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
