import json
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
import gbm as gbm
import pandas as pd
import numpy as np
import io
import random
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import  graphviz


# def l2_loss(y, data):
#     t = data.get_label()
#     grad = y - t
#     hess = np.ones_like(y)
#     return grad, hess
#
#
# def l2_eval(y, data):
#     t = data.get_label()
#     loss = (y - t) ** 2
#     return 'l2', loss.mean(), False


# def custom_asymmetric_train(y_pred, y_true):
#     residual = (y_true - y_pred.label).astype("float")
#     grad = np.where(residual < 0, 2 * 10.0 * residual, 2 * residual)
#     hess = np.where(residual < 0, 2 * 10.0, 2.0)
#     return grad, hess
#
#
# def custom_asymmetric_valid(y_true, y_pred):
#     residual = (y_true - y_pred.label).astype("float")
#     loss = np.where(y_true < 0, (residual ** 2) * 10.0, residual ** 2)
#     return "custom_asymmetric_eval", np.mean(loss), False

def custom_asymmetric_train(y_pred, y_true):
    y_true = y_true.get_label()
    residual = (y_true - y_pred).astype("float")
    grad = np.where(y_true > 0.5, -2 * residual * 1000.0, -2 * residual * 2)
    hess = np.where(y_true > 0.5, 2 * 1000.0, 2 * 2)
    return grad, hess


def custom_asymmetric_valid(y_pred, y_true):
    y_true = y_true.get_label()
    residual = (y_true - y_pred).astype("float")
    loss = np.where(y_true > 0.5, (residual ** 10) * 1000.0, residual)
    return "custom_asymmetric_eval", np.median(loss), False


# def custom_asymmetric_objective(y_true, y_pred):
#     residual = (y_true - y_pred).astype("float")
#     grad = np.where(y_true > 0.5, -2 * 10.0 * residual, -2 * residual)
#     hess = np.where(y_true > 0.5, 2 * 10.0, 2.0)
#     return grad, hess
#
#
# def custom_asymmetric_eval(y_true, y_pred):
#     residual = (y_true - y_pred).astype("float")
#     loss = np.where(y_true > 0.5, (residual ** 2) * 10.0, residual ** 2)
#     return "custom_asymmetric_eval", np.mean(loss), False


alphabet = {"а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м", "н", "о",
            "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я"}

pd.set_option('display.max_columns', None)
input_file = open('resume_ratings_corrected.json', encoding='utf-8')
json_object = json.load(input_file)
json_array = json_object['resumes']
# json_array = np.random.shuffle(json_array)
# with open('resume_ratings.json', encoding='utf-8') as data_file:
#     json_array = json.loads(data_file.read())
# this finds our json files
# here I define my pandas Dataframe with the columns I want to get from the json
X_train = pd.DataFrame(
    columns=['age', 'skills', 'city', 'gender', 'specialty', 'java', 'maven', 'git', 'sql', 'hibernate', 'spring',
             'oop',
             'student', 'education_level', 'total_experience', 'skill_set', 'rubbish'])
X_test = pd.DataFrame(
    columns=['age', 'skills', 'city', 'gender', 'specialty', 'java', 'maven', 'git', 'sql', 'hibernate', 'spring',
             'oop',
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
titles = {}
universities = {}
newcount = 0
meanage = 0
for item in json_array:
    if item['resume']['title'].lower() in titles:
        titles[item['resume']['title'].lower()] += 1
    else:
        titles[item['resume']['title'].lower()] = 1
    tl = item['resume']['title'].lower()
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
        age = meanage
        # age = 0
    meanage = int((meanage + age) / 2)
    city = item['resume']['area']['name']
    gender = item['resume']['gender']['id']
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

    if item['rating'] == 0:
        rating = 0.1
    elif item['rating'] == 0.1:
        rating = 0.2
    elif item['rating'] == 0.2 or item['rating'] == 0.3:
        rating = 0.3
    else:
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

            X_train.loc[i] = [age, skills, city, gender, specialty, java, maven, git, sql, hibernate, spring, oop,
                              student,
                              education_level, total_experience, skill_set, rubbishcount]
            y_train.loc[i] = [rating]
            i += 1
        else:
            if k == 135:
                p = 0
            X_test.loc[k] = [age, skills, city, gender, specialty, java, maven, git, sql, hibernate, spring, oop,
                             student,
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

        X_train.loc[i] = [age, skills, city, gender, specialty, java, maven, git, sql, hibernate, spring, oop, student,
                          education_level, total_experience, skill_set, rubbishcount]
        y_train.loc[i] = [rating]
        i += 1
    elif rating < 0.5:
        if k == 135:
            p = 0
        X_test.loc[k] = [age, skills, city, gender, specialty, java, maven, git, sql, hibernate, spring, oop, student,
                         education_level, total_experience, skill_set, rubbishcount]
        y_test.loc[k] = [rating]
        k += 1
sorted_values = sorted(d.values())  # Sort the values
sorted_dict = {}

for i in sorted_values:
    for k in d.keys():
        if d[k] == i:
            sorted_dict[k] = d[k]
            break

sorted_titles = sorted(titles.values())  # Sort the values
sorted_dict_titles = {}

for i in sorted_titles:
    for k in titles.keys():
        if titles[k] == i:
            sorted_dict_titles[k] = titles[k]
            break
print(sorted_dict_titles)
print(sorted_dict)
print(str(javacount) + " " + str(mavencount) + " " + str(gitcount) + " "
      + str(sqlcount) + " " + str(hibernatecount) + " " + str(springcount) + " " + str(oopcount) + " " + str(
    rubbishcount))
print(str(javarating) + " " + str(mavenrating) + " " + str(gitrating) + " "
      + str(sqlrating) + " " + str(hibernaterating) + " " + str(springrating) + " " + str(ooprating) + " " + str(
    rubbishrating))
# obj_feat = list(X_train.loc[:, X_train.dtypes == 'object'].columns.values)
# for feature in obj_feat:
# X_train[feature] = pd.Series(X_train[feature], dtype="category")

X_train['age'] = X_train['age'].astype(str).astype(int)
X_train['skills'] = X_train['skills'].astype(str).astype(bool)
X_train['city'] = pd.Series(X_train['city'], dtype="category")
X_train['gender'] = pd.Series(X_train['gender'], dtype="category")
X_train['specialty'] = pd.Series(X_train['specialty'], dtype="category")
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
X_test['city'] = pd.Series(X_test['city'], dtype="category")
X_test['gender'] = pd.Series(X_test['gender'], dtype="category")
X_test['specialty'] = pd.Series(X_test['specialty'], dtype="category")
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

categorical_feats = ['skill_set', 'education_level', 'city', 'gender']

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 128,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.75,
    'min_data_in_leaf': 20,
    'bagging_freq': 5,
    'verbose': 0,
    'force_col_wise': True
}
lgb_params = {
    'n_jobs': 1,
    'max_depth': 4,
    'min_data_in_leaf': 10,
    'subsample': 0.9,
    'n_estimators': 80,
    'learning_rate': 0.1,
    'colsample_bytree': 0.9,
    'boosting_type': 'gbdt'
}
lgb_l2 = LGBMRegressor(objective='regression', **lgb_params)
lgb_l2.fit(X_train, y_train)
quantile_alphas = [0.985]

lgb_quantile_alphas = {}
for quantile_alpha in quantile_alphas:
    # to train a quantile regression, we change the objective parameter and
    # specify the quantile value we're interested in
    lgb1 = LGBMRegressor(objective='quantile', alpha=quantile_alpha, **lgb_params)
    lgb1.fit(X_train, y_train)
    lgb_quantile_alphas[quantile_alpha] = lgb
    # lgb.savemodel('quantile0.95model.txt')
    lgb1.booster_.save_model('quantile0.95model.txt')
    y_pred = lgb1.predict(X_test)
    count = 0


    print('Plotting feature importances...')
    ax = lgb.plot_importance(lgb1, max_num_features=20)
    plt.show()
    #
    # print('Plotting split value histogram...')
    # ax = lgb.plot_split_value_histogram(lgb1, feature='total_experience', bins='auto')
    # plt.show()

    print('Plotting 54th tree...')  # one tree use categorical feature to split
    for i in range(80):
        ax = lgb.plot_tree(lgb1, tree_index=i, figsize=(15, 15), show_info=['internal_count'])
        plt.show()


    result = pd.DataFrame(
        columns=['y_true', 'y_pred'])

    y_test1 = y_test.rating.values
    count1 = 0
    count2 = 0
    index = []
    for i in range(len(y_pred)):
        result.loc[i] = [y_test1[i], y_pred[i]]
        if y_test1[i] >= 0.4 and y_pred[i] >= 0.3:
            count1 += 1
        elif y_test1[i] >= 0.4 and y_pred[i] < 0.3:
            index.append(i)
        if y_test1[i] >= 0.4:
            count2 += 1
        if y_pred[i] >= 0.3:
            count += 1
    for ind in index:
        print(ind)
    #p = X_test[135]
    result.to_csv(r'D:\Виканоут\Курсач\HHClient\Parser\export_dataframe.csv', index=False, header=True)
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    print(count)
    print(count1)
    print(count2)



print('Starting training...')
# make new model on new value
# gbm6 = lgb.LGBMRegressor(random_state=33,
#                          early_stopping_rounds=10,
#                          n_estimators=10000
#                          )
#
# gbm6.set_params(**{'objective': custom_asymmetric_objective}, metrics=["mse", 'mae'])
#
# gbm6.fit(
#     X_train,
#     y_train,
#     eval_set=[(X_test, y_test)],
#     eval_metric=custom_asymmetric_eval,
#     verbose=False,
# )
# pred = gbm6.predict(X_test)
# pred = pred.reshape(255, 1)
# loss_gbm6 = custom_asymmetric_eval(y_test, pred)
# score_dict = {
#               'LightGBM with early_stopping, custom training and custom validation loss':
#                   {'asymmetric custom mse (test)': loss_gbm6,
#                    'asymmetric custom mse (train)': custom_asymmetric_eval(y_train, gbm6.predict(X_train))[1],
#                    'symmetric mse': mean_squared_error(y_test, gbm6.predict(X_test)),
#                    '# boosting rounds': gbm6.booster_.current_iteration()}
#               }
# print(loss_gbm6)
# eval
# print('The rmse of prediction is:', mean_squared_error(y_test, y_pred6) ** 0.5)
# train

# model = lgb.LGBMRegressor(**params, n_estimators = 10000, n_jobs = -1)
# gbm1 = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=1000,
#                 init_model=gbm,
#                 fobj=custom_asymmetric_train,
#                 feval=custom_asymmetric_valid,
#                 valid_sets=lgb_eval)
# y_pred1 = gbm1.predict(X_test)
# print('The rmse of prediction is:', mean_squared_error(y_test, y_pred1) ** 0.5)

# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=10,
#                 init_model=gbm,
#                 fobj=custom_asymmetric_train,
#                 feval=custom_asymmetric_valid,
#                 valid_sets=lgb_eval,
#                 early_stopping_rounds=10)
# # tree_imp = lgb.importance(gbm, percentage = True)
# # lgb.plot.importance(tree_imp, top_n = 5, measure = "Gain")
# print('Saving model...')
# # save model to file
# gbm.save_model('model.txt')
#
# print('Starting predicting...')
# # predict
# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
#
# # eval
# print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
