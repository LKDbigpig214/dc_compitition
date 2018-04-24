import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
import lightgbm as lgb

df_protein_train = pd.read_csv('./data/df_protein_train.csv')
df_protein_test = pd.read_csv('./data/df_protein_test.csv')
df_protein = pd.concat([df_protein_train, df_protein_test])

texts = [[word for word in re.findall(r'.{3}', doc)]
         for doc in list(df_protein['Sequence'])]
n = 128
model = Word2Vec(texts, size=n, window=4, min_count=1,negative=3,
                 sg=1, sample=0.001, hs=1, workers=4)
vectors = pd.DataFrame([model[word] for word in (model.wv.vocab)])
vectors['World'] = list[model.wv.vocab)
vectors.columns = ["vec_{0}".format(i) for i in range(0,n)] + ['Word']

wide_vec = pd.DataFrame()
result = []
aa = list(df_protein['Protein_ID'])
for i in range(len(texts)):
    result2=[]
    for j in range(len(texts[i])):
        result2.append(aa[i])
    result1.extend(result2)
wide_vec['Id'] = result1

result1 = []
for i in range(len(texts)):
    result2=[]
    for j in range(len(texts[i])):
        result2.append(texts[i][j])
    result1.extend(result2)
wide_vec['Word'] = result1
del result1,result2

wide_vec = wide_vec.merge(vectors,on='Word',how='left')
wide_vec = wide_vec.drop('Word',axis=1)
cols = ['vec_{0}'.format(i) for i in range(0,n)]
wide_vec.columns = ['Protein_ID'] + cols
del vectors

df_protein = pd.DataFrame(wide_vec.groupby(['Protein_ID'])[cols].agg('mean'))
df_protein = df_protein.reset_index()
del wide_vec

df_molecule = pd.read_csv('./data/df_molecule.csv')
feat = []
for i in range(0,len(df_molecule)):
    feat.append(df_molecule['Fingerprint'][i].split(','))
feat = pd.DataFrame(feat)
feat = feat.astype('int')
feat.columns = ["Fingerprint_{0}".format(i) for i in range(0,167)]
feat['Molecule_ID'] = df_molecule['Molecule_ID']
df_molecule = df_molecule.drop('Fingerprint',axis=1)
df_molecule = df_molecule.merge(feat, on='Molecule_ID')
del feat

df_affinity_train = pd.read_csv('./data/df_affinity_train.csv')
df_affinity_predict = pd.read_csv('./data/df_affinity_test_toBePredicted.csv')

df_train = pd.merge(df_affinity_train,df_protein,on='Protein_ID',how='left')
df_train = pd.merge(df_train, df_molecule, on='Molecule_ID',how='left')

df_predict = pd.merge(df_affinity_predict,df_protein,on='Protein_ID',how='left')
df_predict = pd.merge(df_predict, df_molecule, on='Molecule_ID',how='left')
df_predict = df_predict.drop(columns=['Protein_ID','Molecule_ID'])
df_predict = df_predict.fillna(0)

del df_affinity_train

X = df_train.drop(columns=['Ki','Protein_ID','Molecule_ID']).fillna(0)
y = df_train['Ki'].fillna(0)

'''from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor().fit(X,y)
print(X.shape)
model = SelectFromModel(clf, prefit=True)
X = model.transform(X)
df_predict = model.transform(df_predict)
print(X.shape)
print(df_predict.shape)'''

X_train,X_test,y_train,y_test = train_test_split(X,ytest_size=0.25,random_state=33)

train = lgb.Dataset(X_train, label=y_train)
test = lgb.Dataset(X_test, label=y_test, reference=train)
params={
    'boosting_type' : 'gbdt',
    'objective' : 'regression_l2',
    'metric' : 'l2',
    'min_child_weight' : 3,
    'num_leaves' : 2**5,
    'lambda_l2' : 10,
    'subsample' : 0.7,
    'colsample_bytree' : 0.7,
    'colsample_bylevel' : 0.7,
    'learning_rate' : 0.05,
    'tree_method' : 'exact',
    'seed' : 2017,
    'nthread' : 12,
    'silent' : True
    }
num_round = 3000
gbm = lgb.train(params,
                train,
                num_round,
                verbose_eval=50,
                valid_sets=[train,test]
                )
preds_sub = gbm.predict(df_predict)

df_affinity_predict['Ki'] = preds_sub
df_affinity_predict.to_csv('./lgb.csv',index=False)


#cyp_3a4, cyp_2c9, cyp_2d6, ames_toxicity, fathead_minnow_toxicity
#tetrahymena_pyriformis_toxicity, honey_bee, cell_permeability,
#logP, renal_organic_cation_transporter, CLtotal, hia, biodegradation,
#Vdd, p_glycoprotein_inhibition, NOAEL, solubility, bbb
#g = sns.FacetGrid(df_molecule)
#g.map(sns.kdeplot, 'cyp_2c9',shade=True)
#sns.pairplot(df_molecule[['cyp_3a4','cyp_2c9','cyp_2d6','ames_toxicity']].dropna())
#plt.show()
