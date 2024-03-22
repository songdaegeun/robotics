import pandas as pd
import matplotlib.pyplot as plt   # package for plotting
from sklearn.datasets import load_breast_cancer # 유방암데이터
from sklearn.tree import DecisionTreeClassifier as DT, plot_tree
from sklearn.model_selection import train_test_split

plt.rcParams['axes.unicode_minus'] = False       # 마이너스 부호 깨짐 현상 
plt.rcParams["font.family"] = 'NanumBarunGothic' # 한글폰트 전역 설정

data = load_breast_cancer()
# print("Data dimension:", data.data.shape)
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3, random_state=0)
# print("Train Data:", X_tr.shape, "Test Data:", X_te.shape)

clf = DT(random_state=0)
clf = clf.fit(X_tr, y_tr)
plt.figure(figsize=(7,5), dpi=200)
plot_tree(clf, filled=True)
# plt.show()
plt.savefig('tmp1.png')

# cost-complexity pruninig(ccp)
path = clf.cost_complexity_pruning_path(X_tr, y_tr)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
print(ccp_alphas)
print(ccp_alphas.shape)

print(impurities)
print(impurities.shape)
plot_tree(clf, filled=True)

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("ccp alpha")
ax.set_ylabel("total impurity")
ax.set_title("ccp alpha vs total impurity for training set") 
plt.savefig('tmp2.png')

clfs = []
for ccp_alpha in ccp_alphas[:-1]:
    clf_ = DT(random_state=0, ccp_alpha=ccp_alpha)
    clf_.fit(X_tr, y_tr)
    clfs.append(clf_)

# numbers of nodes for each tree model
# t_nodes = [clf.tree_.node_count for clf in clfs]
# import numpy as np
    
fig, ax = plt.subplots(3,3, figsize=(20/2, 20/2))  
k = 0
for tree_m in clfs[2:]:
    i = k//3 
    j = k%3
    plot_tree(tree_m, feature_names = data.feature_names,
         class_names=data.target_names, filled=True, ax=ax[i,j])
    ax[i,j].set_title(("No. of nodes=", tree_m.tree_.node_count), fontsize=20/2)
    k += 1
plt.savefig('tmp3.png')

train_scores = []
test_scores = []
for ccp_alpha in ccp_alphas[:-1]:
    clf0 = DT(random_state=0, ccp_alpha=ccp_alpha)
    clf0.fit(X_tr, y_tr) # 모형적합
    train_scores.append(clf0.score(X_tr, y_tr)) # 
    test_scores.append(clf0.score(X_te, y_te))

fig, ax = plt.subplots()
ax.set_xlabel("Alpha")
ax.set_ylabel("Accuracy")
ax.set_title("Alpha vs. Accuracy for training and testing sets")
ax.plot(ccp_alphas[:-1], train_scores, marker="o", label="train set accuacy", drawstyle="steps-post")
ax.plot(ccp_alphas[:-1], test_scores, marker="o", label="test set accuracy", drawstyle="steps-post")
ax.legend()
plt.savefig('tmp4.png')

df = pd.DataFrame({"test acc": test_scores, "alpha":ccp_alphas[:-1]})
print(df)
# 정분류율이 가장 작은 모형 선택
max_idx = df["test acc"].idxmax()
print("optimal tree index",max_idx, "optimal alpha", ccp_alphas[max_idx]) #

# 최종모형 선택
clf_p = clfs[max_idx]
# 모형비교
fig, ax = plt.subplots(1,2, figsize=(20, 10))  # whatever size you want
plot_tree(clf, filled=True, ax=ax[0])
plot_tree(clf_p, filled=True, ax=ax[1])
ax[0].set_title('Unprunned DT (최초 pruning전 모형)', fontsize=20)
ax[1].set_title('Prunned DT (최종모형)', fontsize=20)
plt.savefig('tmp5.png')

# Evaluation metric
from sklearn.metrics import confusion_matrix, \
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc

y_pred = clf_p.predict(X_te)
print(y_pred)

# y_te[:10], y_pred[:10]

print("confusion_matrix\n", confusion_matrix(y_te, y_pred))
print("accuracy_score\n",accuracy_score(y_te, y_pred))
print("precision_score\n",precision_score(y_te, y_pred))
print("recall_score\n",recall_score(y_te, y_pred))
print("f1_score\n",accuracy_score(y_te, y_pred))


# ROC를 계산하기 위해서는 확률예측을 해야함
# sample이 positive class에 속할 probability는 predict_proba 반환 변수의 1열에 저장되어 있음
y_pred_prob = clf_p.predict_proba(X_te)[:, 1]
fpr, tpr, _ = roc_curve(y_te, y_pred_prob)
print(fpr, tpr)
roc_auc = auc(fpr, tpr) # auc 계산

plt.figure()
plt.plot(fpr, tpr,
    color="darkorange",
    label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC for Decision Tree model")
plt.legend(loc="lower right")
plt.savefig('tmp6.png')
