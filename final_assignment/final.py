import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.width', 100)

data = pd.read_csv('LOL_rank_10min.csv')

# print("前五行数据:\n", data.head())
print("数据形状:", data.shape)
# pd.set_option('display.width', 10)
# print("数据列名:", data.columns)
# pd.set_option('display.width', 80)
# print("数据概览:\n", data.describe())

data.dropna(axis=0, how='any', inplace=True)
data = data.drop(['gameId'], axis=1)

plt.figure(figsize=(18, 15))
sns.heatmap(round(data.corr(), 1), cmap="coolwarm", annot=True, linewidths=.5)  # 相关性(-1,1)
plt.savefig('热力图相关性分析.jpg', bbox_inches='tight')


# 找出相关系数矩阵中相关性大的一组数据,返回其中一列数据
def remove_redundancy(r):
    to_remove = []
    for i in range(len(r.columns)):
        for j in range(i):
            if abs(r.iloc[i, j]) >= 1 and (r.columns[j] not in to_remove):
                print("相关性:", r.iloc[i, j], r.columns[j], r.columns[i])
                to_remove.append(r.columns[i])
    return to_remove


clean_data = data.drop(remove_redundancy(data.corr()), axis=1)  # 删去相关性较高项

pd.set_option('display.width', 10)
print("初步处理后的数据:", clean_data.columns)

# 将击杀野怪和小兵数合并:
clean_data['blueMinionsTotales'] = clean_data['blueTotalMinionsKilled'] + clean_data['blueTotalJungleMinionsKilled']
clean_data['redMinionsTotales'] = clean_data['redTotalMinionsKilled'] + clean_data['redTotalJungleMinionsKilled']
clean_data = clean_data.drop(['blueTotalMinionsKilled'], axis=1)
clean_data = clean_data.drop(['blueTotalJungleMinionsKilled'], axis=1)
clean_data = clean_data.drop(['redTotalMinionsKilled'], axis=1)
clean_data = clean_data.drop(['redTotalJungleMinionsKilled'], axis=1)

# 等级和经验分析:
plt.figure(figsize=(12, 12))
plt.subplot(121)
sns.scatterplot(x='blueAvgLevel', y='blueTotalExperience', hue='blueWins', data=clean_data)
plt.title('blue')
plt.xlabel('blueAvgLevel')
plt.ylabel('blueTotalExperience')
plt.grid(True)
plt.subplot(122)
sns.scatterplot(x='redAvgLevel', y='redTotalExperience', hue='blueWins', data=clean_data)
plt.title('red')
plt.xlabel('redAvgLevel')
plt.ylabel('redTotalExperience')
plt.grid(True)
plt.savefig('等级和经验分析.jpg', bbox_inches='tight')

# 删去等级列
clean_data = clean_data.drop(['blueAvgLevel'], axis=1)
clean_data = clean_data.drop(['redAvgLevel'], axis=1)

sns.set(font_scale=1.5)
plt.figure(figsize=(20, 20))
sns.set_style("whitegrid")

# 击杀和被击杀数绘制散点图
plt.subplot(321)
sns.scatterplot(x='blueKills', y='blueDeaths', hue='blueWins', data=clean_data)
plt.title('blueKills&&blueDeaths')
plt.xlabel('blueKills')
plt.ylabel('blueDeaths')
plt.grid(True)

# 助攻数绘制散点图
plt.subplot(322)
sns.scatterplot(x='blueAssists', y='redAssists', hue='blueWins', data=clean_data)
plt.title('Assists')
plt.xlabel('blueAssists')
plt.ylabel('redAssists')
plt.tight_layout(pad=1.5)
plt.grid(True)

# 双方金币数绘制散点图
plt.subplot(323)
sns.scatterplot(x='blueTotalGold', y='redTotalGold', hue='blueWins', data=clean_data)
plt.title('TotalGold')
plt.xlabel('blueTotalGold')
plt.ylabel('redTotalGold')
plt.tight_layout(pad=1.5)
plt.grid(True)

# 双方经验绘制散点图
plt.subplot(324)
sns.scatterplot(x='blueTotalExperience', y='redTotalExperience', hue='blueWins', data=clean_data)
plt.title('Experience')
plt.xlabel('blueTotalExperience')
plt.ylabel('redTotalExperience')
plt.tight_layout(pad=1.5)
plt.grid(True)

# 双方插眼数量绘制散点图
plt.subplot(325)
sns.scatterplot(x='blueWardsPlaced', y='redWardsPlaced', hue='blueWins', data=clean_data)
plt.title('WardsPlaced')
plt.xlabel('blueWardsPlaced')
plt.ylabel('redWardsPlaced')
plt.tight_layout(pad=1.5)
plt.grid(True)

# 击杀的小兵和野怪总数绘制散点图
plt.subplot(326)
sns.scatterplot(x='blueMinionsTotales', y='redMinionsTotales', hue='blueWins', data=clean_data)
plt.title('MinionsTotales')
plt.xlabel('blueMinionsTotales')
plt.ylabel('redMinionsTotales')
plt.tight_layout(pad=1.5)
plt.grid(True)
plt.savefig('数据分析.jpg', bbox_inches='tight')

# 将一些数据转换为它们的差值:
clean_data['WardsPlacedDiff'] = clean_data['blueWardsPlaced'] - clean_data['redWardsPlaced']
clean_data['WardsDestroyedDiff'] = clean_data['blueWardsDestroyed'] - clean_data['redWardsDestroyed']
clean_data['AssistsDiff'] = clean_data['blueAssists'] - clean_data['redAssists']
clean_data['blueHeraldsDiff'] = clean_data['blueHeralds'] - clean_data['redHeralds']
clean_data['blueDragonsDiff'] = clean_data['blueDragons'] - clean_data['redDragons']
clean_data['blueTowersDestroyedDiff'] = clean_data['blueTowersDestroyed'] - clean_data['redTowersDestroyed']
clean_data['EliteMonstersDiff'] = clean_data['blueEliteMonsters'] - clean_data['redEliteMonsters']
clean_data = clean_data.drop(['blueWardsPlaced'], axis=1)
clean_data = clean_data.drop(['redWardsPlaced'], axis=1)
clean_data = clean_data.drop(['blueWardsDestroyed'], axis=1)
clean_data = clean_data.drop(['redWardsDestroyed'], axis=1)
clean_data = clean_data.drop(['blueAssists'], axis=1)
clean_data = clean_data.drop(['redAssists'], axis=1)
clean_data = clean_data.drop(['blueHeralds'], axis=1)
clean_data = clean_data.drop(['redHeralds'], axis=1)
clean_data = clean_data.drop(['blueTowersDestroyed'], axis=1)
clean_data = clean_data.drop(['redTowersDestroyed'], axis=1)
clean_data = clean_data.drop(['blueDragons'], axis=1)
clean_data = clean_data.drop(['redDragons'], axis=1)
clean_data = clean_data.drop(['blueEliteMonsters'], axis=1)
clean_data = clean_data.drop(['redEliteMonsters'], axis=1)
clean_data = clean_data.drop(['redTotalGold'], axis=1)
clean_data = clean_data.drop(['redTotalExperience'], axis=1)

print("最终数据:", clean_data.columns)
plt.figure(figsize=(18, 15))
sns.heatmap(round(clean_data.corr(), 1), cmap="coolwarm", annot=True, linewidths=.5, annot_kws={"size": 12})
plt.savefig('最终数据热力图.jpg', bbox_inches='tight')

# 数据标准化
unscaled_inputs = clean_data.filter([
    'blueFirstBlood',
    'blueKills',
    'blueDeaths',
    'blueTotalGold',
    'blueTotalExperience',
    'blueGoldDiff',
    'blueExperienceDiff',
    'blueMinionsTotales',
    'redMinionsTotales',
    'WardsPlacedDiff',
    'WardsDestroyedDiff',
    'AssistsDiff',
    'blueHeraldsDiff',
    'blueDragonsDiff',
    'blueTowersDestroyedDiff',
    'EliteMonstersDiff'], axis=1)
target = clean_data.filter(['blueWins'])


# 创建自定义缩放器类
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy, with_mean, with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


columns_to_omit = ['blueFirstBlood', 'blueDragonsDiff']
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]
blue_scaler = CustomScaler(columns_to_scale)
blue_scaler.fit(unscaled_inputs)
scaled_inputs = blue_scaler.transform(unscaled_inputs)
pd.set_option('display.width', 80)
# print("标准化处理后的数据:", scaled_inputs)

# 模型训练
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, target, train_size=0.8, random_state=2)
reg = LogisticRegression()
reg.fit(x_train, y_train)
variables = unscaled_inputs.columns.values
intercept = reg.intercept_
summary_table = pd.DataFrame(columns=['Variables'], data=variables)
summary_table['Coef'] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table['Odds Ratio'] = np.exp(summary_table.Coef)
summary_table.sort_values(by=['Odds Ratio'], ascending=False)
print("模型变量评价:\n", summary_table.sort_values(by=['Odds Ratio'], ascending=False))
print("训练数据评分:", reg.score(x_train, y_train))
print("测试数据评分:", reg.score(x_test, y_test))

predicted_prob = reg.predict_proba(x_test)
data['predicted'] = reg.predict_proba(scaled_inputs)[:, 1]

col_n = ['blueWins', 'predicted']
a = pd.DataFrame(data, columns=col_n)
print("原始数据和胜率分析对比:\n", a)
