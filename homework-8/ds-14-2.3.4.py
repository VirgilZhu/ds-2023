from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

newsgroups = fetch_20newsgroups(subset='all', categories=['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med'], shuffle=True, random_state=0)

cv = CountVectorizer()
X = cv.fit_transform(newsgroups.data).toarray()
y = newsgroups.target
print("词袋模型向量化的结果向量：", X[0])
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(newsgroups.data).toarray()
print("TF-IDF表示法向量化的结果向量：", X[0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

bayes = GaussianNB()
bayes.fit(X_train, y_train)
y_pred = bayes.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=newsgroups.target_names)
print(f"高斯朴素贝叶斯分类准确度: {accuracy}")
print("分类报告:\n", report)