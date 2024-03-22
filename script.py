from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris_data = load_iris()
X = iris_data['data']
y = iris_data['target']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)

classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(f"accuracy score {accuracy_score(y_test, y_pred)}")