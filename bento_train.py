import bentoml

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

iris = load_iris()

X = iris.data[:, :4]
y = iris.target

model.fit(X,y)

bentoml_model = bentoml.sklearn.save_model('kneighbors', model)
print(f"Model Saved: ", {bentoml_model})
