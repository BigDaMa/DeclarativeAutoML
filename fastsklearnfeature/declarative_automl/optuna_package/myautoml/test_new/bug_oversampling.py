from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split as tts
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

X, y = make_classification(n_classes=2, class_sep=2,
weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)

pca = PCA()
smt = SMOTE(random_state=42)
knn = RandomForestClassifier()
pipeline = Pipeline([('smt', smt), ('pca', pca), ('knn', knn)])
X_train, X_test, y_train, y_test = tts(X, y, random_state=42)
pipeline.fit(X_train, y_train, sample_weight=compute_sample_weight(class_weight='balanced', y=y_train))