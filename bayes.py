import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups, load_iris
from tensorflow.keras.datasets import mnist
from sklearn.metrics import f1_score

# Klasa dla Naive Bayes
class NaiveBayesClassifier:
    def __init__(self, data_type="continuous"):
        self.data_type = data_type  # Typ danych: "continuous", "binary", "text", "categorical"
        self.apriori = None
        self.conditional_probs = None
        if self.data_type == "text":
            self.vectorizer = TfidfVectorizer()
        else:
            self.vectorizer = None

    def calculate_apriori(self, Y):
        classes = sorted(list(set(Y)))
        self.apriori = [np.mean(Y == k) for k in classes]

    def calculate_conditional_probs_continuous(self, X, Y):
        self.conditional_probs = {}
        classes = sorted(list(set(Y)))

        for label in classes:
            self.conditional_probs[label] = {
                "mean": X[Y == label].mean(axis=0),
                "std": X[Y == label].std(axis=0) + 1e-6
            }

    def calculate_conditional_probs_binary(self, X, Y):
        classes = sorted(list(set(Y)))
        self.conditional_probs = np.ones((len(classes), 2, X.shape[1]))

        for label in classes:
            class_X = X[Y == label]
            self.conditional_probs[label, 1, :] = (class_X.sum(axis=0) + 1) / (len(class_X) + 2)
            self.conditional_probs[label, 0, :] = 1 - self.conditional_probs[label, 1, :]

    def calculate_conditional_probs_text(self, X, Y):
        classes = sorted(list(set(Y)))
        self.conditional_probs = {}

        for label in classes:
            class_X = X[Y == label]
            # Dodajemy wygładzanie Laplace'a
            self.conditional_probs[label] = (class_X.sum(axis=0) + 1) / (class_X.sum(axis=0).sum() + len(class_X[0]))

    def calculate_conditional_probs_categorical(self, X, Y):
        classes = sorted(list(set(Y)))
        self.conditional_probs = {}

        for label in classes:
            class_X = X[Y == label]
            self.conditional_probs[label] = {}
            for col in range(X.shape[1]):
                unique_values, counts = np.unique(class_X[:, col], return_counts=True)
                total_count = len(class_X)
                self.conditional_probs[label][col] = {
                    value: (count + 1) / (total_count + len(unique_values)) for value, count in zip(unique_values, counts)
                }

    def fit(self, X, Y):
        self.calculate_apriori(Y)

        if self.data_type == "continuous":
            self.calculate_conditional_probs_continuous(X, Y)
        elif self.data_type == "binary":
            self.calculate_conditional_probs_binary(X, Y)
        elif self.data_type == "text":
            X = self.vectorizer.fit_transform(X).toarray()
            self.calculate_conditional_probs_text(X, Y)
        elif self.data_type == "categorical":
            self.calculate_conditional_probs_categorical(X, Y)

    def gaussian_prob(self, x, mean, std):
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

    def predict(self, X):
        if self.data_type == "text":
            X = self.vectorizer.transform(X).toarray()

        predictions = []
        for x in X:
            posteriors = []
            for label in range(len(self.apriori)):
                posterior = self.apriori[label]

                if self.data_type == "continuous":
                    mean = self.conditional_probs[label]["mean"]
                    std = self.conditional_probs[label]["std"]
                    posterior *= np.prod(self.gaussian_prob(x, mean, std))
                elif self.data_type == "binary":
                    posterior *= np.prod([self.conditional_probs[label, int(x[i]), i] for i in range(len(x))])
                elif self.data_type == "text":
                    posterior *= np.prod(self.conditional_probs[label] ** x)
                elif self.data_type == "categorical":
                    for feature_index in range(len(x)):
                        if feature_index in self.conditional_probs[label]:
                            value = x[feature_index]
                            if value in self.conditional_probs[label][feature_index]:
                                posterior *= self.conditional_probs[label][feature_index][value]
                            else:
                                posterior *= 1e-6  # Mała wartość dla brakujących kategorii

                posteriors.append(posterior)
            predictions.append(np.argmax(posteriors))
        return np.array(predictions)

# Użycie na zbiorze danych Breast Cancer
data = pd.read_csv("Breast_cancer_data.csv")
data = data[["mean_radius", "mean_texture", "mean_smoothness", "diagnosis"]]
train, test = train_test_split(data, test_size=0.2, random_state=41)

X_train = train.iloc[:, :-1].values
Y_train = train.iloc[:, -1].values
X_test = test.iloc[:, :-1].values
Y_test = test.iloc[:, -1].values

classifier = NaiveBayesClassifier(data_type="continuous")
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

# Obliczanie współczynnika F1 dla Breast Cancer Dataset
f1 = f1_score(Y_test, Y_pred, average='weighted')
print("Wyniki dla zbioru danych Breast Cancer:")
print(f"Współczynnik F1: {f1:.2f}")

# Użycie na zbiorze danych MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images_binary = (train_images > 127).astype(int).reshape(train_images.shape[0], -1)
test_images_binary = (test_images > 127).astype(int).reshape(test_images.shape[0], -1)

classifier = NaiveBayesClassifier(data_type="binary")
classifier.fit(train_images_binary, train_labels)
Y_pred = classifier.predict(test_images_binary)

# Obliczanie współczynnika F1 dla MNIST
f1 = f1_score(test_labels, Y_pred, average='weighted')
print("\nWyniki dla zbioru danych MNIST:")
print(f"Współczynnik F1: {f1:.2f}")

# Użycie w klasyfikacji tekstu (20 Newsgroups)
newsgroups = fetch_20newsgroups(subset='all', categories=['rec.sport.baseball', 'sci.space'])
X = newsgroups.data
Y = newsgroups.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

classifier = NaiveBayesClassifier(data_type="text")
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

# Obliczanie współczynnika F1 dla 20 Newsgroups
f1 = f1_score(Y_test, Y_pred, average='weighted')
print("\nWyniki dla zbioru danych 20 Newsgroups:")
print(f"Współczynnik F1: {f1:.2f}")

# Użycie na zbiorze danych Iris (dane kategoryczne)
iris = load_iris()
X_iris = iris.data
Y_iris = iris.target

# Konwersja do danych kategorycznych
X_iris_cat = np.array([np.digitize(X_iris[:, i], bins=np.linspace(0, 8, num=4)) for i in range(X_iris.shape[1])]).T

X_train_cat, X_test_cat, Y_train_cat, Y_test_cat = train_test_split(X_iris_cat, Y_iris, test_size=0.2, random_state=41)

classifier = NaiveBayesClassifier(data_type="categorical")
classifier.fit(X_train_cat, Y_train_cat)
Y_pred_cat = classifier.predict(X_test_cat)

# Obliczanie współczynnika F1 dla zbioru danych Iris
f1 = f1_score(Y_test_cat, Y_pred_cat, average='weighted')
print("\nWyniki dla zbioru danych Iris:")
print(f"Współczynnik F1: {f1:.2f}")


#Wyniki dla zbioru danych Breast Cancer:
#Współczynnik F1: 0.96

#Wyniki dla zbioru danych MNIST:
#Współczynnik F1: 0.84

#Wyniki dla zbioru danych 20 Newsgroups:
#Współczynnik F1: 0.99

#Wyniki dla zbioru danych Iris:
#Współczynnik F1: 0.82
