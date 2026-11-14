
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def main():
    iris = load_iris()
    X = iris.data            # Features: sepal/petal length and width (150 x 4)
    y = iris.target          # Labels: 0 = setosa, 1 = versicolor, 2 = virginica
    target_names = iris.target_names

    # Optional: wrap in a DataFrame for a quick look at the data
    df = pd.DataFrame(X, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(y, target_names)
    print("=== Dataset Preview (first 5 rows) ===")
    print(df.head(), "\n")
    print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {list(target_names)}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,      # 20% for testing, 80% for training
        random_state=42,     # Reproducible split
        shuffle=True,        # Shuffle before splitting
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}\n")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print("=== Model Evaluation ===")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred), "\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))


if __name__ == "__main__":
    main()