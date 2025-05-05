import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from itertools import product

np.random.seed(42)
print("\n" + "="*80)
print("LOADING AND PREPARING DATA".center(80))
print("="*80)

data = pd.read_csv('train.csv')
data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']].dropna()
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
X = data.drop('Survived', axis=1).values
y = data['Survived'].values

print(f"Dataset: Titanic")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Classes: ['Not Survived', 'Survived']")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data has been scaled using StandardScaler")

print("\n" + "="*80)
print("HYPERPARAMETER TUNING".center(80))
print("="*80)

mlp = MLPClassifier(
    random_state=42,
    max_iter=3000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10
)

param_grid = {
    'hidden_layer_sizes': [(10,), (50,), (100,), (10, 10), (50, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.01, 0.0001]
}

print("Parameters being tuned:")
for param, values in param_grid.items():
    print(f"- {param}: {values}")

keys = param_grid.keys()
combinations = list(product(*(param_grid[key] for key in keys)))
all_params = []
for combo in combinations:
    param_dict = dict(zip(keys, combo))
    param_dict['hidden_layer_sizes'] = str(param_dict['hidden_layer_sizes'])
    all_params.append(param_dict)

param_df = pd.DataFrame(all_params)
param_df.to_csv('hyperparameters_grid.csv', index=False)
print("- Saved hyperparameter grid to 'hyperparameters_grid.csv'")

grid_search = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=2
)

print("\nTraining models with grid search. This may take a few minutes...\n")
grid_search.fit(X_train_scaled, y_train)

print("\n" + "="*80)
print("RESULTS".center(80))
print("="*80)

print("\nBest Parameters Found:")
for param, value in grid_search.best_params_.items():
    print(f"- {param}: {value}")

best_params_df = pd.DataFrame([grid_search.best_params_])
best_params_df.to_csv('best_hyperparameters.csv', index=False)
print("- Saved best hyperparameters to 'best_hyperparameters.csv'")

print(f"\nBest Cross-Validation Score: {grid_search.best_score_:.4f}")

best_mlp = grid_search.best_estimator_
y_pred = best_mlp.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Set Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print("-"*60)
report = classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived'])
print(report)

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS".center(80))
print("="*80)

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("- Saved confusion matrix visualization to 'confusion_matrix.png'")

plt.figure(figsize=(10, 6))
plt.plot(best_mlp.loss_curve_)
plt.title('Learning Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('learning_curve.png')
print("- Saved learning curve visualization to 'learning_curve.png'")

plt.figure(figsize=(10, 6))
feature_importance = np.abs(best_mlp.coefs_[0]).sum(axis=1)
feature_importance = feature_importance / np.sum(feature_importance)
sns.barplot(x=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'], y=feature_importance)
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_importance.png')
print("- Saved feature importance visualization to 'feature_importance.png'")

print("\n" + "="*80)
print("TRAINING COMPLETE".center(80))
print("="*80)
print("\nThe best MLP model has been trained and evaluated.")