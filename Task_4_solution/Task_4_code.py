import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve, precision_score, recall_score

data = pd.read_csv('Breast Cancer Wisconsin (Diagnostic) Data Set.csv')

data = data.drop(columns=["id", "Unnamed: 32"])

data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0}) # malignant(M) means the tumours are cancerous

X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # transforms based on the mean and standard deviation of train set


log_regg_model = LogisticRegression(max_iter=1000, random_state=42)
log_regg_model.fit(X_train_scaled, y_train)


y_pred = log_regg_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',xticklabels=['Benign', 'Malignant'],yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()



y_prob = log_regg_model.predict_proba(X_test_scaled)[:, 1]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()


def evaluate_threshold(threshold):
    print(f"\n Evaluation at Threshold = {threshold}")
    y_pred_custom = (y_prob >= threshold).astype(int)
    
    conf_matrix = confusion_matrix(y_test, y_pred_custom)
    report = classification_report(y_test, y_pred_custom)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)

for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    evaluate_threshold(t)
