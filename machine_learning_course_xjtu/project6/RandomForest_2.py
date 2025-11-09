import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, RocCurveDisplay
from sklearn.preprocessing import FunctionTransformer

def add_features(X):
    X = X.copy()
    X["family_size"] = X["sibsp"] + X["parch"] + 1 # 交互特征
    # 票价对数变换
    X["fare_log"] = np.log1p(X["fare"]) # 减少极端值影响
    return X

#  1. 加载 Titanic 数据集（来自 seaborn）
df = sns.load_dataset("titanic").dropna(subset=["survived"])
print("✅ 数据加载成功，共", df.shape[0], "行")

#  2. 选择部分有代表性的特征 
X = df[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
y = df["survived"]

#  3. 区分特征类型  
num_cols = ["age", "sibsp", "parch", "fare"] # 数值型特征
cat_cols = ["pclass", "sex", "embarked"] # 类别型特征

#  4. 数据预处理 
feature_adder = FunctionTransformer(add_features) # 自定义特征添加器

# 先加新特征，再做数值/类别处理
preprocessor = Pipeline([
    ("add", feature_adder),
    ("col", ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols + ["family_size", "fare_log"]),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]))
])

#  5. 数据划分 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

#  6. 模型定义与参数网格 
models = {
    "DecisionTree": (DecisionTreeClassifier(random_state=42),
                     {"clf__max_depth": [None, 3, 5, 8, 10, 15]}), # [none, 5, 10] 三种深度选择,这个参数会传递给流水线中的分类器
    "RandomForest": (RandomForestClassifier(random_state=42, n_jobs=-1), # 使用所有CPU核心
                     {"clf__n_estimators": [50, 100, 200, 300, 500], "clf__max_depth": [None, 3, 5, 8, 10, 15]}) # 森林中树的数量和深度选择
}

results = []

#  7. 模型训练与评估 
for name, (model, grid) in models.items(): # grid 是参数网格
    pipe = Pipeline([("pre", preprocessor), ("clf", model)]) # 构建流水线
    search = GridSearchCV(pipe, grid, cv=5, scoring="f1_macro", n_jobs=-1) # 网格搜索，5折交叉验证，评估指标为 F1 分数
    search.fit(X_train, y_train)
    best = search.best_estimator_ # 最佳模型

    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n🌲 {name}")
    print("最佳参数:", search.best_params_)
    print(f"准确率: {acc:.4f}, F1分数: {f1:.4f}")

    # 混淆矩阵可视化
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # ROC 曲线（仅二分类）
    RocCurveDisplay.from_estimator(best, X_test, y_test)
    plt.title(f"{name} ROC Curve")
    plt.show()

    results.append({"Model": name, "Accuracy": acc, "F1": f1})

#  8. 模型性能对比 
res_df = pd.DataFrame(results)
print("\n模型对比结果：")
print(res_df)

sns.barplot(data=res_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Model", y="Score", hue="Metric")
plt.title("决策树 vs 随机森林 性能对比")
plt.ylim(0, 1)
plt.show()
