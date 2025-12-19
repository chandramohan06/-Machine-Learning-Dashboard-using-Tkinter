
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

import warnings
warnings.filterwarnings("ignore")

class PredictiveDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Predictive Analysis & Visualization Dashboard")
        self.root.geometry("1450x900")
        self.root.configure(bg="#121212")

        self.df = None
        self.model = None
        self.problem_type = None
        self.encoder = None
        self.scaler = None
        self.predictions = None

        self.colors = {
            'bg': '#121212',
            'fg': '#E0E0E0',
            'accent': '#00B4D8',
            'graph_bg': '#1E1E1E',
            'grid': '#3A3A3A'
        }

        # --- Top control bar ---
        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, pady=8)

        ttk.Button(top, text="📂 Load CSV", command=self.load_dataset).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="🧾 Summary", command=self.show_summary).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="🔥 Correlation Heatmap", command=self.plot_heatmap).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="📊 Histogram", command=self.plot_histogram).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="📦 Boxplot", command=self.plot_boxplot).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="🎯 Scatter Plot", command=self.plot_scatter).pack(side=tk.LEFT, padx=5)

        ttk.Label(top, text="Target:", background=self.colors['bg'], foreground=self.colors['fg']).pack(side=tk.LEFT, padx=(20, 2))
        self.target_combo = ttk.Combobox(top, state="readonly")
        self.target_combo.pack(side=tk.LEFT)
        self.target_combo.bind("<<ComboboxSelected>>", self.on_target_selected)

        ttk.Label(top, text="Model:", background=self.colors['bg'], foreground=self.colors['fg']).pack(side=tk.LEFT, padx=(20, 2))
        self.model_combo = ttk.Combobox(top, state="readonly", width=25,
            values=[
                "Auto (Recommended)",
                "Linear Regression",
                "Random Forest (Reg)",
                "Decision Tree (Reg)",
                "SVM (Reg)",
                "KNN (Reg)",
                "Logistic Regression",
                "Random Forest (Class)",
                "Decision Tree (Class)",
                "SVM (Class)",
                "KNN (Class)"
            ])
        self.model_combo.current(0)
        self.model_combo.pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="🚀 Train Model", command=self.train_model).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="📈 Feature Importance", command=self.plot_feature_importance).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="💾 Export Predictions", command=self.export_predictions).pack(side=tk.RIGHT, padx=8)

        # --- Left: Dataset & Logs ---
        left = ttk.Frame(self.root)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        tk.Label(left, text="Dataset Preview", fg=self.colors['accent'], bg=self.colors['bg'], font=("Arial", 12, "bold")).pack(anchor=tk.W)
        self.preview_box = tk.Text(left, width=60, height=20, bg=self.colors['graph_bg'], fg=self.colors['fg'], font=("Consolas", 10))
        self.preview_box.pack(pady=5)

        tk.Label(left, text="Logs", fg=self.colors['accent'], bg=self.colors['bg'], font=("Arial", 12, "bold")).pack(anchor=tk.W)
        self.log_box = tk.Text(left, width=60, height=15, bg=self.colors['graph_bg'], fg=self.colors['fg'], font=("Consolas", 10))
        self.log_box.pack(pady=5)

        # --- Right: Visualization area ---
        right = ttk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        self.ax.set_facecolor(self.colors['graph_bg'])
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        sns.set_style("darkgrid")

        self.log("✅ Welcome! Load a CSV to get started.")

    def log(self, msg):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)

    # ---------- Load Dataset ----------
    def load_dataset(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path: return
        try:
            self.df = pd.read_csv(path)
            self.preview_box.delete(1.0, tk.END)
            self.preview_box.insert(tk.END, self.df.head(20).to_string())
            self.target_combo['values'] = list(self.df.columns)
            self.target_combo.current(0)
            self.on_target_selected()
            self.log(f"Loaded dataset: {path} (shape={self.df.shape})")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_summary(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        buf = []
        buf.append(str(self.df.describe(include='all')))
        buf.append("\nMissing values:\n" + str(self.df.isnull().sum()))
        self.preview_box.delete(1.0, tk.END)
        self.preview_box.insert(tk.END, "\n\n".join(buf))
        self.log("Displayed dataset summary.")

    def on_target_selected(self, event=None):
        if self.df is None: return
        target = self.target_combo.get()
        if pd.api.types.is_numeric_dtype(self.df[target]):
            if self.df[target].nunique() <= 10:
                self.problem_type = "classification"
            else:
                self.problem_type = "regression"
        else:
            self.problem_type = "classification"
        self.log(f"Target selected: {target} — Detected problem type: {self.problem_type}")

    # ---------- Data Prep ----------
    def prepare_data(self):
        df = self.df.copy()
        target = self.target_combo.get()
        df = df.dropna(subset=[target])

        X = df.drop(columns=[target])
        y = df[target]

        X = pd.get_dummies(X, drop_first=True)
        if self.problem_type == "classification":
            if not pd.api.types.is_numeric_dtype(y):
                encoder = LabelEncoder()
                y = encoder.fit_transform(y)

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        return X, y

    # ---------- Model Selection ----------
    def select_model(self):
        name = self.model_combo.get().lower()
        if "linear regression" in name: return LinearRegression()
        if "random forest" in name and "reg" in name: return RandomForestRegressor(random_state=42)
        if "decision tree" in name and "reg" in name: return DecisionTreeRegressor(random_state=42)
        if "svm" in name and "reg" in name: return SVR()
        if "knn" in name and "reg" in name: return KNeighborsRegressor()
        if "logistic" in name: return LogisticRegression(max_iter=1000)
        if "random forest" in name and "class" in name: return RandomForestClassifier(random_state=42)
        if "decision tree" in name and "class" in name: return DecisionTreeClassifier(random_state=42)
        if "svm" in name and "class" in name: return SVC(probability=True)
        if "knn" in name and "class" in name: return KNeighborsClassifier()
        return RandomForestClassifier() if self.problem_type == "classification" else RandomForestRegressor()

    # ---------- Train ----------
    def train_model(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Load dataset first!")
            return
        X, y = self.prepare_data()
        model = self.select_model()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.model = model
        self.y_test, self.y_pred = y_test, y_pred

        if self.problem_type == "regression":
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            self.log(f"✅ Regression Results — R²={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")
        else:
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            self.log(f"✅ Classification Results — Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")

        self.plot_results()

    # ---------- Visualization ----------
    def plot_heatmap(self):
        if self.df is None: return
        num = self.df.select_dtypes(include=[np.number])
        plt.figure(figsize=(8,6))
        sns.heatmap(num.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()

    def plot_histogram(self):
        if self.df is None: return
        cols = list(self.df.select_dtypes(include=[np.number]).columns)
        if not cols:
            messagebox.showinfo("No Numeric Data", "No numeric columns to plot.")
            return
        col = tk.simpledialog.askstring("Histogram", f"Enter column name:\nAvailable: {', '.join(cols)}")
        if col not in cols:
            messagebox.showerror("Invalid", "Invalid column name.")
            return
        plt.figure(figsize=(7,5))
        sns.histplot(self.df[col], kde=True, bins=20, color='skyblue')
        plt.title(f"Histogram of {col}")
        plt.show()

    def plot_boxplot(self):
        if self.df is None: return
        cols = list(self.df.select_dtypes(include=[np.number]).columns)
        col = tk.simpledialog.askstring("Boxplot", f"Enter column name:\nAvailable: {', '.join(cols)}")
        if col not in cols:
            messagebox.showerror("Invalid", "Invalid column name.")
            return
        plt.figure(figsize=(7,5))
        sns.boxplot(y=self.df[col], color='gold')
        plt.title(f"Boxplot of {col}")
        plt.show()

    def plot_scatter(self):
        if self.df is None: return
        cols = list(self.df.select_dtypes(include=[np.number]).columns)
        x = tk.simpledialog.askstring("Scatter Plot", f"Enter X-axis column:\nAvailable: {', '.join(cols)}")
        y = tk.simpledialog.askstring("Scatter Plot", f"Enter Y-axis column:\nAvailable: {', '.join(cols)}")
        if x not in cols or y not in cols:
            messagebox.showerror("Invalid", "Invalid column name(s).")
            return
        plt.figure(figsize=(7,5))
        sns.scatterplot(data=self.df, x=x, y=y, color='lightgreen')
        plt.title(f"{y} vs {x}")
        plt.show()

    def plot_results(self):
        if self.model is None:
            messagebox.showinfo("Train first", "Please train a model first.")
            return
        self.ax.clear()
        if self.problem_type == "regression":
            self.ax.scatter(self.y_test, self.y_pred, color=self.colors['accent'], alpha=0.7)
            self.ax.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], 'r--')
            self.ax.set_title("Actual vs Predicted")
            self.ax.set_xlabel("Actual")
            self.ax.set_ylabel("Predicted")
        else:
            cm = confusion_matrix(self.y_test, self.y_pred)
            sns.heatmap(cm, annot=True, fmt="d", ax=self.ax, cmap="mako")
            self.ax.set_title("Confusion Matrix")
        self.canvas.draw()

    def plot_feature_importance(self):
        if self.model is None:
            messagebox.showinfo("Train first", "Train model first.")
            return
        if hasattr(self.model, "feature_importances_"):
            fi = self.model.feature_importances_
            features = self.df.drop(columns=[self.target_combo.get()]).columns
            imp = pd.Series(fi, index=features).sort_values(ascending=False)[:20]
            plt.figure(figsize=(7,5))
            sns.barplot(x=imp.values, y=imp.index, palette="viridis")
            plt.title("Top Feature Importances")
            plt.show()
        else:
            messagebox.showinfo("N/A", "Feature importance not supported for this model.")

    # ---------- Export ----------
    def export_predictions(self):
        if self.model is None:
            messagebox.showwarning("No Predictions", "Train model first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv")
        if not path: return
        df_export = pd.DataFrame({"Actual": self.y_test, "Predicted": self.y_pred})
        df_export.to_csv(path, index=False)
        self.log(f"Exported predictions to {path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictiveDashboard(root)
    root.mainloop()
