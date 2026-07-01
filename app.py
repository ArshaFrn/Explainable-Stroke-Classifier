import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


class StrokeTriageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VitalSeconds | Clinical Stroke Triage")
        self.root.geometry("600x800")
        self.root.minsize(600, 800)
        self.root.maxsize(600, 800)
        self.root.configure(bg="#f4f8ff")
        self.root.option_add("*Font", "Arial 10")

        self.COLORS = {
            "primary": "#0f4c81",
            "accent": "#3b82f6",
            "danger": "#ef4444",
            "success": "#22c55e",
            "bg": "#f4f8ff",
            "card": "#ffffff",
            "text": "#0f172a",
            "muted": "#64748b",
            "line": "#dbeafe",
            "soft": "#f8fbff"
        }

        self.loadArtifacts()
        if hasattr(self, "model"):
            self.setupStyles()
            self.setupUI()

    def loadArtifacts(self):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.model = joblib.load(os.path.join(base_dir, "models", "winner_model.joblib"))
            self.scaler = joblib.load(os.path.join(base_dir, "models", "scaler.joblib"))
            self.encoders = joblib.load(os.path.join(base_dir, "models", "encoders.joblib"))
            self.threshold = joblib.load(os.path.join(base_dir, "models", "best_threshold.joblib"))
        except Exception as e:
            messagebox.showerror("System Error", f"Model components missing: {e}")
            self.root.destroy()

    def setupStyles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground="white", background="white", bordercolor="#cbd5e1")
        style.map("TCombobox", fieldbackground=[("readonly", "white")], background=[("readonly", "white")])

    def setupUI(self):
        header = tk.Frame(self.root, bg="#0f172a", height=92)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(
            header,
            text="VITAL SECONDS",
            font=("Arial", 18, "bold"),
            fg="white",
            bg="#0f172a"
        ).pack(anchor=tk.W, padx=24, pady=(20, 2))

        tk.Label(
            header,
            text="AI-powered stroke triage assessment",
            font=("Arial", 10),
            fg="#dbeafe",
            bg="#0f172a"
        ).pack(anchor=tk.W, padx=24)

        main = tk.Frame(self.root, bg=self.COLORS["bg"])
        main.pack(fill=tk.BOTH, expand=True)

        card = tk.Frame(
            main,
            bg=self.COLORS["card"],
            padx=20,
            pady=20,
            highlightthickness=1,
            highlightbackground="#e2e8f0"
        )
        card.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        intro = tk.Frame(card, bg=self.COLORS["card"])
        intro.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            intro,
            text="Stroke Risk Assessment",
            font=("Arial", 15, "bold"),
            fg=self.COLORS["text"],
            bg=self.COLORS["card"]
        ).pack(anchor=tk.W)

        tk.Label(
            intro,
            text="Complete the clinical markers below for a fast safety review.",
            font=("Arial", 9),
            fg=self.COLORS["muted"],
            bg=self.COLORS["card"]
        ).pack(anchor=tk.W, pady=(2, 0))

        self.inputs = {}
        fields = [
            ("Gender", ["Male", "Female"]),
            ("Age (Years)", "entry"),
            ("Hypertension", ["No", "Yes"]),
            ("Heart Disease", ["No", "Yes"]),
            ("Glucose Level (mg/dL)", "entry"),
            ("Body Mass Index (BMI)", "entry"),
            ("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
        ]

        form = tk.Frame(card, bg=self.COLORS["card"])
        form.pack(fill=tk.X, pady=(8, 6))

        for label, fType in fields:
            f_frame = tk.Frame(form, bg=self.COLORS["card"], pady=4)
            f_frame.pack(fill=tk.X)

            tk.Label(
                f_frame,
                text=label,
                font=("Arial", 10, "bold"),
                fg=self.COLORS["text"],
                bg=self.COLORS["card"],
                anchor=tk.W
            ).pack(side=tk.LEFT)

            var = tk.StringVar()
            if fType == "entry":
                widget = tk.Entry(
                    f_frame,
                    textvariable=var,
                    font=("Arial", 10),
                    bg=self.COLORS["soft"],
                    relief=tk.FLAT,
                    bd=1,
                    highlightthickness=1,
                    highlightbackground="#cbd5e1",
                    highlightcolor=self.COLORS["accent"],
                    width=24
                )
            else:
                var.set(fType[0])
                widget = ttk.Combobox(
                    f_frame,
                    textvariable=var,
                    values=fType,
                    state="readonly",
                    width=24
                )

            widget.pack(side=tk.RIGHT, ipady=4)
            self.inputs[label] = var

        self.predictBtn = tk.Button(
            card,
            text="ASSESS PATIENT RISK",
            command=self.predictRisk,
            bg=self.COLORS["primary"],
            fg="white",
            font=("Arial", 10, "bold"),
            bd=0,
            cursor="hand2",
            pady=10,
            activebackground=self.COLORS["accent"],
            activeforeground="white"
        )
        self.predictBtn.pack(fill=tk.X, pady=(18, 8))

        tk.Button(
            card,
            text="Clear Data",
            command=self.clearForm,
            fg=self.COLORS["muted"],
            bg=self.COLORS["card"],
            font=("Arial", 9),
            bd=0,
            cursor="hand2",
            activebackground=self.COLORS["card"]
        ).pack()

        self.resultFrame = tk.Frame(
            card,
            bg="#f8fbff",
            padx=14,
            pady=14,
            highlightthickness=1,
            highlightbackground="#dbeafe"
        )
        self.resultFrame.pack(fill=tk.X, pady=(16, 0))

        self.resTitle = tk.Label(
            self.resultFrame,
            text="AWAITING INPUT",
            font=("Arial", 12, "bold"),
            bg="#f8fbff",
            fg="#94a3b8"
        )
        self.resTitle.pack()

        self.resProb = tk.Label(
            self.resultFrame,
            text="---",
            font=("Arial", 10),
            bg="#f8fbff",
            fg=self.COLORS["muted"]
        )
        self.resProb.pack(pady=(5, 0))

    def clearForm(self):
        for var in self.inputs.values():
            var.set("")
        self.resTitle.config(text="AWAITING INPUT", fg="#94a3b8")
        self.resProb.config(text="---")
        self.resultFrame.config(bg="#f8fbff")

    def predictRisk(self):
        try:
            raw = {
                "gender": self.inputs["Gender"].get(),
                "age": float(self.inputs["Age (Years)"].get()),
                "hypertension": 1 if self.inputs["Hypertension"].get() == "Yes" else 0,
                "heart_disease": 1 if self.inputs["Heart Disease"].get() == "Yes" else 0,
                "avg_glucose_level": float(self.inputs["Glucose Level (mg/dL)"].get()),
                "bmi": float(self.inputs["Body Mass Index (BMI)"].get()),
                "smoking_status": self.inputs["Smoking Status"].get()
            }

            raw["ageHypertensionInteraction"] = raw["age"] * raw["hypertension"]

            for col, le in self.encoders.items():
                if col in raw:
                    raw[col] = le.transform([raw[col]])[0]

            cols = list(self.scaler.feature_names_in_)
            vec = pd.DataFrame([raw])[cols]

            scaled = self.scaler.transform(vec)
            scaled_df = pd.DataFrame(scaled, columns=cols)
            prob = self.model.predict_proba(scaled_df)[0, 1]
            high = prob >= self.threshold

            status = "CRITICAL: HIGH RISK" if high else "NORMAL: LOW RISK"
            color = self.COLORS["danger"] if high else self.COLORS["success"]
            bg_color = "#fef2f2" if high else "#f0fdf4"

            self.resTitle.config(text=status, fg=color)
            self.resProb.config(text=f"Stroke Probability: {prob:.2%}", fg=self.COLORS["text"])
            self.resultFrame.config(bg=bg_color)

        except ValueError:
            messagebox.showwarning("Validation Error", "Please enter valid numeric values for Age, Glucose, and BMI.")
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = StrokeTriageApp(root)
    root.mainloop()