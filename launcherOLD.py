import customtkinter as ctk
import subprocess
import os
import sys

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("RMD20L - Table Tennis AI")
        self.geometry("500x600")
        self.resizable(False, False)
        
        self.active_processes = []
        self.create_ui()
    def create_ui(self):
        ctk.CTkLabel(self, text="Table Tennis AI", font=ctk.CTkFont(size=28, weight="bold")).pack(pady=(30, 4))
        ctk.CTkLabel(self, text="RMD20L SAGE Project  —  Albert Wu", font=ctk.CTkFont(size=13), text_color="gray").pack(pady=(0, 30))
        
        self.section_header("Data Collection")
        self.run_button(
            "Analyze Results - main.py",
            "View MediaPipe landmarks and YOLOv11n predictions",
            self.launch_main
        )
        self.run_button(
            "Run mediapipe and YOLO on images - image.py",
            "Draw the key landmarks or the balls location on an image",
            self.launch_image
        )
        
        self.section_header("Training Models")
        self.run_button(
            "Label Strokes - classify.py",
            "Label stroke segments to build the training dataset",
            self.launch_classifier
        )
        self.run_button(
            "Train the Classifier - train-classifier.py",
            "Train the random forest ML algorithm on different strokes",
            self.launch_train_classifier
        )
        
        
    def section_header(self, text):
        ctk.CTkLabel(self, text=text.upper(), font=ctk.CTkFont(size=11, weight="bold"), text_color="white").pack(anchor="w", padx=24, pady=(10, 2))
    
    def run_button(self, title, subtitle, command, disabled=False): # Only three actual arguments are title, subtitle, and command
        frame = ctk.CTkFrame(self, fg_color=("gray90", "gray17"), corner_radius=10)
        frame.pack(fill="x", padx=20, pady=4)
        
        text_frame = ctk.CTkFrame(frame, fg_color="transparent")
        text_frame.pack(side="left", fill="both", expand=True, padx=14, pady=10)
        
        ctk.CTkLabel(text_frame, text=title, font=ctk.CTkFont(size=14, weight="bold"), anchor="w").pack(anchor="w")
        ctk.CTkLabel(text_frame, text=subtitle, font=ctk.CTkFont(size=11), text_color="grey", anchor="w", wraplength=300).pack(anchor="w")
        
        button = ctk.CTkButton(
            frame,
            text="Run" if not disabled else "Soon", # Checks if currently exists
            width = 70,
            height = 36,
            state="disabled" if disabled else "normal",
            fg_color="grey40" if disabled else None,
            command=command
        )
        button.pack(side="right", padx=14, pady=10)
        

    def run_script(self, name):
        script_path = os.path.join(os.path.dirname(__file__), name)
        # if not os.path.exists(script_path):
        #     self.log(f"[!] Could not find {name}")
        #     return

        # self.log(f"[>] Launching {name}...")
        cwd = os.path.dirname(os.path.abspath(script_path))

        proc = subprocess.Popen( # Had to look online about how to run external programs as well as the tkinter program
            [sys.executable, script_path],
            cwd=cwd
        )
        self.active_processes.append(proc)
        
    def launch_classifier(self):
        self.run_script("classify.py")
    def launch_main(self):
        self.run_script("main.py")
    def launch_train_classifier(self):
        self.run_script("train-classifier.py")
    def launch_image(self):
        self.run_script("image.py")

app = App()
app.mainloop()