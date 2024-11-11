import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DataAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("University Data Analysis Application")
        
        # Define attributes for storing data
        self.activity_log = None
        self.component_codes = None
        self.user_log = None
        self.merged_data = None
        self.interaction_counts = None
        
        # Load and Clean Data Buttons
        load_activity_btn = tk.Button(root, text="Load Activity Log", command=self.load_activity_log)
        load_component_btn = tk.Button(root, text="Load Component Codes", command=self.load_component_codes)
        load_user_log_btn = tk.Button(root, text="Load User Log", command=self.load_user_log)
        
        # Place buttons on the GUI
        load_activity_btn.pack()
        load_component_btn.pack()
        load_user_log_btn.pack()
        
        # Analysis Buttons
        clean_data_btn = tk.Button(root, text="Clean and Merge Data", command=self.clean_and_merge_data)
        stats_btn = tk.Button(root, text="Generate Statistics", command=self.generate_statistics)
        corr_btn = tk.Button(root, text="Correlation Analysis", command=self.correlation_analysis)
        
        # Place analysis buttons
        clean_data_btn.pack()
        stats_btn.pack()
        corr_btn.pack()

    def load_activity_log(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.activity_log = pd.read_csv(file_path)
            self.activity_log.rename(columns={"User Full Name *Anonymized": "User_ID"}, inplace=True)
            messagebox.showinfo("Data Load", "Activity Log loaded successfully.")
    
    def load_component_codes(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.component_codes = pd.read_csv(file_path)
            messagebox.showinfo("Data Load", "Component Codes loaded successfully.")

    def load_user_log(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.user_log = pd.read_csv(file_path)
            self.user_log.rename(columns={"User Full Name *Anonymized": "User_ID"}, inplace=True)
            self.user_log['Date'] = pd.to_datetime(self.user_log['Date'], format='%d/%m/%Y %H:%M')
            self.user_log['Month'] = self.user_log['Date'].dt.to_period('M')
            messagebox.showinfo("Data Load", "User Log loaded successfully.")
            
    def clean_and_merge_data(self):
        if self.activity_log is not None and self.component_codes is not None:
            self.activity_log = self.activity_log[~self.activity_log['Component'].isin(['System', 'Folder'])]
            self.merged_data = pd.merge(self.activity_log, self.component_codes, on="Component", how="left")
            messagebox.showinfo("Data Processing", "Data cleaned and merged successfully.")
        else:
            messagebox.showwarning("Data Missing", "Please load all data files first.")
    
    def generate_statistics(self):
        if self.merged_data is not None:
            self.interaction_counts = self.merged_data.pivot_table(index="User_ID", columns="Component", values="Action", aggfunc="count", fill_value=0)
            mean_values = self.interaction_counts.mean()
            median_values = self.interaction_counts.median()
            mode_values = self.interaction_counts.mode().iloc[0]
            messagebox.showinfo("Statistics", f"Mean:\n{mean_values}\n\nMedian:\n{median_values}\n\nMode:\n{mode_values}")
        else:
            messagebox.showwarning("Data Missing", "Please clean and merge the data first.")
    
    def correlation_analysis(self):
        if self.user_log is not None and self.interaction_counts is not None:
            user_monthly_data = self.user_log[['User_ID', 'Month']].merge(self.interaction_counts, on="User_ID", how="left")
            selected_columns = ['Assignment', 'Quiz', 'Lecture', 'Book', 'Project', 'Course']
            correlation_data = user_monthly_data[selected_columns].corr()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(correlation_data, annot=True, cmap="coolwarm", ax=ax)
            plt.title("Correlation Analysis")
            
            canvas = FigureCanvasTkAgg(fig, master=self.root)
            canvas.draw()
            canvas.get_tk_widget().pack()
        else:
            messagebox.showwarning("Data Missing", "Please clean and merge the data first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalysisApp(root)
    root.mainloop()
