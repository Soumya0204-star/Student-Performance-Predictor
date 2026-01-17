# CREATOR: SOUMYA | GitHub: https://github.com/Soumya0204-star/Student-Performance-Predictor
"""
=======================================================
STUDENT PERFORMANCE PREDICTION - GUI VERSION
=======================================================
GUI application with real-time visualization
=======================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

class FuzzyPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Performance Prediction System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize fuzzy system
        self.setup_fuzzy_system()
        
        # Setup GUI
        self.setup_gui()
        
    def setup_fuzzy_system(self):
        """Setup the fuzzy logic system"""
        # Inputs
        self.attendance = ctrl.Antecedent(np.arange(0, 101, 1), 'attendance')
        self.internal = ctrl.Antecedent(np.arange(0, 101, 1), 'internal')
        self.assignment = ctrl.Antecedent(np.arange(0, 101, 1), 'assignment')
        
        # Output
        self.performance = ctrl.Consequent(np.arange(0, 101, 1), 'performance')
        
        # Membership functions
        self.attendance['low'] = fuzz.trimf(self.attendance.universe, [0, 0, 50])
        self.attendance['medium'] = fuzz.trimf(self.attendance.universe, [30, 50, 70])
        self.attendance['high'] = fuzz.trimf(self.attendance.universe, [60, 100, 100])
        
        self.internal['poor'] = fuzz.trimf(self.internal.universe, [0, 0, 50])
        self.internal['average'] = fuzz.trimf(self.internal.universe, [30, 50, 70])
        self.internal['good'] = fuzz.trimf(self.internal.universe, [60, 100, 100])
        
        self.assignment['low'] = fuzz.trimf(self.assignment.universe, [0, 0, 50])
        self.assignment['medium'] = fuzz.trimf(self.assignment.universe, [30, 50, 70])
        self.assignment['high'] = fuzz.trimf(self.assignment.universe, [60, 100, 100])
        
        self.performance['poor'] = fuzz.trimf(self.performance.universe, [0, 0, 40])
        self.performance['average'] = fuzz.trimf(self.performance.universe, [30, 50, 70])
        self.performance['good'] = fuzz.trimf(self.performance.universe, [60, 80, 90])
        self.performance['excellent'] = fuzz.trimf(self.performance.universe, [80, 100, 100])
        
        # Rules
        rule1 = ctrl.Rule(self.attendance['low'] & self.internal['poor'], self.performance['poor'])
        rule2 = ctrl.Rule(self.attendance['medium'] & self.internal['average'], self.performance['average'])
        rule3 = ctrl.Rule(self.attendance['high'] & self.assignment['high'], self.performance['excellent'])
        rule4 = ctrl.Rule(self.internal['good'] & self.assignment['medium'], self.performance['good'])
        rule5 = ctrl.Rule(self.attendance['high'] & self.internal['good'], self.performance['excellent'])
        rule6 = ctrl.Rule(self.attendance['medium'] & self.assignment['high'], self.performance['good'])
        
        self.performance_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
        self.performance_sim = ctrl.ControlSystemSimulation(self.performance_ctrl)
    
    def setup_gui(self):
        """Setup the GUI layout"""
        # Title
        title_label = tk.Label(
            self.root, 
            text="🎓 Student Performance Prediction System",
            font=('Arial', 20, 'bold'),
            bg='#4a6fa5',
            fg='white',
            padx=20,
            pady=10
        )
        title_label.pack(fill=tk.X)
        
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Inputs
        left_panel = ttk.LabelFrame(main_container, text="Student Details", padding=20)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Attendance input
        ttk.Label(left_panel, text="Attendance (%):", font=('Arial', 11)).grid(row=0, column=0, sticky=tk.W, pady=10)
        self.attendance_slider = tk.Scale(left_panel, from_=0, to=100, orient=tk.HORIZONTAL, length=300)
        self.attendance_slider.set(78)
        self.attendance_slider.grid(row=0, column=1, padx=10)
        
        self.attendance_entry = ttk.Entry(left_panel, width=10)
        self.attendance_entry.insert(0, "78")
        self.attendance_entry.grid(row=0, column=2, padx=10)
        
        # Internal marks input
        ttk.Label(left_panel, text="Internal Marks:", font=('Arial', 11)).grid(row=1, column=0, sticky=tk.W, pady=10)
        self.internal_slider = tk.Scale(left_panel, from_=0, to=100, orient=tk.HORIZONTAL, length=300)
        self.internal_slider.set(65)
        self.internal_slider.grid(row=1, column=1, padx=10)
        
        self.internal_entry = ttk.Entry(left_panel, width=10)
        self.internal_entry.insert(0, "65")
        self.internal_entry.grid(row=1, column=2, padx=10)
        
        # Assignment input
        ttk.Label(left_panel, text="Assignment Score:", font=('Arial', 11)).grid(row=2, column=0, sticky=tk.W, pady=10)
        self.assignment_slider = tk.Scale(left_panel, from_=0, to=100, orient=tk.HORIZONTAL, length=300)
        self.assignment_slider.set(72)
        self.assignment_slider.grid(row=2, column=1, padx=10)
        
        self.assignment_entry = ttk.Entry(left_panel, width=10)
        self.assignment_entry.insert(0, "72")
        self.assignment_entry.grid(row=2, column=2, padx=10)
        
        # Bind entries to sliders
        self.attendance_slider.config(command=lambda v: self.attendance_entry.delete(0, tk.END) or self.attendance_entry.insert(0, v))
        self.internal_slider.config(command=lambda v: self.internal_entry.delete(0, tk.END) or self.internal_entry.insert(0, v))
        self.assignment_slider.config(command=lambda v: self.assignment_entry.delete(0, tk.END) or self.assignment_entry.insert(0, v))
        
        self.attendance_entry.bind('<Return>', lambda e: self.update_slider(self.attendance_entry, self.attendance_slider))
        self.internal_entry.bind('<Return>', lambda e: self.update_slider(self.internal_entry, self.internal_slider))
        self.assignment_entry.bind('<Return>', lambda e: self.update_slider(self.assignment_entry, self.assignment_slider))
        
        # Control buttons
        button_frame = ttk.Frame(left_panel)
        button_frame.grid(row=3, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Show Rules", command=self.show_rules).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Visualize", command=self.visualize).pack(side=tk.LEFT, padx=5)
        
        # Right panel - Results
        right_panel = ttk.LabelFrame(main_container, text="Prediction Results", padding=20)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Result display
        self.result_text = scrolledtext.ScrolledText(right_panel, height=15, width=50, font=('Courier', 10))
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Performance meter
        meter_frame = ttk.Frame(right_panel)
        meter_frame.pack(fill=tk.X, pady=20)
        
        ttk.Label(meter_frame, text="Performance Score:", font=('Arial', 12)).pack()
        
        self.score_label = tk.Label(
            meter_frame, 
            text="0.00", 
            font=('Arial', 36, 'bold'),
            fg='#2e7d32'
        )
        self.score_label.pack()
        
        self.category_label = tk.Label(
            meter_frame,
            text="",
            font=('Arial', 16),
            fg='#1565c0'
        )
        self.category_label.pack()
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(meter_frame, length=400, mode='determinate')
        self.progress_bar.pack(pady=10)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Make initial prediction
        self.predict()
    
    def update_slider(self, entry, slider):
        """Update slider from entry"""
        try:
            value = float(entry.get())
            if 0 <= value <= 100:
                slider.set(value)
            else:
                messagebox.showerror("Error", "Value must be between 0 and 100")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
    
    def predict(self):
        """Make prediction based on inputs"""
        try:
            # Get values
            att = float(self.attendance_entry.get())
            internal = float(self.internal_entry.get())
            assign = float(self.assignment_entry.get())
            
            # Validate
            if not (0 <= att <= 100 and 0 <= internal <= 100 and 0 <= assign <= 100):
                messagebox.showerror("Error", "All values must be between 0 and 100")
                return
            
            # Compute prediction
            self.performance_sim.input['attendance'] = att
            self.performance_sim.input['internal'] = internal
            self.performance_sim.input['assignment'] = assign
            self.performance_sim.compute()
            
            score = self.performance_sim.output['performance']
            
            # Update display
            self.score_label.config(text=f"{score:.2f}")
            self.progress_bar['value'] = score
            
            # Determine category
            if score >= 80:
                category = "EXCELLENT 🎉"
                color = '#2e7d32'  # Green
            elif score >= 60:
                category = "GOOD 👍"
                color = '#7b1fa2'  # Purple
            elif score >= 40:
                category = "AVERAGE ⚖️"
                color = '#f57c00'  # Orange
            else:
                category = "NEEDS IMPROVEMENT 📈"
                color = '#c62828'  # Red
            
            self.category_label.config(text=category, fg=color)
            self.score_label.config(fg=color)
            
            # Show detailed results
            self.result_text.delete(1.0, tk.END)
            result = f"""
{'='*50}
STUDENT PERFORMANCE PREDICTION
{'='*50}

Input Values:
  • Attendance:      {att}%
  • Internal Marks:  {internal}/100
  • Assignment:      {assign}/100

Prediction Result:
  • Performance Score: {score:.2f}/100
  • Category: {category}

Fuzzy Logic Process:
  1. Fuzzification: Inputs converted to fuzzy sets
  2. Rule Evaluation: 6 fuzzy rules applied
  3. Defuzzification: Converted to crisp score

Recommendations:
"""
            if att < 70:
                result += "  • Improve attendance for better results\n"
            if internal < 50:
                result += "  • Focus on internal test preparation\n"
            if assign < 50:
                result += "  • Spend more time on assignments\n"
            if score >= 80:
                result += "  • Continue current study habits\n"
            
            self.result_text.insert(1.0, result)
            
            self.status_bar.config(text=f"Prediction complete: Score = {score:.2f}")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def clear(self):
        """Clear all inputs"""
        self.attendance_slider.set(0)
        self.internal_slider.set(0)
        self.assignment_slider.set(0)
        self.attendance_entry.delete(0, tk.END)
        self.internal_entry.delete(0, tk.END)
        self.assignment_entry.delete(0, tk.END)
        self.attendance_entry.insert(0, "0")
        self.internal_entry.insert(0, "0")
        self.assignment_entry.insert(0, "0")
        self.result_text.delete(1.0, tk.END)
        self.score_label.config(text="0.00")
        self.category_label.config(text="")
        self.progress_bar['value'] = 0
        self.status_bar.config(text="Cleared all inputs")
    
    def show_rules(self):
        """Show fuzzy rules"""
        rules_window = tk.Toplevel(self.root)
        rules_window.title("Fuzzy Rules")
        rules_window.geometry("500x400")
        
        rules_text = """
FUZZY RULES USED:

1. IF attendance is low AND internal is poor
   THEN performance is poor

2. IF attendance is medium AND internal is average
   THEN performance is average

3. IF attendance is high AND assignment is high
   THEN performance is excellent

4. IF internal is good AND assignment is medium
   THEN performance is good

5. IF attendance is high AND internal is good
   THEN performance is excellent

6. IF attendance is medium AND assignment is high
   THEN performance is good

MEMBERSHIP FUNCTIONS:

Attendance:
  • Low: 0-50%
  • Medium: 30-70%
  • High: 60-100%

Internal Marks:
  • Poor: 0-50
  • Average: 30-70
  • Good: 60-100

Assignment:
  • Low: 0-50
  • Medium: 30-70
  • High: 60-100

Performance:
  • Poor: 0-40
  • Average: 30-70
  • Good: 60-90
  • Excellent: 80-100
"""
        
        text_widget = scrolledtext.ScrolledText(rules_window, width=60, height=20)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(1.0, rules_text)
        text_widget.config(state=tk.DISABLED)
    
    def visualize(self):
        """Show membership function visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Plot attendance
        ax = axes[0, 0]
        for term in ['low', 'medium', 'high']:
            ax.plot(self.attendance.universe, self.attendance[term].mf, label=term)
        ax.set_title('Attendance Membership')
        ax.legend()
        ax.grid(True)
        
        # Plot internal
        ax = axes[0, 1]
        for term in ['poor', 'average', 'good']:
            ax.plot(self.internal.universe, self.internal[term].mf, label=term)
        ax.set_title('Internal Marks Membership')
        ax.legend()
        ax.grid(True)
        
        # Plot assignment
        ax = axes[1, 0]
        for term in ['low', 'medium', 'high']:
            ax.plot(self.assignment.universe, self.assignment[term].mf, label=term)
        ax.set_title('Assignment Membership')
        ax.legend()
        ax.grid(True)
        
        # Plot performance
        ax = axes[1, 1]
        for term in ['poor', 'average', 'good', 'excellent']:
            ax.plot(self.performance.universe, self.performance[term].mf, label=term)
        ax.set_title('Performance Membership')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        # Create Tkinter window for plot
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Membership Function Visualization")
        
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = FuzzyPredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()