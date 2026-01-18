# Add at the top of your run_demo.py
import sys
import time
from colorama import init, Fore, Back, Style
init()

class DemoPresenter:
    def __init__(self):
        self.width = 60
        
    def print_header(self):
        print(Fore.CYAN + "╔" + "═" * self.width + "╗")
        print("║" + " " * 20 + "STUDENT PERFORMANCE PREDICTOR" + " " * 20 + "║")
        print("║" + " " * 18 + "FUZZY LOGIC INTELLIGENCE SYSTEM" + " " * 17 + "║")
        print("╚" + "═" * self.width + "╝" + Style.RESET_ALL)
    
    def print_step(self, step_num, description):
        print(Fore.YELLOW + f"\n[{step_num}] " + Fore.WHITE + f"{description}" + Style.RESET_ALL)
        time.sleep(0.5)
    
    def print_result(self, title, value, color=Fore.GREEN):
        print(f"{Fore.WHITE}{title}: {color}{value}{Style.RESET_ALL}")
    
    def progress_bar(self, duration=2, steps=20):
        for i in range(steps + 1):
            percent = i * 100 // steps
            bar = "█" * i + "░" * (steps - i)
            sys.stdout.write(f"\r[{bar}] {percent}%")
            sys.stdout.flush()
            time.sleep(duration / steps)
        print()

# In your main function, use it like this:
presenter = DemoPresenter()
presenter.print_header()

presenter.print_step("1", "Loading Fuzzy Logic System")
presenter.progress_bar(1.5)

presenter.print_step("2", "Generating Synthetic Dataset (50 students)")
# Your dataset generation code here

presenter.print_step("3", "Applying Fuzzy Inference Rules")
presenter.progress_bar(2)

presenter.print_step("4", "Generating Performance Predictions")
# Your prediction code here

# Show results in a table format
print(Fore.MAGENTA + "\n" + "═" * 60)
print("PREDICTION RESULTS".center(60))
print("═" * 60 + Style.RESET_ALL)
print(f"{'Student ID':<12} {'Attendance':<12} {'Assignments':<12} {'Prediction':<15} {'Confidence':<10}")
print("-" * 60)

# Add sample predictions
samples = [
    ("S001", 95, 88, "EXCELLENT", "92%"),
    ("S002", 75, 82, "GOOD", "78%"),
    ("S003", 45, 60, "NEEDS IMPROVEMENT", "65%"),
    ("S004", 88, 92, "VERY GOOD", "85%"),
    ("S005", 30, 40, "AT RISK", "45%"),
]

for sid, att, assign, pred, conf in samples:
    color = Fore.GREEN if "EXCELLENT" in pred or "GOOD" in pred else Fore.YELLOW if "NEEDS" in pred else Fore.RED
    print(f"{sid:<12} {att:<12} {assign:<12} {color}{pred:<15}{Style.RESET_ALL} {conf:<10}")

print(Fore.CYAN + "\n" + "═" * 60)
print("ANALYSIS COMPLETE".center(60))
print("═" * 60 + Style.RESET_ALL)

# Install colorama first: pip install colorama