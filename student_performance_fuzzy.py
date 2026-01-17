# CREATOR: SOUMYA | GitHub: https://github.com/Soumya0204-star/Student-Performance-Predictor
"""
=======================================================
STUDENT PERFORMANCE PREDICTION SYSTEM USING FUZZY LOGIC
=======================================================
Author: Your Name
Description: A complete fuzzy logic system to predict student performance
             based on attendance, internal marks, and assignment scores.
=======================================================
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

class StudentPerformancePredictor:
    """Complete fuzzy logic system for student performance prediction"""
    
    def __init__(self):
        """Initialize the fuzzy logic system with comprehensive rules"""
        self.setup_fuzzy_system()
        self.setup_rules()
        
    def setup_fuzzy_system(self):
        """Setup all fuzzy variables and membership functions"""
        
        # ---------- Define Input Variables ----------
        self.attendance = ctrl.Antecedent(np.arange(0, 101, 1), 'attendance')
        self.internal = ctrl.Antecedent(np.arange(0, 101, 1), 'internal')
        self.assignment = ctrl.Antecedent(np.arange(0, 101, 1), 'assignment')
        
        # ---------- Define Output Variable ----------
        self.performance = ctrl.Consequent(np.arange(0, 101, 1), 'performance')
        
        # ---------- Define Membership Functions ----------
        # Attendance: Trapezoidal for smoother transitions
        self.attendance['very_low'] = fuzz.trapmf(self.attendance.universe, [0, 0, 20, 35])
        self.attendance['low'] = fuzz.trimf(self.attendance.universe, [30, 40, 50])
        self.attendance['medium'] = fuzz.trimf(self.attendance.universe, [45, 60, 75])
        self.attendance['high'] = fuzz.trimf(self.attendance.universe, [70, 80, 90])
        self.attendance['excellent'] = fuzz.trapmf(self.attendance.universe, [85, 90, 100, 100])
        
        # Internal Marks
        self.internal['very_poor'] = fuzz.trapmf(self.internal.universe, [0, 0, 20, 35])
        self.internal['poor'] = fuzz.trimf(self.internal.universe, [30, 40, 50])
        self.internal['average'] = fuzz.trimf(self.internal.universe, [45, 55, 65])
        self.internal['good'] = fuzz.trimf(self.internal.universe, [60, 70, 80])
        self.internal['excellent'] = fuzz.trapmf(self.internal.universe, [75, 85, 100, 100])
        
        # Assignment Scores
        self.assignment['very_low'] = fuzz.trapmf(self.assignment.universe, [0, 0, 20, 35])
        self.assignment['low'] = fuzz.trimf(self.assignment.universe, [30, 40, 50])
        self.assignment['medium'] = fuzz.trimf(self.assignment.universe, [45, 60, 75])
        self.assignment['high'] = fuzz.trimf(self.assignment.universe, [70, 80, 90])
        self.assignment['excellent'] = fuzz.trapmf(self.assignment.universe, [85, 90, 100, 100])
        
        # Performance Output (using trapezoidal for better defuzzification)
        self.performance['poor'] = fuzz.trapmf(self.performance.universe, [0, 0, 30, 45])
        self.performance['average'] = fuzz.trimf(self.performance.universe, [40, 50, 65])
        self.performance['good'] = fuzz.trimf(self.performance.universe, [60, 72, 85])
        self.performance['excellent'] = fuzz.trapmf(self.performance.universe, [80, 88, 100, 100])
        
        # Optional: Show membership functions (commented for CLI)
        # self.plot_membership_functions()
    
    def setup_rules(self):
        """Setup comprehensive rule base"""
        
        # ---------- Rule Set ----------
        rules = []
        
        # Rule Group 1: Very Low Performance Scenarios
        rules.append(ctrl.Rule(
            self.attendance['very_low'] & self.internal['very_poor'] & self.assignment['very_low'],
            self.performance['poor']
        ))
        rules.append(ctrl.Rule(
            self.attendance['very_low'] & self.internal['very_poor'],
            self.performance['poor']
        ))
        
        # Rule Group 2: Low Performance
        rules.append(ctrl.Rule(
            self.attendance['low'] & self.internal['poor'],
            self.performance['poor']
        ))
        rules.append(ctrl.Rule(
            self.assignment['low'] & self.internal['poor'],
            self.performance['poor']
        ))
        
        # Rule Group 3: Average Performance
        rules.append(ctrl.Rule(
            self.attendance['medium'] & self.internal['average'] & self.assignment['medium'],
            self.performance['average']
        ))
        rules.append(ctrl.Rule(
            self.attendance['medium'] & self.internal['average'],
            self.performance['average']
        ))
        rules.append(ctrl.Rule(
            self.attendance['low'] & self.internal['good'] & self.assignment['medium'],
            self.performance['average']
        ))
        
        # Rule Group 4: Good Performance
        rules.append(ctrl.Rule(
            self.attendance['high'] & self.internal['average'] & self.assignment['medium'],
            self.performance['good']
        ))
        rules.append(ctrl.Rule(
            self.attendance['medium'] & self.internal['good'] & self.assignment['high'],
            self.performance['good']
        ))
        rules.append(ctrl.Rule(
            self.attendance['high'] & self.internal['good'],
            self.performance['good']
        ))
        rules.append(ctrl.Rule(
            self.assignment['high'] & self.internal['good'],
            self.performance['good']
        ))
        
        # Rule Group 5: Excellent Performance
        rules.append(ctrl.Rule(
            self.attendance['excellent'] & self.internal['excellent'] & self.assignment['excellent'],
            self.performance['excellent']
        ))
        rules.append(ctrl.Rule(
            self.attendance['excellent'] & self.internal['good'],
            self.performance['excellent']
        ))
        rules.append(ctrl.Rule(
            self.attendance['high'] & self.internal['excellent'],
            self.performance['excellent']
        ))
        rules.append(ctrl.Rule(
            self.assignment['excellent'] & self.internal['good'],
            self.performance['excellent']
        ))
        rules.append(ctrl.Rule(
            self.attendance['high'] & self.assignment['high'],
            self.performance['excellent']
        ))
        
        # Rule Group 6: Mixed but Positive
        rules.append(ctrl.Rule(
            self.attendance['high'] & self.internal['average'] & self.assignment['high'],
            self.performance['good']
        ))
        rules.append(ctrl.Rule(
            self.attendance['medium'] & self.internal['good'] & self.assignment['excellent'],
            self.performance['good']
        ))
        
        # ---------- Create Control System ----------
        self.performance_ctrl = ctrl.ControlSystem(rules)
        self.performance_sim = ctrl.ControlSystemSimulation(self.performance_ctrl)
    
    def predict(self, attendance, internal, assignment):
        """Predict performance for given inputs"""
        
        # Validate inputs
        if not (0 <= attendance <= 100):
            raise ValueError("Attendance must be between 0 and 100")
        if not (0 <= internal <= 100):
            raise ValueError("Internal marks must be between 0 and 100")
        if not (0 <= assignment <= 100):
            raise ValueError("Assignment score must be between 0 and 100")
        
        # Set inputs
        self.performance_sim.input['attendance'] = attendance
        self.performance_sim.input['internal'] = internal
        self.performance_sim.input['assignment'] = assignment
        
        # Compute
        try:
            self.performance_sim.compute()
            score = self.performance_sim.output['performance']
            return score
        except Exception as e:
            raise RuntimeError(f"Error in fuzzy computation: {e}")
    
    def get_performance_category(self, score):
        """Convert score to performance category"""
        if score >= 85:
            return "EXCELLENT"
        elif score >= 70:
            return "GOOD"
        elif score >= 50:
            return "AVERAGE"
        else:
            return "NEEDS IMPROVEMENT"
    
    def plot_membership_functions(self):
        """Visualize membership functions"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Attendance
        for label in self.attendance.terms:
            ax1.plot(self.attendance.universe, self.attendance[label].mf, label=label)
        ax1.set_title('Attendance Membership Functions')
        ax1.set_xlabel('Attendance (%)')
        ax1.set_ylabel('Membership')
        ax1.legend()
        ax1.grid(True)
        
        # Internal Marks
        for label in self.internal.terms:
            ax2.plot(self.internal.universe, self.internal[label].mf, label=label)
        ax2.set_title('Internal Marks Membership Functions')
        ax2.set_xlabel('Marks')
        ax2.set_ylabel('Membership')
        ax2.legend()
        ax2.grid(True)
        
        # Assignment
        for label in self.assignment.terms:
            ax3.plot(self.assignment.universe, self.assignment[label].mf, label=label)
        ax3.set_title('Assignment Scores Membership Functions')
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Membership')
        ax3.legend()
        ax3.grid(True)
        
        # Performance
        for label in self.performance.terms:
            ax4.plot(self.performance.universe, self.performance[label].mf, label=label)
        ax4.set_title('Performance Output Membership Functions')
        ax4.set_xlabel('Performance Score')
        ax4.set_ylabel('Membership')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def explain_prediction(self, attendance, internal, assignment):
        """Provide explanation for the prediction"""
        print("\n" + "="*60)
        print("PREDICTION EXPLANATION")
        print("="*60)
        
        # Show input analysis
        print(f"\n📊 INPUT ANALYSIS:")
        print(f"   Attendance: {attendance}%")
        print(f"   Internal Marks: {internal}/100")
        print(f"   Assignment Score: {assignment}/100")
        
        # Calculate prediction
        score = self.predict(attendance, internal, assignment)
        category = self.get_performance_category(score)
        
        print(f"\n🎯 PREDICTION RESULT:")
        print(f"   Performance Score: {score:.2f}/100")
        print(f"   Category: {category}")
        
        print(f"\n📈 INTERPRETATION:")
        if category == "EXCELLENT":
            print("   The student shows consistently high performance across all parameters.")
            print("   Likely to excel in final examinations.")
        elif category == "GOOD":
            print("   The student demonstrates good academic performance.")
            print("   With slight improvements, can achieve excellent results.")
        elif category == "AVERAGE":
            print("   The student has average performance.")
            print("   Needs to focus on weaker areas for improvement.")
        else:
            print("   The student needs significant improvement.")
            print("   Recommend additional support and monitoring.")
        
        print(f"\n💡 SUGGESTIONS:")
        if attendance < 75:
            print("   - Improve attendance")
        if internal < 60:
            print("   - Focus on internal test preparation")
        if assignment < 60:
            print("   - Spend more time on assignments")
        
        print("="*60)

def main():
    """Main program interface"""
    
    print("\n" + "="*60)
    print("🎓 STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("="*60)
    print("Predict student performance using Fuzzy Logic")
    print("="*60)
    
    # Initialize predictor
    predictor = StudentPerformancePredictor()
    
    while True:
        print("\n" + "-"*60)
        print("MAIN MENU")
        print("-"*60)
        print("1. Predict for a single student")
        print("2. Predict for multiple students")
        print("3. Show membership functions (visualization)")
        print("4. Test with sample data")
        print("5. Exit")
        print("-"*60)
        
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':
                # Single student prediction
                print("\nEnter student details:")
                try:
                    att = float(input("Attendance % (0-100): "))
                    internal_marks = float(input("Internal Marks (0-100): "))
                    assign = float(input("Assignment Score (0-100): "))
                    
                    predictor.explain_prediction(att, internal_marks, assign)
                    
                except ValueError as e:
                    print(f"❌ Error: {e}")
                except Exception as e:
                    print(f"❌ Unexpected error: {e}")
            
            elif choice == '2':
                # Multiple students
                print("\nEnter number of students: ")
                try:
                    n = int(input("Number of students: "))
                    results = []
                    
                    for i in range(n):
                        print(f"\nStudent {i+1}:")
                        att = float(input("  Attendance %: "))
                        internal_marks = float(input("  Internal Marks: "))
                        assign = float(input("  Assignment Score: "))
                        
                        score = predictor.predict(att, internal_marks, assign)
                        category = predictor.get_performance_category(score)
                        results.append({
                            'student': i+1,
                            'score': score,
                            'category': category
                        })
                    
                    # Display summary
                    print("\n" + "="*60)
                    print("SUMMARY OF PREDICTIONS")
                    print("="*60)
                    print(f"{'Student':<10} {'Score':<12} {'Category':<20}")
                    print("-"*60)
                    for res in results:
                        print(f"{res['student']:<10} {res['score']:<12.2f} {res['category']:<20}")
                    
                    # Statistics
                    avg_score = np.mean([r['score'] for r in results])
                    print(f"\n📊 Average Performance Score: {avg_score:.2f}")
                    
                except ValueError as e:
                    print(f"❌ Error: {e}")
            
            elif choice == '3':
                # Show visualizations
                print("\nGenerating membership function visualizations...")
                predictor.plot_membership_functions()
            
            elif choice == '4':
                # Test with sample data
                print("\n" + "="*60)
                print("SAMPLE PREDICTIONS")
                print("="*60)
                
                sample_students = [
                    (95, 92, 88, "Top Performer"),
                    (78, 65, 72, "Good Student"),
                    (60, 55, 58, "Average Student"),
                    (45, 38, 42, "Needs Improvement"),
                    (30, 25, 20, "At Risk")
                ]
                
                print(f"{'Description':<20} {'Attendance':<12} {'Internal':<12} {'Assignment':<12} {'Score':<10} {'Category':<15}")
                print("-"*90)
                
                for att, internal, assign, desc in sample_students:
                    score = predictor.predict(att, internal, assign)
                    category = predictor.get_performance_category(score)
                    print(f"{desc:<20} {att:<12} {internal:<12} {assign:<12} {score:<10.2f} {category:<15}")
                
                print("="*60)
            
            elif choice == '5':
                print("\nThank you for using the Student Performance Prediction System!")
                print("Goodbye! 👋")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-5.")
        
        except KeyboardInterrupt:
            print("\n\nProgram interrupted. Exiting...")
            break
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()