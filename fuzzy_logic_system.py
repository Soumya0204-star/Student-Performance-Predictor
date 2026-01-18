# fuzzy_logic_system.py
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

class StudentPerformanceFuzzySystem:
    """Fuzzy Logic System for Student Performance Prediction"""
    
    def __init__(self):
        print("🧠 Initializing Fuzzy Logic System...")
        self.setup_fuzzy_system()
        
    def setup_fuzzy_system(self):
        """Setup fuzzy variables, membership functions, and rules"""
        
        # 1. Define Fuzzy Variables (Universe of Discourse)
        # Range: 0-100 for all inputs and output
        self.attendance = ctrl.Antecedent(np.arange(0, 101, 1), 'attendance')
        self.internal = ctrl.Antecedent(np.arange(0, 101, 1), 'internal_marks')
        self.assignment = ctrl.Antecedent(np.arange(0, 101, 1), 'assignment_marks')
        self.performance = ctrl.Consequent(np.arange(0, 101, 1), 'performance')
        
        # 2. Define Membership Functions (Triangular/Trapezoidal)
        # Attendance: Poor, Average, Good
        self.attendance['poor'] = fuzz.trimf(self.attendance.universe, [0, 0, 70])
        self.attendance['average'] = fuzz.trimf(self.attendance.universe, [60, 75, 90])
        self.attendance['good'] = fuzz.trimf(self.attendance.universe, [80, 95, 100])
        
        # Internal Marks: Low, Medium, High
        self.internal['low'] = fuzz.trimf(self.internal.universe, [0, 0, 70])
        self.internal['medium'] = fuzz.trimf(self.internal.universe, [60, 75, 85])
        self.internal['high'] = fuzz.trimf(self.internal.universe, [75, 90, 100])
        
        # Assignment: Weak, Satisfactory, Excellent
        self.assignment['weak'] = fuzz.trimf(self.assignment.universe, [0, 0, 75])
        self.assignment['satisfactory'] = fuzz.trimf(self.assignment.universe, [65, 80, 95])
        self.assignment['excellent'] = fuzz.trimf(self.assignment.universe, [85, 95, 100])
        
        # Performance: Fail, Pass, Good, Excellent
        self.performance['fail'] = fuzz.trimf(self.performance.universe, [0, 0, 60])
        self.performance['pass'] = fuzz.trimf(self.performance.universe, [50, 65, 80])
        self.performance['good'] = fuzz.trimf(self.performance.universe, [70, 80, 90])
        self.performance['excellent'] = fuzz.trimf(self.performance.universe, [85, 95, 100])
        
        # 3. Define Fuzzy Rules (Mamdani Inference)
        self.rules = [
            # Rule 1: Excellent in all → Excellent performance
            ctrl.Rule(self.attendance['good'] & self.internal['high'] & self.assignment['excellent'], 
                     self.performance['excellent']),
            
            # Rule 2: Good attendance and marks → Good performance
            ctrl.Rule(self.attendance['good'] & self.internal['medium'] & self.assignment['satisfactory'], 
                     self.performance['good']),
            
            # Rule 3: Average in all → Pass
            ctrl.Rule(self.attendance['average'] & self.internal['medium'] & self.assignment['satisfactory'], 
                     self.performance['pass']),
            
            # Rule 4: Poor attendance or low marks → Risk of fail
            ctrl.Rule(self.attendance['poor'] | self.internal['low'] | self.assignment['weak'], 
                     self.performance['fail']),
            
            # Rule 5: Excellent assignments can compensate
            ctrl.Rule(self.assignment['excellent'] & (self.internal['medium'] | self.attendance['average']), 
                     self.performance['good']),
            
            # Rule 6: Good attendance helps
            ctrl.Rule(self.attendance['good'] & (self.internal['medium'] | self.assignment['satisfactory']), 
                     self.performance['good']),
            
            # Rule 7: High internal marks boost
            ctrl.Rule(self.internal['high'] & (self.attendance['average'] | self.assignment['satisfactory']), 
                     self.performance['good']),
        ]
        
        # 4. Create Control System
        self.performance_ctrl = ctrl.ControlSystem(self.rules)
        self.performance_sim = ctrl.ControlSystemSimulation(self.performance_ctrl)
        
        print(f"✅ Fuzzy system initialized with {len(self.rules)} rules")
    
    def predict(self, attendance, internal, assignment):
        """Predict performance using fuzzy logic"""
        try:
            # Set inputs
            self.performance_sim.input['attendance'] = attendance
            self.performance_sim.input['internal_marks'] = internal
            self.performance_sim.input['assignment_marks'] = assignment
            
            # Compute output
            self.performance_sim.compute()
            
            # Get crisp output
            predicted_score = self.performance_sim.output['performance']
            
            # Get fuzzy membership values for each category
            membership = {
                'fail': fuzz.interp_membership(self.performance.universe, 
                                              self.performance['fail'].mf, 
                                              predicted_score),
                'pass': fuzz.interp_membership(self.performance.universe,
                                              self.performance['pass'].mf,
                                              predicted_score),
                'good': fuzz.interp_membership(self.performance.universe,
                                              self.performance['good'].mf,
                                              predicted_score),
                'excellent': fuzz.interp_membership(self.performance.universe,
                                                   self.performance['excellent'].mf,
                                                   predicted_score)
            }
            
            # Determine category
            category = max(membership, key=membership.get).upper()
            
            return {
                'score': round(predicted_score, 2),
                'category': category,
                'membership': membership,
                'method': 'fuzzy_logic'
            }
            
        except Exception as e:
            print(f"⚠️ Fuzzy system error: {e}")
            # Fallback to weighted average
            score = attendance * 0.3 + internal * 0.4 + assignment * 0.3
            return {
                'score': round(score, 2),
                'category': self._get_category(score),
                'membership': {},
                'method': 'weighted_average_fallback'
            }
    
    def _get_category(self, score):
        """Helper to get category from score"""
        if score >= 85: return "EXCELLENT"
        elif score >= 75: return "GOOD"
        elif score >= 60: return "PASS"
        else: return "FAIL"
    
    def visualize_fuzzy_sets(self):
        """Create visualization of fuzzy sets"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Attendance fuzzy sets
        ax1.plot(self.attendance.universe, self.attendance['poor'].mf, 'r', linewidth=1.5, label='Poor')
        ax1.plot(self.attendance.universe, self.attendance['average'].mf, 'g', linewidth=1.5, label='Average')
        ax1.plot(self.attendance.universe, self.attendance['good'].mf, 'b', linewidth=1.5, label='Good')
        ax1.set_title('Attendance Fuzzy Sets')
        ax1.set_ylabel('Membership')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Internal marks fuzzy sets
        ax2.plot(self.internal.universe, self.internal['low'].mf, 'r', linewidth=1.5, label='Low')
        ax2.plot(self.internal.universe, self.internal['medium'].mf, 'g', linewidth=1.5, label='Medium')
        ax2.plot(self.internal.universe, self.internal['high'].mf, 'b', linewidth=1.5, label='High')
        ax2.set_title('Internal Marks Fuzzy Sets')
        ax2.set_ylabel('Membership')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Assignment fuzzy sets
        ax3.plot(self.assignment.universe, self.assignment['weak'].mf, 'r', linewidth=1.5, label='Weak')
        ax3.plot(self.assignment.universe, self.assignment['satisfactory'].mf, 'g', linewidth=1.5, label='Satisfactory')
        ax3.plot(self.assignment.universe, self.assignment['excellent'].mf, 'b', linewidth=1.5, label='Excellent')
        ax3.set_title('Assignment Fuzzy Sets')
        ax3.set_xlabel('Marks')
        ax3.set_ylabel('Membership')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Performance fuzzy sets
        ax4.plot(self.performance.universe, self.performance['fail'].mf, 'r', linewidth=1.5, label='Fail')
        ax4.plot(self.performance.universe, self.performance['pass'].mf, 'y', linewidth=1.5, label='Pass')
        ax4.plot(self.performance.universe, self.performance['good'].mf, 'g', linewidth=1.5, label='Good')
        ax4.plot(self.performance.universe, self.performance['excellent'].mf, 'b', linewidth=1.5, label='Excellent')
        ax4.set_title('Performance Fuzzy Sets')
        ax4.set_xlabel('Score')
        ax4.set_ylabel('Membership')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fuzzy_sets_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Fuzzy sets visualization saved as 'fuzzy_sets_visualization.png'")
    
    def test_examples(self):
        """Test with example cases"""
        test_cases = [
            (95, 92, 90, "Excellent student"),
            (85, 75, 80, "Good student"),
            (75, 65, 70, "Average student"),
            (55, 50, 60, "Weak student"),
            (90, 50, 90, "Irregular student"),
        ]
        
        print("\n🧪 Testing Fuzzy System with Examples:")
        print("-" * 60)
        
        for att, int_m, ass, desc in test_cases:
            result = self.predict(att, int_m, ass)
            print(f"\n{desc}:")
            print(f"  Input: Attendance={att}%, Internal={int_m}, Assignment={ass}")
            print(f"  Fuzzy Prediction: {result['score']}% ({result['category']})")
            print(f"  Method: {result['method']}")
            if result['membership']:
                print(f"  Membership values: {result['membership']}")

# Main execution
if __name__ == "__main__":
    # Create fuzzy system
    fuzzy_system = StudentPerformanceFuzzySystem()
    
    # Visualize fuzzy sets
    fuzzy_system.visualize_fuzzy_sets()
    
    # Test with examples
    fuzzy_system.test_examples()
    
    # Test with your values
    print("\n🎓 Testing with your values (85, 75, 80):")
    result = fuzzy_system.predict(85, 75, 80)
    print(f"Result: {result}")