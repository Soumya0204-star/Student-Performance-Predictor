"""
=======================================================
STUDENT PERFORMANCE PREDICTION WITH DATASET
=======================================================
Integrates real dataset, compares with traditional methods
=======================================================
"""

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import random

class EnhancedStudentPredictor:
    """Advanced predictor with dataset integration and comparison"""
    
    def __init__(self):
        self.setup_fuzzy_system()
        self.dataset = None
        self.results = None
        
    def setup_fuzzy_system(self):
        """Setup enhanced fuzzy system"""
        # Inputs with more granular membership
        self.attendance = ctrl.Antecedent(np.arange(0, 101, 1), 'attendance')
        self.internal = ctrl.Antecedent(np.arange(0, 101, 1), 'internal')
        self.assignment = ctrl.Antecedent(np.arange(0, 101, 1), 'assignment')
        
        # Output
        self.performance = ctrl.Consequent(np.arange(0, 101, 1), 'performance')
        
        # Enhanced membership functions (5 categories each)
        # Attendance
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
        
        # Assignment
        self.assignment['very_low'] = fuzz.trapmf(self.assignment.universe, [0, 0, 20, 35])
        self.assignment['low'] = fuzz.trimf(self.assignment.universe, [30, 40, 50])
        self.assignment['medium'] = fuzz.trimf(self.assignment.universe, [45, 60, 75])
        self.assignment['high'] = fuzz.trimf(self.assignment.universe, [70, 80, 90])
        self.assignment['excellent'] = fuzz.trapmf(self.assignment.universe, [85, 90, 100, 100])
        
        # Performance Output
        self.performance['very_poor'] = fuzz.trapmf(self.performance.universe, [0, 0, 20, 35])
        self.performance['poor'] = fuzz.trimf(self.performance.universe, [30, 40, 50])
        self.performance['average'] = fuzz.trimf(self.performance.universe, [45, 55, 70])
        self.performance['good'] = fuzz.trimf(self.performance.universe, [65, 75, 85])
        self.performance['excellent'] = fuzz.trapmf(self.performance.universe, [80, 90, 100, 100])
        
        # Comprehensive Rule Set (20 rules)
        rules = []
        
        # Very Poor Performance Rules
        rules.append(ctrl.Rule(
            self.attendance['very_low'] & self.internal['very_poor'] & self.assignment['very_low'],
            self.performance['very_poor']
        ))
        rules.append(ctrl.Rule(
            self.attendance['very_low'] & self.internal['very_poor'],
            self.performance['very_poor']
        ))
        
        # Poor Performance Rules
        rules.append(ctrl.Rule(
            self.attendance['low'] & self.internal['poor'] & self.assignment['low'],
            self.performance['poor']
        ))
        rules.append(ctrl.Rule(
            self.attendance['very_low'] & self.internal['average'],
            self.performance['poor']
        ))
        
        # Average Performance Rules
        rules.append(ctrl.Rule(
            self.attendance['medium'] & self.internal['average'] & self.assignment['medium'],
            self.performance['average']
        ))
        rules.append(ctrl.Rule(
            self.attendance['low'] & self.internal['good'] & self.assignment['medium'],
            self.performance['average']
        ))
        rules.append(ctrl.Rule(
            self.attendance['medium'] & self.internal['poor'] & self.assignment['medium'],
            self.performance['average']
        ))
        
        # Good Performance Rules
        rules.append(ctrl.Rule(
            self.attendance['high'] & self.internal['average'] & self.assignment['high'],
            self.performance['good']
        ))
        rules.append(ctrl.Rule(
            self.attendance['medium'] & self.internal['good'] & self.assignment['high'],
            self.performance['good']
        ))
        rules.append(ctrl.Rule(
            self.attendance['high'] & self.internal['good'] & self.assignment['medium'],
            self.performance['good']
        ))
        
        # Excellent Performance Rules
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
            self.attendance['high'] & self.assignment['excellent'],
            self.performance['excellent']
        ))
        rules.append(ctrl.Rule(
            self.internal['excellent'] & self.assignment['excellent'],
            self.performance['excellent']
        ))
        
        # Mixed Scenarios
        rules.append(ctrl.Rule(
            self.attendance['excellent'] & self.internal['average'],
            self.performance['good']
        ))
        rules.append(ctrl.Rule(
            self.attendance['medium'] & self.internal['excellent'],
            self.performance['good']
        ))
        rules.append(ctrl.Rule(
            self.attendance['low'] & self.internal['excellent'],
            self.performance['average']
        ))
        rules.append(ctrl.Rule(
            self.attendance['excellent'] & self.internal['poor'],
            self.performance['average']
        ))
        rules.append(ctrl.Rule(
            self.attendance['high'] & self.internal['poor'],
            self.performance['average']
        ))
        
        # Create control system
        self.performance_ctrl = ctrl.ControlSystem(rules)
        self.performance_sim = ctrl.ControlSystemSimulation(self.performance_ctrl)
    
    def generate_sample_dataset(self, n_students=100):
        """Generate synthetic student dataset"""
        np.random.seed(42)
        
        # Generate correlated data
        base_performance = np.random.normal(70, 15, n_students)
        base_performance = np.clip(base_performance, 0, 100)
        
        # Attendance correlates with performance
        attendance = np.clip(base_performance + np.random.normal(0, 10, n_students), 0, 100)
        
        # Internal marks correlate with performance
        internal = np.clip(base_performance + np.random.normal(0, 12, n_students), 0, 100)
        
        # Assignment scores correlate with performance
        assignment = np.clip(base_performance + np.random.normal(0, 8, n_students), 0, 100)
        
        # Add some noise/outliers
        outlier_indices = np.random.choice(n_students, size=10, replace=False)
        attendance[outlier_indices] = np.random.uniform(0, 100, 10)
        internal[outlier_indices] = np.random.uniform(0, 100, 10)
        assignment[outlier_indices] = np.random.uniform(0, 100, 10)
        
        # Create DataFrame
        self.dataset = pd.DataFrame({
            'Student_ID': range(1, n_students + 1),
            'Attendance': attendance,
            'Internal_Marks': internal,
            'Assignment_Score': assignment
        })
        
        # Calculate actual performance (weighted average for comparison)
        self.dataset['Actual_Performance'] = (
            self.dataset['Attendance'] * 0.2 +
            self.dataset['Internal_Marks'] * 0.5 +
            self.dataset['Assignment_Score'] * 0.3
        )
        
        return self.dataset
    
    def predict_for_dataset(self):
        """Predict performance for all students in dataset"""
        if self.dataset is None:
            print("No dataset loaded. Generating sample dataset...")
            self.generate_sample_dataset()
        
        predictions = []
        categories = []
        
        for idx, row in self.dataset.iterrows():
            try:
                # Set inputs
                self.performance_sim.input['attendance'] = row['Attendance']
                self.performance_sim.input['internal'] = row['Internal_Marks']
                self.performance_sim.input['assignment'] = row['Assignment_Score']
                
                # Compute
                self.performance_sim.compute()
                score = self.performance_sim.output['performance']
                predictions.append(score)
                
                # Categorize
                if score >= 85:
                    category = "EXCELLENT"
                elif score >= 70:
                    category = "GOOD"
                elif score >= 50:
                    category = "AVERAGE"
                elif score >= 35:
                    category = "POOR"
                else:
                    category = "VERY_POOR"
                categories.append(category)
                
            except Exception as e:
                predictions.append(np.nan)
                categories.append("ERROR")
        
        # Add predictions to dataset
        self.dataset['Fuzzy_Prediction'] = predictions
        self.dataset['Fuzzy_Category'] = categories
        
        # Calculate actual category for comparison
        def get_actual_category(score):
            if score >= 85:
                return "EXCELLENT"
            elif score >= 70:
                return "GOOD"
            elif score >= 50:
                return "AVERAGE"
            elif score >= 35:
                return "POOR"
            else:
                return "VERY_POOR"
        
        self.dataset['Actual_Category'] = self.dataset['Actual_Performance'].apply(get_actual_category)
        
        # Calculate traditional method (weighted average)
        self.dataset['Traditional_Prediction'] = (
            self.dataset['Attendance'] * 0.2 +
            self.dataset['Internal_Marks'] * 0.5 +
            self.dataset['Assignment_Score'] * 0.3
        )
        
        return self.dataset
    
    def evaluate_performance(self):
        """Evaluate fuzzy system performance"""
        if 'Fuzzy_Prediction' not in self.dataset.columns:
            self.predict_for_dataset()
        
        # Calculate errors
        mae_fuzzy = mean_absolute_error(
            self.dataset['Actual_Performance'], 
            self.dataset['Fuzzy_Prediction']
        )
        
        mae_traditional = mean_absolute_error(
            self.dataset['Actual_Performance'], 
            self.dataset['Traditional_Prediction']
        )
        
        # Calculate R² scores
        r2_fuzzy = r2_score(
            self.dataset['Actual_Performance'], 
            self.dataset['Fuzzy_Prediction']
        )
        
        r2_traditional = r2_score(
            self.dataset['Actual_Performance'], 
            self.dataset['Traditional_Prediction']
        )
        
        # Calculate category accuracy
        fuzzy_correct = (self.dataset['Fuzzy_Category'] == self.dataset['Actual_Category']).sum()
        category_accuracy = fuzzy_correct / len(self.dataset) * 100
        
        results = {
            'MAE_Fuzzy': mae_fuzzy,
            'MAE_Traditional': mae_traditional,
            'R2_Fuzzy': r2_fuzzy,
            'R2_Traditional': r2_traditional,
            'Category_Accuracy': category_accuracy,
            'Fuzzy_Better_MAE': mae_fuzzy < mae_traditional,
            'Fuzzy_Better_R2': r2_fuzzy > r2_traditional
        }
        
        return results
    
    def visualize_comparison(self):
        """Create comparison visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Scatter plot: Fuzzy vs Traditional
        ax = axes[0, 0]
        ax.scatter(self.dataset['Traditional_Prediction'], 
                  self.dataset['Fuzzy_Prediction'], 
                  alpha=0.6)
        ax.plot([0, 100], [0, 100], 'r--', alpha=0.5)
        ax.set_xlabel('Traditional Prediction')
        ax.set_ylabel('Fuzzy Prediction')
        ax.set_title('Fuzzy vs Traditional Predictions')
        ax.grid(True, alpha=0.3)
        
        # 2. Error comparison
        ax = axes[0, 1]
        errors = pd.DataFrame({
            'Method': ['Traditional', 'Fuzzy'],
            'MAE': [mean_absolute_error(self.dataset['Actual_Performance'], 
                                       self.dataset['Traditional_Prediction']),
                   mean_absolute_error(self.dataset['Actual_Performance'], 
                                       self.dataset['Fuzzy_Prediction'])]
        })
        bars = ax.bar(errors['Method'], errors['MAE'], color=['skyblue', 'lightcoral'])
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Prediction Error Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')
        
        # 3. Category distribution
        ax = axes[0, 2]
        category_counts = self.dataset['Fuzzy_Category'].value_counts()
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'darkgreen']
        ax.pie(category_counts.values, labels=category_counts.index, 
               autopct='%1.1f%%', colors=colors)
        ax.set_title('Performance Category Distribution (Fuzzy)')
        
        # 4. Correlation heatmap
        ax = axes[1, 0]
        corr_matrix = self.dataset[['Attendance', 'Internal_Marks', 'Assignment_Score', 
                                   'Actual_Performance', 'Fuzzy_Prediction']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Matrix')
        
        # 5. Prediction vs Actual (Fuzzy)
        ax = axes[1, 1]
        ax.scatter(self.dataset['Actual_Performance'], 
                  self.dataset['Fuzzy_Prediction'], 
                  alpha=0.6, label='Fuzzy')
        ax.scatter(self.dataset['Actual_Performance'], 
                  self.dataset['Traditional_Prediction'], 
                  alpha=0.6, label='Traditional', color='red')
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.5)
        ax.set_xlabel('Actual Performance')
        ax.set_ylabel('Predicted Performance')
        ax.set_title('Predictions vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Residual plot
        ax = axes[1, 2]
        fuzzy_residuals = self.dataset['Fuzzy_Prediction'] - self.dataset['Actual_Performance']
        traditional_residuals = self.dataset['Traditional_Prediction'] - self.dataset['Actual_Performance']
        
        ax.hist(fuzzy_residuals, bins=20, alpha=0.7, label='Fuzzy', color='blue')
        ax.hist(traditional_residuals, bins=20, alpha=0.7, label='Traditional', color='red')
        ax.axvline(x=0, color='black', linestyle='--')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_student(self, student_id):
        """Analyze a specific student's prediction"""
        if student_id not in self.dataset['Student_ID'].values:
            return f"Student ID {student_id} not found in dataset."
        
        student = self.dataset[self.dataset['Student_ID'] == student_id].iloc[0]
        
        analysis = f"""
{'='*60}
ANALYSIS FOR STUDENT #{student_id}
{'='*60}

📊 INPUT VALUES:
   • Attendance:       {student['Attendance']:.1f}%
   • Internal Marks:   {student['Internal_Marks']:.1f}/100
   • Assignment Score: {student['Assignment_Score']:.1f}/100

📈 PERFORMANCE PREDICTIONS:
   • Actual Performance:       {student['Actual_Performance']:.1f}/100
   • Traditional Prediction:   {student['Traditional_Prediction']:.1f}/100
   • Fuzzy Logic Prediction:   {student['Fuzzy_Prediction']:.1f}/100
   • Fuzzy Category:           {student['Fuzzy_Category']}
   • Actual Category:          {student['Actual_Category']}

📉 PREDICTION ACCURACY:
   • Traditional Error: {abs(student['Traditional_Prediction'] - student['Actual_Performance']):.2f}
   • Fuzzy Logic Error: {abs(student['Fuzzy_Prediction'] - student['Actual_Performance']):.2f}
   • Better Method:     {'Fuzzy Logic' if abs(student['Fuzzy_Prediction'] - student['Actual_Performance']) < 
                                 abs(student['Traditional_Prediction'] - student['Actual_Performance']) 
                                 else 'Traditional'}

💡 RECOMMENDATIONS:
"""
        
        # Add recommendations based on scores
        if student['Attendance'] < 75:
            analysis += "   • Improve attendance (current: {:.1f}%)\n".format(student['Attendance'])
        if student['Internal_Marks'] < 60:
            analysis += "   • Focus on internal test preparation (current: {:.1f})\n".format(student['Internal_Marks'])
        if student['Assignment_Score'] < 60:
            analysis += "   • Spend more time on assignments (current: {:.1f})\n".format(student['Assignment_Score'])
        
        if student['Fuzzy_Category'] in ['EXCELLENT', 'GOOD']:
            analysis += "   • Continue current study habits\n"
        else:
            analysis += "   • Consider seeking academic support\n"
        
        analysis += "="*60
        
        return analysis
    
    def export_results(self, filename='fuzzy_predictions_results.csv'):
        """Export predictions and analysis to CSV"""
        if self.dataset is not None:
            self.dataset.to_csv(filename, index=False)
            print(f"Results exported to {filename}")
            return True
        return False

def main():
    """Main interactive interface"""
    print("\n" + "="*70)
    print("🎓 ADVANCED STUDENT PERFORMANCE PREDICTION WITH DATASET")
    print("="*70)
    print("Features:")
    print("  • Synthetic dataset generation")
    print("  • Fuzzy Logic vs Traditional method comparison")
    print("  • Performance evaluation metrics")
    print("  • Visualization of results")
    print("  • Individual student analysis")
    print("="*70)
    
    # Initialize predictor
    predictor = EnhancedStudentPredictor()
    
    while True:
        print("\n" + "-"*70)
        print("MAIN MENU")
        print("-"*70)
        print("1. Generate sample dataset")
        print("2. Make predictions for all students")
        print("3. Evaluate system performance")
        print("4. Visualize comparisons")
        print("5. Analyze specific student")
        print("6. Show sample predictions")
        print("7. Export results to CSV")
        print("8. Exit")
        print("-"*70)
        
        try:
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == '1':
                n = input("Enter number of students to generate (default: 100): ").strip()
                n = int(n) if n else 100
                dataset = predictor.generate_sample_dataset(n)
                print(f"✅ Generated dataset with {n} students")
                print("\nDataset Preview:")
                print(dataset.head())
                
            elif choice == '2':
                if predictor.dataset is None:
                    print("Generating default dataset first...")
                    predictor.generate_sample_dataset()
                
                predictions = predictor.predict_for_dataset()
                print("✅ Predictions completed for all students")
                print("\nFirst 10 predictions:")
                print(predictions[['Student_ID', 'Attendance', 'Internal_Marks', 
                                  'Assignment_Score', 'Fuzzy_Prediction', 'Fuzzy_Category']].head(10))
                
            elif choice == '3':
                if predictor.dataset is None:
                    print("Please generate dataset and make predictions first (options 1 & 2)")
                    continue
                
                results = predictor.evaluate_performance()
                print("\n" + "="*70)
                print("SYSTEM PERFORMANCE EVALUATION")
                print("="*70)
                
                print(f"\n📊 ERROR METRICS:")
                print(f"   Fuzzy Logic MAE:      {results['MAE_Fuzzy']:.4f}")
                print(f"   Traditional MAE:      {results['MAE_Traditional']:.4f}")
                print(f"   Difference:           {abs(results['MAE_Fuzzy'] - results['MAE_Traditional']):.4f}")
                print(f"   Fuzzy is {'BETTER' if results['Fuzzy_Better_MAE'] else 'WORSE'} in MAE")
                
                print(f"\n📈 R² SCORES (closer to 1 is better):")
                print(f"   Fuzzy Logic R²:       {results['R2_Fuzzy']:.4f}")
                print(f"   Traditional R²:       {results['R2_Traditional']:.4f}")
                print(f"   Fuzzy is {'BETTER' if results['Fuzzy_Better_R2'] else 'WORSE'} in R²")
                
                print(f"\n🎯 CATEGORY ACCURACY:")
                print(f"   Correct categories:   {results['Category_Accuracy']:.2f}%")
                
                print("\n" + "="*70)
                
            elif choice == '4':
                if predictor.dataset is None:
                    print("Please generate dataset and make predictions first (options 1 & 2)")
                    continue
                
                print("Generating visualizations...")
                predictor.visualize_comparison()
                
            elif choice == '5':
                if predictor.dataset is None:
                    print("Please generate dataset first (option 1)")
                    continue
                
                try:
                    student_id = int(input("Enter Student ID: "))
                    analysis = predictor.analyze_student(student_id)
                    print(analysis)
                except ValueError:
                    print("❌ Please enter a valid Student ID number")
                    
            elif choice == '6':
                if predictor.dataset is None:
                    print("Generating sample dataset...")
                    predictor.generate_sample_dataset(20)
                    predictor.predict_for_dataset()
                
                print("\nSAMPLE PREDICTIONS (First 10 Students):")
                print("="*90)
                print(f"{'ID':<5} {'Attend':<8} {'Internal':<10} {'Assign':<8} {'Actual':<10} {'Trad':<10} {'Fuzzy':<10} {'Fuzzy Cat':<12} {'Actual Cat':<12}")
                print("-"*90)
                
                for i in range(min(10, len(predictor.dataset))):
                    row = predictor.dataset.iloc[i]
                    print(f"{int(row['Student_ID']):<5} "
                          f"{row['Attendance']:<8.1f} "
                          f"{row['Internal_Marks']:<10.1f} "
                          f"{row['Assignment_Score']:<8.1f} "
                          f"{row['Actual_Performance']:<10.1f} "
                          f"{row['Traditional_Prediction']:<10.1f} "
                          f"{row['Fuzzy_Prediction']:<10.1f} "
                          f"{row['Fuzzy_Category']:<12} "
                          f"{row['Actual_Category']:<12}")
                
                print("="*90)
                
            elif choice == '7':
                if predictor.dataset is None:
                    print("No data to export. Please generate predictions first.")
                else:
                    filename = input("Enter filename (default: fuzzy_predictions_results.csv): ").strip()
                    filename = filename if filename else 'fuzzy_predictions_results.csv'
                    predictor.export_results(filename)
                    
            elif choice == '8':
                print("\nThank you for using the Advanced Student Performance Prediction System!")
                print("Goodbye! 👋")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-8.")
                
        except KeyboardInterrupt:
            print("\n\nProgram interrupted. Exiting...")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()