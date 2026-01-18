# generate_dataset.py
import numpy as np
import pandas as pd
import random
from datetime import datetime

def generate_fuzzy_student_dataset(num_students=500):
    """
    Generate synthetic student dataset using fuzzy logic concepts
    """
    print("🎓 Generating Fuzzy Logic Student Dataset...")
    
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    for student_id in range(1, num_students + 1):
        # Base characteristics (realistic ranges)
        attendance_base = np.random.normal(80, 15)  # Mean 80%, SD 15
        attendance_base = np.clip(attendance_base, 40, 100)
        
        internal_base = np.random.normal(75, 20)    # Mean 75, SD 20
        internal_base = np.clip(internal_base, 30, 100)
        
        assignment_base = np.random.normal(78, 18)  # Mean 78, SD 18
        assignment_base = np.clip(assignment_base, 40, 100)
        
        # Add some correlation (good attendance → better marks)
        correlation_factor = attendance_base / 100
        
        internal_final = internal_base * (0.7 + 0.3 * correlation_factor) + np.random.normal(0, 5)
        assignment_final = assignment_base * (0.6 + 0.4 * correlation_factor) + np.random.normal(0, 4)
        
        # Clip to valid ranges
        attendance = np.clip(attendance_base, 40, 100)
        internal = np.clip(internal_final, 30, 100)
        assignment = np.clip(assignment_final, 40, 100)
        
        # Apply FUZZY LOGIC RULES for final grade (not simple average)
        # Rule 1: Excellent in all → High score
        if attendance > 85 and internal > 80 and assignment > 85:
            final_grade = (attendance*0.25 + internal*0.35 + assignment*0.40) * 1.05
        # Rule 2: Poor attendance → Penalty
        elif attendance < 60:
            final_grade = (attendance*0.20 + internal*0.40 + assignment*0.40) * 0.9
        # Rule 3: Excellent assignments → Boost
        elif assignment > 90:
            final_grade = (attendance*0.25 + internal*0.35 + assignment*0.40) * 1.08
        # Rule 4: Default fuzzy calculation
        else:
            # Fuzzy weights based on values
            if attendance > 75:
                att_weight = 0.30
            elif attendance > 60:
                att_weight = 0.25
            else:
                att_weight = 0.20
                
            if internal > 80:
                int_weight = 0.45
            elif internal > 65:
                int_weight = 0.40
            else:
                int_weight = 0.35
                
            if assignment > 85:
                ass_weight = 0.35
            elif assignment > 70:
                ass_weight = 0.30
            else:
                ass_weight = 0.25
                
            # Normalize weights to sum to 1
            total = att_weight + int_weight + ass_weight
            att_weight /= total
            int_weight /= total
            ass_weight /= total
            
            final_grade = (attendance * att_weight + 
                          internal * int_weight + 
                          assignment * ass_weight)
        
        # Add some random noise (±3%)
        final_grade = final_grade + np.random.normal(0, 2)
        final_grade = np.clip(final_grade, 35, 100)
        
        # Determine category (fuzzy boundaries)
        if final_grade >= 85:
            category = "EXCELLENT"
        elif final_grade >= 75:
            category = "GOOD"
        elif final_grade >= 60:
            category = "AVERAGE"
        else:
            category = "NEEDS_IMPROVEMENT"
        
        data.append([
            student_id,
            round(attendance, 2),
            round(internal, 2),
            round(assignment, 2),
            round(final_grade, 2),
            category
        ])
    
    # Create DataFrame
    columns = ['student_id', 'attendance', 'internal_marks', 
               'assignment_marks', 'final_grade', 'category']
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'student_fuzzy_dataset_{timestamp}.csv'
    df.to_csv(filename, index=False)
    
    print(f"✅ Dataset generated: {filename}")
    print(f"📊 Records: {len(df)}")
    print(f"📈 Statistics:")
    print(df[['attendance', 'internal_marks', 'assignment_marks', 'final_grade']].describe())
    
    # Show sample
    print(f"\n🎯 Sample data (first 5 students):")
    print(df.head())
    
    return df, filename

if __name__ == "__main__":
    df, filename = generate_fuzzy_student_dataset(500)
    
    # Create a smaller test dataset for demo
    test_df = df.sample(10, random_state=42)
    test_df.to_csv('test_student_dataset.csv', index=False)
    print(f"\n📝 Test dataset saved: test_student_dataset.csv (10 samples)")