# run_demo.py
from generate_dataset import generate_fuzzy_student_dataset
from fuzzy_logic_system import StudentPerformanceFuzzySystem
import pandas as pd

def main():
    print("=" * 60)
    print("🎓 STUDENT PERFORMANCE PREDICTOR - FUZZY LOGIC DEMO")
    print("=" * 60)
    
    # Step 1: Generate dataset
    print("\n📊 STEP 1: Generating Dataset...")
    df, filename = generate_fuzzy_student_dataset(300)  # Smaller for demo
    
    # Step 2: Initialize fuzzy system
    print("\n🧠 STEP 2: Initializing Fuzzy Logic System...")
    fuzzy_system = StudentPerformanceFuzzySystem()
    
    # Step 3: Test with dataset samples
    print("\n🔍 STEP 3: Testing with Dataset Samples...")
    sample_students = df.sample(3)
    
    for idx, student in sample_students.iterrows():
        print(f"\nStudent ID: {student['student_id']}")
        print(f"Actual: Attendance={student['attendance']}%, "
              f"Internal={student['internal_marks']}, "
              f"Assignment={student['assignment_marks']}")
        print(f"Actual Final Grade: {student['final_grade']}% ({student['category']})")
        
        # Fuzzy prediction
        result = fuzzy_system.predict(
            student['attendance'],
            student['internal_marks'],
            student['assignment_marks']
        )
        print(f"Fuzzy Prediction: {result['score']}% ({result['category']})")
    
    # Step 4: Interactive demo
    print("\n🎮 STEP 4: Interactive Demo")
    print("Enter student details (or press Enter for default 85, 75, 80):")
    
    try:
        att = float(input("Attendance % (0-100): ") or 85)
        int_m = float(input("Internal Marks (0-100): ") or 75)
        ass = float(input("Assignment Marks (0-100): ") or 80)
        
        result = fuzzy_system.predict(att, int_m, ass)
        
        print(f"\n📈 FUZZY LOGIC PREDICTION:")
        print(f"Score: {result['score']}%")
        print(f"Category: {result['category']}")
        print(f"Method: {result['method']}")
        
        if result['membership']:
            print("\n🎭 Membership Degrees:")
            for category, degree in result['membership'].items():
                print(f"  {category.upper()}: {degree:.3f}")
    
    except ValueError:
        print("⚠️ Invalid input. Using default values.")
        result = fuzzy_system.predict(85, 75, 80)
        print(f"\nDefault prediction: {result['score']}% ({result['category']})")
    
    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print(f"📁 Dataset saved as: {filename}")
    print("🖼️ Fuzzy sets visualization: fuzzy_sets_visualization.png")
    print("=" * 60)

if __name__ == "__main__":
    main()