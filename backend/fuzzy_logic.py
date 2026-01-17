# CREATOR: SOUMYA | GitHub: https://github.com/Soumya0204-star/Student-Performance-Predictor
"""
Simple fuzzy logic implementation for student performance prediction
No external dependencies required
"""

class FuzzyPredictor:
    def __init__(self):
        pass
    
    def predict(self, attendance, internal, assignment):
        """
        Simple fuzzy-like prediction
        """
        # Normalize inputs
        att_norm = attendance / 100.0
        int_norm = internal / 100.0
        ass_norm = assignment / 100.0
        
        # Simple fuzzy rules
        if att_norm >= 0.8 and int_norm >= 0.8 and ass_norm >= 0.8:
            base_score = 90
        elif att_norm >= 0.7 and int_norm >= 0.7 and ass_norm >= 0.7:
            base_score = 80
        elif att_norm >= 0.6 and int_norm >= 0.6 and ass_norm >= 0.6:
            base_score = 70
        elif att_norm >= 0.5 and int_norm >= 0.5 and ass_norm >= 0.5:
            base_score = 60
        else:
            base_score = 50
        
        # Add weighted components
        weighted_score = (att_norm * 30 + int_norm * 40 + ass_norm * 30)
        
        # Combine
        final_score = (base_score * 0.4 + weighted_score * 0.6)
        
        # Ensure within bounds
        return min(100, max(0, round(final_score, 2)))

def get_category(score):
    if score >= 80:
        return "EXCELLENT"
    elif score >= 60:
        return "GOOD"
    elif score >= 40:
        return "AVERAGE"
    else:
        return "NEEDS IMPROVEMENT"

# Create instance
predictor = FuzzyPredictor()