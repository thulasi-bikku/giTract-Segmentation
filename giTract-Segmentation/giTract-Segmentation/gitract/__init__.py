"""Difficulty-guided curriculum learning for GI tract segmentation in abdominal MRI."""
ORGANS = ("large_bowel", "small_bowel", "stomach")
GROUPS = ("Easy", "Medium", "Hard")
PREVALENCE = (0.791, 0.155, 0.054)  # Easy, Medium, Hard proportions reported in the paper
