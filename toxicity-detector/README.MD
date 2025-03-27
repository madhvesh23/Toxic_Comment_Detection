## This is Machine Learning Project

## Title: Multi-task Learning for Toxic Comment Classification and Rationale Extraction



Why We Use F1-Score to Evaluate Toxic Comment Detection
In toxic comment classification, we’re dealing with an imbalanced and sensitive problem, where:
	•	Many more comments are non-toxic (label = 0) than toxic (label = 1)
	•	Making mistakes — especially false positives (flagging a non-toxic comment as toxic) — has real-world impact on user experience and trust

we have to look at predictions for which model predicted correctly for toxicity as non-toxicity & also where the model did't predicted correctly for toxicity & non-toxicity, thats why we used F1 





Testing model on Test.csv
What This Tells You:
	•	The model generalizes well — it performs consistently across training, validation, and test sets.
	•	It’s ready for real-world usage, or for further stages like:
	•	Interpretability (SHAP/IG)
	•	Adversarial robustness
	•	False positive filtering
	•	Demo app interface