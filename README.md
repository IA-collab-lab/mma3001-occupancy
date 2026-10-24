# Predicting room occupancy (Building 69, G.57)

MMA3001 individual project. Predicts whether a monitored room will be occupied 1 hour ahead (and optionally 24 hours ahead) from historical occupancy sensor events.

## Acceptance criterion

The gradient boosting model passes if its test F1 exceeds the stronger baseline's test F1 by at least 0.03. The margin was chosen after viewing validation results (persistence F1 0.719, gradient boosting F1 0.764) and before the test period was scored.
