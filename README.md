# DrivenData's Blood Donations Prediction Challenge
**In Progress**
One of the sites simpler datasets - from 4 columns, predict which donors will donate blood in March.

## Approach
Initial exploratory analysis conducted in EDA.ipynb with several features engineered.

Firstly experimented with the following classifiers from the Scikit-Learn library:
- RandomForestClassifier
- GradientBoostingClassifier
- MultiLayerPerceptronClassifier

Best model was a GradientBoostingClassifier.
**DrivenData Score: 0.4911. Rank: 376**

Improved upon this with a single layer Keras Neural Network.
**DrivenData Score: 0.4376. Rank: 77**

## Future
Continue experimenting with ANN's to improve upon score. 
