# Atmospheric Particle Event Classifier (NPF)

This repository contains the codebase and technical report for classifying New Particle Formation (NPF) events using atmospheric measurement data from a forestry field station.

### Project Overview

The objective was to train a classifier to predict whether a new particle formation event happens on a given day (binary classification) and, if so, what kind of event it is (multi-class prediction). 

To ensure the model learned the underlying physical chemistry rather than simply memorizing temporal patterns, all time and date information was strictly excluded from the training data. Furthermore, instead of splitting the task into two separate models, the architecture was designed to handle both binary and multi-class predictions within a single, unified model.

### Full Methodology & Results
**Please refer to the attached PDF report in this repository for the complete data analysis, methodology, and scientific conclusions.** 

The Python source code used for data processing and model training is housed in `src` for the sake of reproducibility, though the main deliverable of this project is the report `NPF-Project-Report.pdf`. 