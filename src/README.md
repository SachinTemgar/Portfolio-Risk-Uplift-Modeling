# src/

This folder is reserved for production-ready modular code.

Planned modules:
- data_prep.py       - cleaning and feature engineering functions
- experiment_design.py - PSM and treatment simulation functions  
- uplift_models.py   - T-Learner, X-Learner, Causal Forest wrappers
- metrics.py         - Qini curve, AUUC, uplift decile functions

Currently the logic lives inside the notebooks. 
Refactoring into modules is planned for v2.0 of this project.