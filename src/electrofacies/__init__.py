"""
Electrofacies — Production-quality electrofacies prediction for the Delaware Mountain Group.

Architecture:
    data/wells/inbox/  -->  [Ingest]  -->  [Transform & Validate]  -->  [Predict]  -->  outputs/
                             LAS/CSV       Mnemonic mapping            RF / XGBoost      CSV + LAS + Plots
                                           Physical range QC           Tier routing       Confidence + QC
                                           Completeness check          OOD detection      Batch summaries
"""

__version__ = "1.0.0"
