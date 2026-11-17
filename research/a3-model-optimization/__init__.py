"""
Modul Eksperimen Model & Optimasi (Divisi A3) - TA-PCVK.
Mengimplementasikan:
- Baseline LDA + SVM-RBF
- Least Squares Support Vector Machine (LS-SVM)
- Algoritma Optimasi Metaheuristik (Jaya)
"""

from .lssvm import LSSVMClassifier, OneVsOneLSSVMClassifier, MulticlassLSSVM

__all__ = ["LSSVMClassifier", "OneVsOneLSSVMClassifier", "MulticlassLSSVM"]
