"""
Feature Engineering Subpackage for Facial Expression Recognition (FER).

Replication of Kar et al. (2021):
"A hybrid feature descriptor with Jaya optimised least squares SVM for
facial expression recognition", IET Image Processing.
"""

from .features import swt_haar_level1, phog, extract_features

__all__ = ["swt_haar_level1", "phog", "extract_features"]
