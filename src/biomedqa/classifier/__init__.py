"""Fine-tuned PubMedBERT yes / no / maybe classifier.

Pure helpers live in :mod:`biomedqa.classifier.core`; torch / transformers code lives in
:mod:`biomedqa.classifier.train` and is imported lazily, so CI does not need GPU libraries.
"""
