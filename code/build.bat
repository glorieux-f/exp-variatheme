@echo off
setlocal EnableExtensions EnableDelayedExpansion
:: Compare some scorers
:: g2idfa0.1 g2idfa0.2 g2idfa0.5 g2idfa0.75
python .\keywords.py --vocab all --scorer  g2 g2idf  poissondev poissonidf subtfidf subtfidfa0.3 subtfidfa0.5 subtfidfa0.75 tfidf tfidfa0.5 tfidfa0.75 tfidfa0.9 tfidfa1.1 tfidfa1.5 tfidfa2 txm --top 100  ..\data\3_contingency\ddr- ..\results\keywords100\
python .\keywords-matrix.py ..\results\keywords100\ ..\results\keywords100\matrix.tsv
 python .\pcoa.py ..\results\keywords100\matrix.tsv
