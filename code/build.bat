@echo off
setlocal EnableExtensions EnableDelayedExpansion
:: Compare G2 to fullG2
python .\keywords.py --vocab all --scorer g2 subtfidf subtfidfa0.3 simpleg2 bm25 txm tfidf bm25k100b1 g2idf --top 100  ..\data\3_contingency\ddr- ..\results\keywords100\