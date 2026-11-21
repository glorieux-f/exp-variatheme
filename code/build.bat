@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: python .\keywords.py --vocab all --scorer bm25 g2 tfidf  tfidfa0.75 tfidfa0.9 tfidfa1.1 tfidfa1.5 tfidfa2 txm --top 100  ..\data\3_contingency\ddr- ..\results\keywords100\


::python .\keywords-matrix.py ..\results\keywords100\ ..\results\keywords100\matrix.tsv
:: python .\pcoa.py ..\results\keywords100\matrix.tsv
:: python .\keywords_stats.py  --terms ..\data\3_contingency\ddr- --stopwords .\stopwords.txt --top 100 --stopwords-top 20 -o ..\results\keywords100-stats.tsv  "..\results\keywords100\ddr-keywords100-all-*.txt"


python .\keywords-glob.py ..\data\3_contingency\ddr- "ddr19770721aena*" ..\results\keywords\ --scorer tfidf simplemaths simpleridf fisher --vocab all 