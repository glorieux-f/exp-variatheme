@echo off
setlocal EnableExtensions EnableDelayedExpansion
:: Compare some scorers
python .\keywords.py --vocab all --scorer bm25 poissondev poissonidfa0.5 poissonidfa0.75 poissonidfa0.9 poissonidfa1 poissonidfa1.1  poissonidfa1.5 poissonidfa2  g2 subtfidf subtfidfa0.3 subtfidfa0.5 subtfidfa0.75 tfidf  tfidfa0.75 tfidfa0.9 tfidfa1.1 tfidfa1.5 tfidfa2 txm --top 100  ..\data\3_contingency\ddr- ..\results\keywords100\
python .\keywords-matrix.py ..\results\keywords100\ ..\results\keywords100\matrix.tsv
python .\pcoa.py ..\results\keywords100\matrix.tsv
python .\keywords_stats.py  --terms ..\data\3_contingency\ddr- --stopwords .\stopwords.txt --top 100 --stopwords-top 20 -o ..\results\keywords100-stats.tsv  "..\results\keywords100\ddr-keywords100-all-*.txt"
:: python .\keywords-glob.py ..\data\3_contingency\ddr- "ddr19721100ao*" ..\results\keywords\ --scorer tfidf subtfidfa0.3 poissonidfa0.9  poissonidfa1 poissonidfa1.1  poissonidfa1.5  txm poissondev g2 g2pos --vocab all  