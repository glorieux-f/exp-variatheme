@echo off
setlocal EnableExtensions EnableDelayedExpansion
:: Compare G2 to fullG2
python .\keywords.py --vocab all --scorer g2 fullg2 --top 100  ..\data\3_contingency\ddr- ..\results\keywords100\