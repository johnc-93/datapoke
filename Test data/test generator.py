# Save a CSV with Windows-1252 encoding (which includes invalid UTF-8 bytes)
text = (
    "ID,Name,Comment\n"
    "1,André,Great performance\n"
    "2,Jürgen,Service was okay – could be better\n"
    "3,Anaïs,Price is €50\n"
    "4,László,Wasn’t impressed\n"
    "5,Chloë,—\n"
)

# Save with cp1252 (Windows-1252) — not UTF-8
with open(r"load tests\utf-8_test_cp1252.csv", "w", encoding="cp1252") as f:
    f.write(text)
