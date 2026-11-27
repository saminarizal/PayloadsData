## Tabel A. Model utama vs baseline (tanpa normalisasi operator); test set Kaggle 80:20 (n = 6161) dan 15332 string sqlmap

| Kelompok    | Model                 | Jenis                            |   Akurasi (%) |   Presisi (%) |   Recall (%) |   F1 (%) |   FPR (%) |   FP |   FN |   Deteksi sqlmap (%) |
|:------------|:----------------------|:---------------------------------|--------------:|--------------:|-------------:|---------:|----------:|-----:|-----:|---------------------:|
| Model utama | KNN                   | ML (TF-IDF char 1-3)             |         98.96 |         98.97 |        98.19 |    98.58 |      0.59 |   23 |   41 |                96.33 |
| Model utama | SVM                   | ML (TF-IDF char 1-3)             |         99.69 |         99.82 |        99.34 |    99.58 |      0.10 |    4 |   15 |                96.16 |
| Model utama | Random Forest         | ML (TF-IDF char 1-3)             |         99.68 |         99.96 |        99.16 |    99.56 |      0.03 |    1 |   19 |                94.21 |
| Model utama | XGBoost               | ML (TF-IDF char 1-3)             |         99.63 |         99.91 |        99.07 |    99.49 |      0.05 |    2 |   21 |                96.03 |
| Baseline    | Logistic Regression   | ML (TF-IDF char 1-3)             |         99.66 |         99.87 |        99.20 |    99.53 |      0.08 |    3 |   18 |                96.69 |
| Baseline    | Naive Bayes           | ML (TF-IDF char 1-3)             |         94.45 |         87.52 |        98.98 |    92.90 |      8.18 |  319 |   23 |                97.35 |
| Baseline    | libinjection          | Signature (tokenizer)            |         73.38 |         59.08 |        89.25 |    71.10 |     35.81 | 1397 |  243 |                84.16 |
| Baseline    | OWASP CRS v4.30.0 PL1 | Rule-based (20 aturan, ambang 5) |         73.74 |         59.04 |        92.79 |    72.16 |     37.30 | 1455 |  163 |                97.94 |
| Baseline    | OWASP CRS v4.30.0 PL2 | Rule-based (43 aturan, ambang 5) |         63.51 |         50.14 |        98.58 |    66.47 |     56.81 | 2216 |   32 |                99.94 |


## Tabel B. Sebelum vs sesudah normalize_sql_operators() pada data latih, test set Kaggle, dan payload sqlmap

| Model                 |   F1 sebelum (%) |   F1 sesudah (%) |   FPR sebelum (%) |   FPR sesudah (%) | p McNemar (Kaggle)   |   sqlmap sebelum (%) |   sqlmap sesudah (%) | p McNemar (sqlmap)   |   Δ sqlmap (pp) |
|:----------------------|-----------------:|-----------------:|------------------:|------------------:|:---------------------|---------------------:|---------------------:|:---------------------|----------------:|
| KNN                   |            98.58 |            98.74 |              0.59 |              0.85 | 0.281                |                96.33 |                99.52 | < 0.001              |            3.18 |
| SVM                   |            99.58 |            99.56 |              0.10 |              0.13 | 1.000                |                96.16 |                99.86 | < 0.001              |            3.70 |
| Random Forest         |            99.56 |            99.56 |              0.03 |              0.03 | 1.000                |                94.21 |                99.95 | < 0.001              |            5.75 |
| XGBoost               |            99.49 |            99.44 |              0.05 |              0.10 | 0.625                |                96.03 |                99.71 | < 0.001              |            3.68 |
| Logistic Regression   |            99.53 |            99.49 |              0.08 |              0.10 | 0.688                |                96.69 |                99.88 | < 0.001              |            3.19 |
| Naive Bayes           |            92.90 |            92.66 |              8.18 |              8.38 | 0.002                |                97.35 |                98.23 | < 0.001              |            0.88 |
| libinjection          |            71.10 |            71.10 |             35.81 |             35.81 | 1.000                |                84.16 |                84.16 | 1.000                |            0.00 |
| OWASP CRS v4.30.0 PL1 |            72.16 |            72.47 |             37.30 |             37.30 | < 0.001              |                97.94 |                97.98 | 0.118                |            0.05 |
| OWASP CRS v4.30.0 PL2 |            66.47 |            66.43 |             56.81 |             56.81 | 0.500                |                99.94 |                99.96 | 0.250                |            0.02 |


Catatan B: hyperparameter sama dengan kondisi tanpa normalisasi; split baris identik. Setelah normalisasi, 4 baris uji identik dengan baris latih dan 56 string sqlmap identik dengan teks Kaggle (lihat paired_tests_rev10.csv untuk angka tanpa baris tersebut).