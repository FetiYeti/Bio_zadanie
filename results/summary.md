# Výsledky – detekcia voice changera

Nahrávok: 1400 (pravé 700, zmenené 700), hovoriaci: awb, kal, rms, slt, príznakov: 131


## P0 – známy hovoriaci (5-násobná CV podľa vety; scenár verifikácie 1:1)

| klasifikator   |    EER |    AUC |   presnost |
|:---------------|-------:|-------:|-----------:|
| LogReg         | 2.1429 | 0.9972 |     0.9793 |
| SVM-RBF        | 0.9286 | 0.9989 |     0.9929 |
| RandomForest   | 2.6429 | 0.997  |     0.9714 |
| HistGB         | 2.2143 | 0.9971 |     0.9786 |



## P1 – neznámy hovoriaci (LOSO)

EER [%] podľa testovaného hovoriaceho:

| klasifikator   |   SPOLU |   awb |   kal |   rms |   slt |
|:---------------|--------:|------:|------:|------:|------:|
| HistGB         |   14.64 |  7.86 | 10.18 |  4.29 | 33.21 |
| LogReg         |   17.29 |  3.93 |  6.43 |  3.93 | 39.64 |
| RandomForest   |   11.79 |  4.29 |  7.32 |  3.57 | 23.93 |
| SVM-RBF        |   19.29 |  5.36 |  9.82 |  3.93 | 37.14 |


Súhrn (všetky foldy spolu):

| klasifikator   |     EER |    AUC |   presnost |
|:---------------|--------:|-------:|-----------:|
| LogReg         | 17.2857 | 0.8974 |     0.8371 |
| SVM-RBF        | 19.2857 | 0.8897 |     0.6557 |
| RandomForest   | 11.7857 | 0.9436 |     0.7    |
| HistGB         | 14.6429 | 0.9242 |     0.775  |


Najlepší klasifikátor: **RandomForest**


Úspešnosť podľa typu (RandomForest, prah = EER prah 0.700; pri pravom hlase = správne prijaté):

| typ_sk                 |   n |   uspesnost |   EER_vs_pravy |
|:-----------------------|----:|------------:|---------------:|
| prevzorkovanie         | 100 |      0.49   |         0.2943 |
| posun výšky            | 100 |      0.86   |         0.1286 |
| pravý hlas             | 700 |      0.8843 |       nan      |
| výška + efekt          | 100 |      0.9    |         0.1007 |
| posun formantov        | 100 |      0.93   |         0.0907 |
| fázový vokóder         | 100 |      0.98   |         0.05   |
| výška + zach. formanty | 100 |      1      |         0.055  |
| robot                  | 100 |      1      |         0.0093 |


Podľa prenosového kanála:

| kanal     |   n |    EER |    AUC |
|:----------|----:|-------:|-------:|
| clean     | 271 | 0.0993 | 0.9623 |
| mp3       | 298 | 0.0774 | 0.9718 |
| noise     | 259 | 0.1507 | 0.9194 |
| opus      | 286 | 0.1434 | 0.9293 |
| telephone | 286 | 0.1435 | 0.9448 |



## P2 – neznámy typ voice changera (leave-one-attack-out, zároveň neznámy hovoriaci)

| typ_sk                 |    EER |    AUC |   detegované (prah 0.5) |
|:-----------------------|-------:|-------:|------------------------:|
| posun formantov        | 0.1156 | 0.9555 |                  0.9875 |
| výška + efekt          | 0.1379 | 0.9065 |                  0.9625 |
| fázový vokóder         | 0.0563 | 0.9783 |                  1      |
| výška + zach. formanty | 0.0545 | 0.9793 |                  1      |
| posun výšky            | 0.1223 | 0.9068 |                  0.9375 |
| prevzorkovanie         | 0.2866 | 0.7775 |                  0.7625 |
| robot                  | 0.042  | 0.984  |                  1      |



## P3 – ablácia skupín príznakov (LOSO, RandomForest)

| skupina                                 |   pocet |    EER |    AUC |   EER_awb |   EER_kal |   EER_rms |   EER_slt |
|:----------------------------------------|--------:|-------:|-------:|----------:|----------:|----------:|----------:|
| kepstrálne (MFCC+LFCC)                  |      92 | 0.1536 | 0.922  |    0.075  |    0.1339 |    0.0357 |    0.2464 |
| spektrálne + LTAS                       |      34 | 0.2229 | 0.8491 |    0.3393 |    0.1357 |    0.0679 |    0.2929 |
| prozódia (F0, jitter, periodicita)      |      10 | 0.4486 | 0.566  |    0.5    |    0.45   |    0.1643 |    0.4214 |
| formanty + F0/formanty                  |      10 | 0.39   | 0.6254 |    0.5179 |    0.2911 |    0.3786 |    0.4857 |
| fáza                                    |       9 | 0.34   | 0.6824 |    0.3429 |    0.3929 |    0.5    |    0.3429 |
| modulácie                               |       5 | 0.4107 | 0.6162 |    0.4071 |    0.4643 |    0.3643 |    0.3643 |
| harmonická štruktúra (HNR, CPP, cutoff) |      13 | 0.3886 | 0.6744 |    0.3    |    0.4161 |    0.2893 |    0.3107 |
| detektor (bez príznakov identity)       |     131 | 0.1179 | 0.9436 |    0.0429 |    0.0732 |    0.0357 |    0.2393 |
| VŠETKY vrátane príznakov identity       |     173 | 0.1471 | 0.9393 |    0.0607 |    0.0554 |    0.025  |    0.2214 |



## P4 – určenie typu voice changera (8 tried, LOSO)
Presnosť 63.57 %, macro-F1 0.5285


## P5 – odhad posunu výšky hlasu (LOSO)
MAE 3.44 poltónu, správny smer posunu 77.6 %, odchýlka do ±2 poltónov 44.2 %

MAE podľa typu: posun výšky 3.27, výška + zach. formanty 3.61, prevzorkovanie 3.89, fázový vokóder 3.18, výška + efekt 3.23

Odhad posunu formantov (v poltónových jednotkách, všetky typy okrem robota): MAE 3.03
