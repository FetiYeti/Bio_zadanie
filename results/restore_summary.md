# Odhalenie – obnova pôvodného hlasu

Testovaných zmenených nahrávok: 300 (bez robota; nepárne vety). Identifikácia hovoriaceho na PRAVÝCH nahrávkach: 100.0 %.

| režim         |   chyba F0 [pt] (medián) |   chyba F0 [pt] (priemer) |   identifikácia hovoriaceho [%] |
|:--------------|-------------------------:|--------------------------:|--------------------------------:|
| bez obnovy    |                    4.642 |                     4.619 |                          60.667 |
| naslepo       |                    2.762 |                     3.123 |                          72.333 |
| s referenciou |                    0.362 |                     0.622 |                          91.667 |
| orákulum      |                    0.137 |                     0.299 |                          91     |


## Presnosť odhadu parametrov voice changera

|               |   MAE posunu F0 [pt] |   MAE posunu formantov [pt] |
|:--------------|---------------------:|----------------------------:|
| naslepo       |                3.087 |                       3.013 |
| s referenciou |                0.65  |                       1.568 |


## Identifikácia hovoriaceho podľa typu voice changera [%]

|                        |   bez obnovy |   naslepo |   s referenciou |   orákulum |
|:-----------------------|-------------:|----------:|----------------:|-----------:|
| posun formantov        |           96 |        68 |              94 |         90 |
| výška + efekt          |           50 |        66 |              88 |         88 |
| fázový vokóder         |           52 |        78 |              90 |         92 |
| výška + zach. formanty |           62 |        72 |              84 |         86 |
| posun výšky            |           44 |        76 |              96 |         94 |
| prevzorkovanie         |           60 |        74 |              98 |         96 |

