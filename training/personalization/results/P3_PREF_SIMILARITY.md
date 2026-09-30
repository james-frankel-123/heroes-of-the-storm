# Hero similarity by play preference

26,122 players with 100+ games (April 2024 to May 2026). Correlation across players of volume-centered log(1 + games) per hero. Split-half stability of the whole matrix: r = 0.98. A rank-12 embedding explains 35% of the variance.

## Clusters (average linkage on embedding cosine)

- Stitches, Diablo, Garrosh, Tyrael, Johanna, Muradin, Anub'arak, E.T.C., Uther, Kharazim, Varian, Arthas, Mal'Ganis
- Falstad, Hanzo, Greymane
- Thrall, Sonya, Imperius, Malthael, Leoric, Dehaka, Hogger, Blaze
- Deckard, Stukov, Malfurion, Tyrande, Auriel, Ana, Alexstrasza, Whitemane, Li Li, Lt. Morales, Lúcio, Rehgar, Anduin, Brightwing
- Illidan, Samuro, Alarak, Kel'Thuzad, Tracer, Medivh, Zeratul, Genji, Kerrigan, Maiev, Rexxar, Chen, Yrel, Sgt. Hammer, Fenix, Probius, Zarya, Gall, The Lost Vikings, Cho, Abathur, Murky, The Butcher, Nova, Valeera
- Li-Ming, Junkrat, Mephisto, Chromie, Orphea, Qhira, Jaina, Lunara, Sylvanas, Cassia
- Gul'dan, Kael'thas, Tassadar, Raynor, Tychus, Valla, Zul'jin
- Mei, Deathwing, Gazlowe, D.Va, Nazeebo, Zagara, Azmodan, Artanis, Ragnaros, Xul

## Strongest pairs

| hero | hero | r |
|---|---|---|
| Anduin | Brightwing | +0.39 |
| Hanzo | Genji | +0.33 |
| Cho | Gall | +0.31 |
| Brightwing | Rehgar | +0.30 |
| Genji | Maiev | +0.28 |
| Anub'arak | Johanna | +0.28 |
| Auriel | Anduin | +0.28 |
| Zeratul | Genji | +0.28 |
| Anduin | Rehgar | +0.27 |
| Auriel | Brightwing | +0.26 |
| Zeratul | Maiev | +0.25 |
| Alexstrasza | Whitemane | +0.25 |
| Probius | Cho | +0.25 |
| Nazeebo | Azmodan | +0.25 |
| The Lost Vikings | Cho | +0.24 |
| Probius | The Lost Vikings | +0.24 |
| Muradin | Johanna | +0.23 |
| Diablo | Garrosh | +0.23 |
| Anub'arak | E.T.C. | +0.23 |
| Anduin | Stukov | +0.23 |
| Probius | Gall | +0.23 |
| Abathur | Murky | +0.22 |
| Alarak | Kel'Thuzad | +0.22 |
| Nova | Valeera | +0.22 |
| Zeratul | Alarak | +0.22 |

## Strongest cross-role pairs

| hero | hero | r |
|---|---|---|
| Hanzo | Genji | +0.33 |
| Cho | Gall | +0.31 |
| Genji | Maiev | +0.28 |
| Zeratul | Genji | +0.28 |
| Probius | Cho | +0.25 |
| The Lost Vikings | Cho | +0.24 |
| Probius | The Lost Vikings | +0.24 |
| Abathur | Murky | +0.22 |
| Alarak | Kel'Thuzad | +0.22 |
| Nova | Valeera | +0.22 |
| Nova | The Butcher | +0.21 |
| Rexxar | The Lost Vikings | +0.21 |
| Fenix | Probius | +0.20 |
| The Lost Vikings | Gall | +0.20 |
| Murky | Nova | +0.20 |

## Most negative pairs (rarely the same players)

| hero | hero | r |
|---|---|---|
| Hanzo | Lt. Morales | -0.17 |
| Thrall | Probius | -0.17 |
| Johanna | Kel'Thuzad | -0.17 |
| Nova | Johanna | -0.17 |
| Anduin | Illidan | -0.18 |
| Anduin | Zeratul | -0.18 |
| Li Li | Greymane | -0.18 |
| Brightwing | Alarak | -0.18 |
| Anduin | The Butcher | -0.18 |
| Li Li | Hanzo | -0.21 |
