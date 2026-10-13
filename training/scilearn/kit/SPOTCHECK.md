# Kit table spot check (kit_2.57)

15 heroes chosen by a fixed seeded hash (sha256 of "kit-2.57-spotcheck-v1|<hero>", lowest 15). Kit file sha256: `8da4f2715fec3c43717fc30a895b188ae55578913cf0bc5fa0363e03398f6ad9`.

Values are the mean of three coders (0 to 4; early/late -2 to +2). A `*` marks an axis where the coders' SD is above 1.0. Columns: hCC hard CC, peel soft CC and peel, heal ally healing, self self-sustain and shields, burst, sust sustained damage, range, mob mobility, wave waveclear, siege, global global presence, e/l early (-) vs late (+).

Max: mark each row OK, or note the axis and the value you would expect.

| Hero | Role | hCC | peel | heal | self | burst | sust | range | mob | wave | siege | global | e/l | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Abathur | Support | 1.00 | 1.33 | 2.33 | 1.00 | 1.33 | 1.67 | 1.67 | 2.00 | 2.33 | 2.67 | 3.67 | 0.33 | |
| Alarak | Melee Assassin | 3.00 | 1.00 | 0.00 | 2.00 | 3.00 | 3.00 | 0.33 | 2.00 | 2.00 | 0.00 | 0.00 | 1.00 | |
| Artanis | Bruiser | 1.00 | 1.67 | 0.00 | 3.00 | 2.67 | 2.67 | 0.67 | 3.00 | 1.33 | 0.00 | 1.33 | 0.00 | |
| Dehaka | Bruiser | 2.33 | 1.67 | 0.00 | 4.00 | 2.00 | 3.00 | 0.00 | 2.00 | 2.00 | 0.00 | 1.33 | 0.00 | |
| Gall | Ranged Assassin | 1.00 | 1.33 | 0.00 | 0.67 | 2.33 | 2.33 | 1.33 | 1.00 | 3.00 | 0.00 | 1.00 | 0.00 | |
| Genji | Ranged Assassin | 0.00 | 0.67 | 0.00 | 2.00 | 3.00 | 1.33 | 1.33 | 3.00 | 2.00 | 0.00 | 0.00 | 0.33 | |
| Hogger | Bruiser | 3.00 | 2.00 | 0.33 | 2.00 | 3.00 | 2.67 | 1.00 | 2.00 | 3.00 | 0.00 | 0.00 | 0.00 | |
| Kel'Thuzad | Ranged Assassin | 3.00 | 2.00 | 0.00 | 1.00 | 3.00 | 3.00 | 2.00 | 0.00 | 3.33 | 0.33 | 1.00 | 1.33 | |
| Malfurion | Healer | 2.67 | 1.67 | 3.33 | 0.67 | 1.00 | 1.33 | 2.00 | 0.00 | 2.33 | 0.00 | 0.00 | 0.00 | |
| Probius | Ranged Assassin | 0.67 | 2.67 | 1.00 | 1.00 | 2.67 | 2.33 | 2.67 | 1.33 | 3.67 | 1.67 | 1.00 | 1.00 | |
| Rehgar | Healer | 0.00 | 3.00 | 4.00 | 1.00 | 1.00 | 1.67 | 1.00 | 1.33 | 1.33 | 0.00 | 0.67 | 0.00 | |
| Samuro | Melee Assassin | 0.00 | 0.00 | 0.00 | 2.33 | 2.00 | 3.67 | 0.00 | 2.67 | 1.33 | 1.00 | 0.67 | 0.00 | |
| The Butcher | Melee Assassin | 2.00 | 2.00 | 0.00 | 2.33 | 3.00 | 3.33 | 0.00 | 2.00 | 2.00 | 0.00 | 0.00 | 1.67 | |
| Tyrande | Healer | 1.33 | 1.67 | 3.00 | 1.00 | 1.67 | 1.00 | 2.67 | 0.33 | 1.00 | 0.00 | 1.00 | 1.00 | |
| Yrel | Bruiser | 2.33 | 2.00 | 1.00 | 3.33 | 2.33 | 2.00 | 0.33 | 2.33 | 2.00 | 0.00 | 0.00 | -0.33 | |

Per-coder raw values for these heroes are in kit_2.57.json under heroes.<name>.raw.
