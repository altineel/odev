\ Model deter_v2
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  1000 var_x2..0 - 1000 var_x1..0 - 1200 var_x1..1 + 0 var_x2..1
   + 7000 var_b..0 + 1200 var_x2..1 - 800 var_x1..2 + 0 var_x2..2
   + 5000 var_b..1 + 800 var_x2..2 - 1100 var_x1..3 + 0 var_x2..3
   + 1000 var_b..2 + 1100 var_x2..3 + 0 var_x2..4 + 3500 var_b..3
Subject To
 R0: var_x2..0 - var_x1..0 >= 0
 R1: var_x2..0 - var_x1..1 = 30
 R2: - var_x1..1 + var_x2..1 >= 0
 R3: var_x2..1 - var_x1..2 = 25
 R4: - var_x1..2 + var_x2..2 >= 0
 R5: var_x2..2 - var_x1..3 = 30
 R6: - var_x1..3 + var_x2..3 >= 0
 R7: var_x2..3 - var_x1..4 = 15
 R8: var_x1..0 = 0
 GC0: var_b..0 = 1 -> var_x2..0 - var_x1..0 >= 0.1
 GC1: var_b..0 = 0 -> var_x2..0 - var_x1..0 = 0
 GC2: var_b..1 = 1 -> - var_x1..1 + var_x2..1 >= 0.1
 GC3: var_b..1 = 0 -> - var_x1..1 + var_x2..1 = 0
 GC4: var_b..2 = 1 -> - var_x1..2 + var_x2..2 >= 0.1
 GC5: var_b..2 = 0 -> - var_x1..2 + var_x2..2 = 0
 GC6: var_b..3 = 1 -> - var_x1..3 + var_x2..3 >= 0.1
 GC7: var_b..3 = 0 -> - var_x1..3 + var_x2..3 = 0
Bounds
 var_x2..0 <= 150
 var_x1..0 <= 150
 var_x1..1 <= 150
 var_x2..1 <= 150
 var_x2..1 <= 150
 var_x1..2 <= 150
 var_x2..2 <= 150
 var_x2..2 <= 150
 var_x1..3 <= 150
 var_x2..3 <= 150
 var_x2..3 <= 150
 var_x1..4 <= 150
 var_x2..4 <= 150
Binaries
 var_b..0 var_b..1 var_b..2 var_b..3
End
