COLORS = ("R", "G","Y")
regions = ("A", "B", "C", "D")
adjacency = {
 ("A", "B"), ("A", "C"), ("B", "C"),
 ("B", "D"), ("C", "D"),
 }
 
def different(x, y):
   return 1 if x != y else 0



from itertools import product
from textwrap import indent

def score(value):
   return all(
      different(value[x],value[y])
      for (x,y) in adjacency
   )


solutions = []

for v in product(COLORS,repeat=len(regions)):
   t = dict(zip(regions,v))

   if score(t):
      solutions.append(t)


print(solutions,sep="\n")
print(f"Num of solution:{len(solutions)}")
