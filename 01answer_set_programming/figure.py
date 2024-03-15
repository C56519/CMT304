
import pandas as pd
df_template = pd.DataFrame(index=[f"req{i}" for i in range(1, 9)], columns=[f"mem{i}" for i in range(1, 4)]).fillna('')
# Process and create a table for Answer 3 based on the provided assignments
n_table = 3
assignments_answer_5 = [
    ("mem1", "req1"), ("mem2", "req1"), ("mem1", "req2"), ("mem3", "req2"),
    ("mem1", "req3"), ("mem2", "req3"), ("mem1", "req4"), ("mem3", "req4"),
    ("mem1", "req5"), ("mem2", "req5"), ("mem2", "req6"), ("mem3", "req6"),
    ("mem1", "req7"), ("mem2", "req7"), ("mem2", "req8"), ("mem3", "req8")
]

assignments_answer_3 = [
    ("mem2", "req1"), ("mem2", "req7"), ("mem2", "req8"), ("mem1", "req1"),
    ("mem1", "req2"), ("mem3", "req2"), ("mem1", "req3"), ("mem3", "req3"),
    ("mem1", "req4"), ("mem3", "req4"), ("mem1", "req5"), ("mem2", "req5"),
    ("mem2", "req6"), ("mem3", "req6"), ("mem3", "req7"), ("mem3", "req8")
]

assignments_answer_2 = [
    ("mem1", "req1"), ("mem1", "req2"), ("mem1", "req3"), ("mem1", "req4"),
    ("mem1", "req5"), ("mem1", "req7"), ("mem2", "req1"), ("mem2", "req5"),
    ("mem2", "req6"), ("mem2", "req7"), ("mem2", "req8"), ("mem3", "req2"),
    ("mem3", "req3"), ("mem3", "req4"), ("mem3", "req6"), ("mem3", "req8")
]


assignments_finally_answer = [
    ("mem1", "req1"), ("mem1", "req2"), ("mem1", "req3"), ("mem1", "req4"),
    ("mem1", "req5"), ("mem1", "req7"), ("mem2", "req1"), ("mem2", "req3"),
    ("mem2", "req5"), ("mem2", "req6"), ("mem2", "req8"), ("mem3", "req2"),
    ("mem3", "req4"), ("mem3", "req6"), ("mem3", "req7"), ("mem3", "req8")
]

# Initializing an empty dataframe for Answer 3 from the template and filling it
df_answer_3 = df_template.copy()
for mem, req in assignments_finally_answer:
    df_answer_3.at[req, mem] = 1

print(f"Answer{n_table}")
print(df_answer_3)
