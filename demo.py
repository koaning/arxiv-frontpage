import questionary 

a = questionary.select(
    "Which label do you want to annotate?",
    choices=["a", "b", "c"],
).ask()

b = questionary.select(
    "Which tactic do you want to apply?",
    choices=["1", "2", "3"],
).ask()

c = questionary.select(
    "Which tactic do you want to apply?",
    choices=["i", "ii", "iii"],
).ask()

print(a, b, c)