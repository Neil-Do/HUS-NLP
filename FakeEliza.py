import re
print("I am Fake Eliza.")
s = input("You: ")
while s != "":
    if re.search(r"(^|\W)[aA]ll\W", s) != None:
        print("Fake Eliza: In what way?")
    elif re.search(r"\W[aA]lways\W", s) != None:
        print("Fake Eliza: Can you think of a specific example?")
    elif re.search(r"I am (depressed|sad)", s):
        print("Fake Eliza: I am sorry to hear you are depressed.")
    else:
        print("Not Found.")
    s = input("You: ")
