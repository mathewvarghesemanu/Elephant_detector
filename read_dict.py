
import re
a_dictionary = {}
a_file = open("mapping.txt")
for line in a_file:
    key, value = line.split(":")
    key=re.findall(r'\d+',key)[0]
    key=int(key)

    a_dictionary[key] = value


print(a_dictionary[1])