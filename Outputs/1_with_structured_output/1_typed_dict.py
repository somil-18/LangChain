from typing import TypedDict

# like a blueprint for the dictionary
class Person(TypedDict):
    name: str
    age: int 

new_dict: Person = {'name': 'Somil', 'age': 19}
print(new_dict)

# no validation at runtime
new_person: Person = {'name': 17, 'age': '24'}
print(new_person)
