from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str
    age: int = 18 # default value
    accounts: Optional[int] = None # none if not provided
    email: EmailStr # validate E-mail address
    cgpa: float = Field(gt=0, lt=10, default=5, description="A decimal value representing cgpa of student") # cgpa >=0 and less than = 10

# pydantic can do type-coercion (explicitly conversion of one data type to another) whenever it's possible
# for e.g. here age which is a str is explicitly converted to int
new_student = {'name': 'somil', 'age': '69', 'email': 'abc@gmail.com', 'cgpa': 6}

# this will give ERROR
# new_student = {'name': 19, 'email': 'abc', 'cgpa': 12}

student = Student(**new_student)

# returns pydantic object
print(student)

# dictionary
print(student.model_dump())

# JSON
print(student.model_dump_json())

