from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = "Sushil"    # Default value for name is set to "Sushil"
    age: Optional[int] = None   # age is optional and can be None if not provided. Its required to provide a default value for optional fields in pydantic models, otherwise it will throw an error if the field is not provided during instantiation.
    email: Optional[EmailStr] = None     # EmailStr is a special type provided by pydantic that validates if the input is a valid email address.
    # cgpa: float = Field(..., ge=0.0, le=10.0, default=5, description="CGPA must be between 0.0 and 10.0")   # cgpa is a required field (indicated by ...) and must be between 0.0 and 10.0 inclusive. If not provided, it will default to 5. The description is just metadata that can be used for documentation purposes.
    cgpa: float = Field(ge=0.0, le=10.0, default=5, description="CGPA must be between 0.0 and 10.0")   # cgpa must be between 0.0 and 10.0 inclusive. If not provided, it will default to 5. The description is just metadata that can be used for documentation purposes.


student_data = {
    "name": "John Doe",
    "age": 20,
    "email": "john.doe@example.com",
    "cgpa": 8.5
}   

student = Student(**student_data)

print("Student 0: ", student)

print("Student 1: ",Student(**{"name": "John Snow", "age": '33'})) 

print("Student 2: ",Student(**{"age": '33', "cgpa": 9.0}))

print("Student 3: ",Student(**{"age": 15, "cgpa": 9.0}).model_dump())  # model_dump() is a method provided by pydantic models that returns the model data as a dictionary. This is useful for serializing the model data or for debugging purposes.

print("Student 4: ",Student(**{"age": '33', "email": "sushil@example.com"}).model_dump_json())  # model_dump_json() is a method provided by pydantic models that returns the model data as a JSON string. This is useful for serializing the model data to JSON format, which can be easily transmitted over the network or stored in a file.

