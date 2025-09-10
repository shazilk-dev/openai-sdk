from pydantic import BaseModel, Field, field_validator, EmailStr
from pydantic_settings import BaseSettings

# class User(BaseModel):
#     name: str
#     age: int
#     is_active: bool = True  # Default value

# user = User(name="Shazil", age=21)
# # user = User(name="Shazil", age="21") # it automatically converts 21 into integer
# # user = User(name="Shazil", age="twenty") # now it raises an error
# print(user.name)  # Output: "Alice"
# print(user.age)   # Output: 25
# print(user.is_active)  # Output: True (default)


# extra rules
class Product(BaseModel):
    name: str
    price: float = Field(..., gt=0)
    category: str = Field(min_length=3, max_length=50)

# product = Product("Laptop", "0", "electronics") # This will raise a error bcz it requires keyword arguments but we provided positional arguments and.. the error is 4 position arguments were given but 1 expected means :
# Why it says "4 positional arguments were given"

# When you call Product("Laptop", "0", "electronics"), Python implicitly calls the class init with the instance as the first positional argument. Under the hood it's like: Product.init(self, "Laptop", "0", "electronics")
# That call passes 4 positional values: the implicit self plus the three strings you provided. So Python reports "4 were given".
# The error text says BaseModel.init() "takes 1 positional argument" because BaseModel.init only accepts the instance (self) as a positional argument — it does not accept field values positionally. Pydantic expects field values as keyword arguments (or to be passed through its validation APIs), so positional field values are rejected with that TypeError before any validation runs.

product = Product(name="Laptop", price=1000, category="electronics")
print(type(product))



# Nested Models
class Address(BaseModel):
    city: str
    country: str

class UserProfile(BaseModel):
    name:str
    address:Address

user_profile = UserProfile(
    name = "Shazil",
    address={"city":"Karachi", "country":"Pakistan"}
)

# print(type(user_profile))
# print(user_profile)

# user_profile_json = user_profile.model_dump_json()
# print(user_profile_json)
# print(type(user_profile_json))

# new_user = user_profile.model_validate_json('{"name":"Shazil","address":{"city":"Karachi","country":"Pakistan"}}')
# print(new_user)


# pydantic setting

# class Settings(BaseSettings):
#     api_key:str
#     debug: bool = False

#     class Config:
#         env_file = ".env"


# setting = Settings()
# print(setting)


#  Validators for Custom Rules
# class User(BaseSettings):
#     name: str
#     email: EmailStr

#     @field_validator("name")
#     def name_must_not_be_empty(cls, value):
#         print( not False)
#         if not value.strip():
#             print('error')
#             # raise ValueError("name cannot be empty")
#         return value

# # user = User(name="Shazil", email="shazil.akn@gmail.com")
# user = User(name="", email="shazil.akn@gmail.com")

# print(user)


from pydantic import BaseModel

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

output = HomeworkOutput(is_homework=True, reasoning="This looks like math practice")
print(type(output))
print(f'.dict(): {output.dict()}\n')
print(f'.model_dump(): {output.model_dump()}\n')
print(f'.model_dump_json(): {output.model_dump_json()}\n')
print(f'.json(): {output.json()}\n')
