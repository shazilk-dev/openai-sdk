# # to_dict:

# def to_dict[T,U](keys:list[T], values:list[U]) -> dict[T,U]:
#     zipped = zip(keys,values)
#     print(zipped)
#     return dict(zipped)


# print(to_dict(["a", "b"], [1, 2]))  


# # invert_dict:

# # def invert_dict[T,U](dic_item: dict[T,U]) -> dict[U,T]:
# #     return dict((v, k) for k, v in dic_item.items())

# ## dict comprehension version:
# def invert_dict[T, U](dic_item: dict[T, U]) -> dict[U, T]:
#     return {v: k for k, v in dic_item.items()}

# print(invert_dict({"a": 1, "b": 2}))



# # merge_dicts:
# def merge_dicts[T,U](d1: dict[T,U], d2:dict[T,U]) -> dict[T,U]:
#     """
#     Merge two dictionaries. If there are overlapping keys, values from d2 will override those from d1.
#     """
#     return {**d1, **d2}


# print(merge_dicts({"a": 1}, {"b": 2}))


# # find_max:
# def find_max[T](items: list[T]) -> T:
#     if not items:
#         raise ValueError("The list is empty.")
#     max_item = items[0]
#     for item in items[1:]:
#         if item > max_item:
#             max_item = item
#     return max_item

# print(find_max([3, 1, 4, 1, 5, 9]))



# swap two numbers
def swap[T](a: T, b: T) -> tuple[T, T]:
    return b, a


x, y = swap(10, 20)    
# a, b = swap("A", "B") 
a, b = swap("A", 20) 

print(x, y)
print(a, b)
print(type(a), type(b))


# generics classes

class Stack[T]:
    def __init__(self):
        self.items: list[T] = []

    def push(self, item: T) -> None:
        self.items.append(item)

    def pop(self) -> None:
        self.items.pop()

int_stack = Stack[int]()        
int_stack.push(10)
print(int_stack.pop())  # ✅ Returns int

str_stack = Stack[str]()
str_stack.push("hello")
print(str_stack.pop())  # ✅ Returns str