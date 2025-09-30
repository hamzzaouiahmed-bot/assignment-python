import sys
from utils import greet


username = "Ahmed"
age = 23
tasks = ["Learn Python", "Write Code", "Submit Homework"]
task_status = {"Learn Python": "Done", "Write Code": "In Progress"}


print("Enumerate Example:")
for index, task in enumerate(tasks):
    print(index, task)

print("Range Example:", list(range(3)))
print("ID Example:", id(username))


if age < 18:
    print("You are a minor.")
elif age < 30:
    print("You are a young adult.")
else:
    print("You are an adult.")


print("\nFor loop:")
for task in tasks:
    print("Task:", task)

print("\nWhile loop:")
count = 0
while count < 3:
    print("Counting:", count)
    count += 1


num_str = "100"
num_int = int(num_str)
print("Converted:", num_int, type(num_int))


print(greet(username))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Command line arguments:", sys.argv[1:])
    else:
        print("No command line arguments provided.")
