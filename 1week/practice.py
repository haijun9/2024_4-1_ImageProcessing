a = 10
b = 10.0
c = 'Hello, Image processing'

print (f"a: {a}, a type: {type(a)}")
print (f"b: {b}, a type: {type(b)}")
print (f"c: {c}, a type: {type(c)}")

a = 3
b = 5

if a == b:
    print("a와 b는 같다.")
else:
    print("a와 b는 같지 않다.")

# 한 줄 주석입니다.
"""
여러
줄
주석입니다.
"""
'''
여러
줄
주석입니다.
'''

list0 = list()
list1 = [1, 3, 5, 7]
list2 = [2.0, 4.0, 6.0]
list3 = ["Test1", "Test2"]
list4 = ["University", 3, "Student", 2.0]

print(list0)
print(list1)
print(list2)
print(list3)
print(list4)

list1 = [1, 3, 5, 7, 9]
print(f'list1[0]: {list1[0]}, list1[2]: {list1[2]}, list1[4]: {list1[4]}')
print(f'list1[-1]: {list1[-1]}, list1[-3]: {list1[-3]}, list1[-5]: {list1[-5]}')

print(f'list1[:]: {list1[:]}')
print(f'list1[1:4]: {list1[1:4]}')
print(f'list1[:-1]: {list1[:-1]}')

list1.append(11)
print(f'list1.append(11) 결과: {list1}')
list1.insert(3, -1) # insert(index, value)
print(f'list1.insert(3, -1) 결과: {list1}')

tuple1 = (1, 2, 3, 4, 5)
print(f'tuple1: {tuple1}')

dic1 = {"사과": 700, "배": 500}
print(f'dic1: {dic1}')
print(f"dic1['사과']: {dic1['사과']}")

print(10 / 3)
print(10 // 3)
print(10 % 3)
print(10 ** 3)

a = 10
if a < 0:
    print("음수입니다.")
elif a % 2 == 0:
    print("짝수입니다.")
else:
    print("홀수입니다.")

for i in range(10):
    print(i, end=' ')
print()
for i in range(5, 10):
    print(i, end=' ')
print()
for i in range(0, 10, 2):
    print(i, end=' ')
print()
list1 = [1, 3, 5, 7, 9]
for i in list1:
    print(i, end=' ')
print()

list1 = [1, 2, 3]
dic1 = {'a': 1, 'b': 2}
print(f'len(list1): {len(list1)}, len(dic1): {len(dic1)}')
print(f'type(list1): {type(list1)}, type(dic1): {type(dic1)}')

def calc_add(a, b):
    return a + b
print(f'calc_add(1, 3): {calc_add(1, 3)}')