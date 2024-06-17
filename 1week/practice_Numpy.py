import numpy as np

a = np.array([1, 2, 3], dtype=np.uint8) # 1차원 배열 생성, 자료형 unit8 0 ~ 255
print(a) # 배열의 원소 전부 출력
print(a.shape)
print(type(a)) # type

# np.arrange
arr1 = np.arange(10) # 0부터 9까지
print(arr1)

arr2 = np.arange(0, 10, 2) # 0부터 9까지 2간격으로
print(arr2)
print(arr2.shape)

# np.linspace
arr2 = np.linspace(0.5, 10 - 0.5, 10)
print(arr2)
print(len(arr2)) # 총 10개

# np.zeros
a = np.zeros((2,2)) # 모든 원소의 값이 0인 2 x 2 행렬 생성
print(a)
print(a.shape) # 행렬의 크기

# np.ones
b = np.ones((1,2)) # 모든 원소의 값이 1인 1 x 2 행렬 생성
print(b)
print(b.shape) # 행렬의 크기

# np.full
c = np.full((2,2), 7)
print(c)
print(c.shape)

# np.eye
d = np.eye(2)
print(d)
print(d.shape)

# 크기 변환 : reshape
a = np.arange(6).reshape(3, 2) # 1차원 -> 3 x 2 행렬로 변환
print(a)
print(a.shape)

# 붙이기 : concat
x = np.array([[1, 2, 3],
              [4, 5, 6]])
y = np.array([[7, 8, 9],
              [10, 11, 12]])
z = np.concatenate([x, y], axis=0) # 2차원 2개의 행렬을 행 축으로 연결함
print(z)

z = np.concatenate([x, y], axis=1) # 2차원 2개의 형렬을 열 축으로 연결함
print(z)

# Boolean indexing을 통한 원하는 값 뽑기.
a = np.array([[1, 2], [3, 4], [5, 6]])

bool_idx = (a > 2) # 해당 조건을 만족하는 값을 찾는다
                   # 해당 조건을 만족하면 True, 아니면 False를 원소 위치에 반환
print(bool_idx) # boolean mask라고도 불린다.
print(bool_idx.shape)

# print(a[bool_idx]) # a 에서 2보다 큰 값만 반환
print(a[a > 2])

# 기본적인 사칙 연산자 사용 시 모두 element-wise 연산이 적용
a = np.array([6, 12, 16])
b = np.array([3, 4, 4])

print(a + b)
print(a - b)
print(a * b)
print(a / b)

# array와 상수의 사칙 연산 또한 element-wise 연산이 적용
c = 10
print(a + c)

# 행렬 곱
a = np.array([[1, 0],
              [0, 1]]) # 단위 행렬 2 x 2
b = np.array([[5, 6],
              [7, 8]]) # 2 x 2 행렬

c = np.dot(a, b)
print(c)
print(c.shape)
d = a @ b
print(d)
print(d.shape)

# Transpose
a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(a)
print(a.shape)
a_ = a.T # 행과 열이 바뀜
print(a_)
print(a_.shape)

# min, max, sum
a = np.array([[1, 2, 3],
              [4, 5, 6]])
print("min : {}".format(np.min(a)))
print("max : {}".format(np.max(a)))
print("total sum : {}".format(np.sum(a)))

# numpy가 입력 데이터를 보고 타입을 결정
x = np.array([1, 2])
y = np.array([1.0, 2.0])
# 사용자가 데이터 타입을 지정
z = np.array([1, 2], dtype=np.int64)

print(x.dtype, y.dtype, z.dtype)

# 타입 변환 : astype 사용
src = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(src.dtype)
print(src)

src = src.astype(np.uint8)  # 0 ~ 255 사이의 값
print(src.dtype)
print(src)

