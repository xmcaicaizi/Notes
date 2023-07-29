# python basic
- 目录：  
    - [print](#print)  
    - [list and tuple](#list-and-tuple)
    - [set](#set函数)
    - [try](#try)
## print
### 1、print的函数语法
```
print(*objects, sep=' ', end='\n', file=sys.stdout)  
```  
objects – 复数，表示可以一次输出多个对象。输出多个对象时，需要用 , 分隔。  
sep – 用来间隔多个对象，默认值是一个空格。  
end – 用来设定以什么结尾。默认值是换行符 \n，我们可以换成其他字符串。  
file – 要写入的文件对象。  
### 2、格式化输出
```
print('The length of %s is %d' % (s,x))
```
与c语言不同的是，用%分隔，%后变量按顺序填入占位符。
### 3、format()函数
通过位置来填充字符
```
- print('hello {0} i am {1}'.format('world','python'))    
  输出结果：hello world i am python

- print('hello {} i am {}'.format('world','python') ) 
  输出结果：hello world i am python

- print('hello {0} i am {1} . a now language-- {1}'.format('world','python')
  输出结果：hello world i am python . a now language-- python
```  
foramt 把参数按位置顺序填充到字符串中，第一个参数是0，然后1...
也可以不输入数字，这样也会按顺序来填充。  
同一个参数可以填充多次，这个是 format 比 % 先进的地方

## list
### List
#### 索引
可通过下标索引```list[0]```或```list[0:4]```
#### slice(切片)操作
可以自己赋值```indexes = slice(0, 5, 2)```也可以使用索引```indexes = slice[0:5]``  
#### operations
1. append方法  
插入元素本身到list末尾```append(value)```

2. extend方法  
插入元素值到list末尾```extend(value)```

3. insert方法  
按索引插入元素```insert(index, value)```

4. del  
按索引删除元素```del[-2]```删除倒数第二个

5. remove  
按给定值删除元素```remove('red')```删除元素red

6. pop  
默认弹出list中最后一个元素```value = list.pop()```

7. clear  
清空list

8. index方法  
取对应元素索引```index = list.index(value)```

9. count方法  
统计对应元素个数```count = list.count(value)```

10. len  
计算对应变量长度```lenth = len(value)```

11. max函数  
取变量中最大值```max = max(value)```

12. min函数  
取变量中最最值```min = min(value)```

13. sort  
按增序对变量进行排序
```
  nums = [4, 3, 5, 2, 1, 6]
  nums.sort()
  nums = [1, 2, 3, 4, 5, 6]
```

14. reverse  
按照索引逆序排序 ```reverse(nums)  nums = [6, 5, 4, 3, 2, 1]```

1. copy   
从对应变量拷贝 ```another = value.copy()```

1. 加法运算  
直接按顺序相加 ```this + another = thisanother```

1. 乘法运算  
```print(another * 2)```打印another两次

1. 查询操作  
查询对应元素是否在list里  
```print('2' in nums, '2' not in nums)  return Ture or False```

## set函数
创建一个集合
### 符号操作
```
  print(A | B)  # 并集
  print(A - B)  # 差集
  print(A & B)  # 交集
  print(A ^ B)  # 补集
```
### 方法
1. dfference  
返回集合中的差集
```value = A.difference(B)```

2. intersection
返回集合中的交集
```value = A.intersection(B)```

3. union  
返回集合中的并集
```value = A.union(B)```

4. symmetric_difference  
返回两个集合中不重合的元素,会移除都存在的元素
```
  A = {1, 2, 3, 4, 5}
  B = {4, 5, 6, 7, 8}
  value = A.symmetric_difference(B)
  value = {1, 2, 3, 6, 7, 8}
```
5. issubset  
判断B是否是A子集
```A.issubset(B)```

6. issuperset
判断A是否是B的子集
```A.issuperset(B)```

7.  isdisjoint  
判断两个集合是否含有相同的元素，没有返回Ture，反之返回False
```A.isdisjoint(B)```

## try
```
try:
    语句块1
except 异常类型:
    语句块2
```