# cookbook
- 目录：  
    - [c01](#c01)  
    - [list and tuple](#list-and-tuple)
    - [set](#set函数)
    - [try](#try)

## c01
1. 将序列(list或tuple)解包为单独的序列  
```value 1, value 2, value 3 = [value 1, value 2, value 3]```

2. 在list中找最大或最小的几个数  
```
nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]
print(heapq.nlargest(3, nums))  # [42, 37, 23]
print(heapq.nsmallest(3, nums))  # [-4, 1, 2]
```
nlargest(number, value)，在value中找前number个最大  
nsmallest(number, value)，在value中找后number个最小

还有一种用法是 ```another = heapq.nlargest(numbers, value, key)```  
key可以是lambda创建的匿名函数例如：
key = lambda s: s['price'],输入对象是s，输出为list **s** 中的price元素

3. PriorityQueue
```
q = PriorityQueue()
    q.push(Item('foo'), 1)
    q.push(Item('bar'), 5)
    q.push(Item('spam'), 4)
    q.push(Item('grok'), 1)
    print(q.pop())  # Item('bar')
    print(q.pop())  # Item('spam')
    print(q.pop())  # Item('foo')
    print(q.pop())  # Item('grok')
```
```q.push(value, level)```数字越小优先级越高

4. 多重字典  
    1. 创建时```value = defaultdict(tpye)```创建类型为type的多重字典value  
    1. 按照关键字填入 ```value['key'].operation```对应type的方法键入
    1. 可以使用for方法迭代
    ```
    for key in d:
        print(key, d[key])  # "foo 1", "bar 2", "spam 3", "grok 4"
    ```
    4. zip函数  
    可以将对象打包为元组后用list返回