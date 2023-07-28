# python basic
- 目录：  
    - [print](#print)  
    - [input](#input)

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
## input
###