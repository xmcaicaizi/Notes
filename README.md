# Notes
关于机器学习的学习笔记
## 1、梯度下降
###  梯度下降的主要原理  
#### 1.确定一个小目标--预测函数  
        ```
        学习算法  
        发现规律  
        改进模型  
        做出预测
        ```  

#### 2.找到差距--代价函数  
```
例如均方误差:  
e = (y - w * x)²  
展开后：  
e = x² * w² - 2(x * y) * w + y²  

多个方差  
e1 = x1² * w1² - 2(x1 * y1) * w1 + y1²  
e2 = x2² * w2² - 2(x2 * y2) * w2 + y2²  
e3 = x3² * w3² - 2(x3 * y3) * w3 + y3²  
...  
en = xn² * wn² - 2(xn * yn) * wn + yn²  
合并同类项：
e = 1/n((x1²+...+xn²)*w² + (-2*x1*y1-...-2*xn*yn)*w + (y1²+...+yn²))  
由于x、y为已知数，用常量a、b、c分别代替不同项系数  
e = 1/n(a*w² + b*w + c)  
```

#### 3.明确搜索方向--梯度计算  
向陡峭程度最大的方向走  
对e求导  
#### 4.大胆的往前走吗？--学习率  
`新w = 旧w - 导数 * 学习率`

## todo  
### July 27 2023  
——————————————————————————  
Short term goal:        
 学习pyFast              
- [ ] 1.学习pyFast          
- [ ] 2.跑实验,记录实验结果    
- [ ] 3.做好笔记               
——————————————————————————  
    