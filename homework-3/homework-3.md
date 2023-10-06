<h1>
<center>
Data Science Homework-3
    </center>
</h1>

<h4>
    <center>
    	朱维清 10215300402
    </center>

---

#### 1.十进制到二进制小数转换：

<img src="image/homework-3/image-20231006190648028.png" alt="image-20231006190648028" style="zoom:67%;" />

<img src="image/homework-3/image-20231006190708642.png" alt="image-20231006190708642" style="zoom: 80%;" />

​		由于直接对输入小数乘2取余，若该十进制小数无法转化为有限二进制小数，会在多次迭代后出现误差越来越大的情况，故采用字符串的方式处理，确定十进制小数最初长度（即小数位数）length，保证后续迭代不出现误差。

#### 2.产生10-20之间随机浮点数：

<img src="image/homework-3/image-20231006191141746.png" alt="image-20231006191141746" style="zoom:50%;" />

#### 3.正则表达式简单验证身份证号是否合法：

<img src="image/homework-3/image-20231006191757446.png" alt="image-20231006191757446" style="zoom:67%;" />

​		简单匹配了18位身份证号，并未按照各位权重计算校验码是否正确。

#### 4.实现单向链表：

<img src="image/homework-3/image-20231006191929518.png" alt="image-20231006191929518" style="zoom: 67%;" />

#### 5.6.7输出结果：

<img src="image/homework-3/image-20231006192103138.png" alt="image-20231006192103138" style="zoom:67%;" />

<img src="image/homework-3/image-20231006192153972.png" alt="image-20231006192153972" style="zoom:67%;" />

<img src="image/homework-3/image-20231006192209230.png" alt="image-20231006192209230" style="zoom:67%;" />

#### 8.插入、快速、希尔、选择、归并排序在不同长度数列下的运行效果：

<img src="image/homework-3/image-20231006192307164.png" alt="image-20231006192307164" style="zoom:67%;" />

​		插入排序和选择排序时间复杂度相同，在数组较大时耗时最多；归并排序耗时第二多；希尔排序耗时第三；快速排序最快。

#### 9.构建乘积数组：

<img src="image/homework-3/image-20231006192600839.png" alt="image-20231006192600839" style="zoom:67%;" />

<img src="image/homework-3/image-20231006192620317.png" alt="image-20231006192620317" style="zoom:67%;" />

​		对于B的每个元素全部叠乘的时间复杂度太高。这一算法采用两次循环和中间值temp用来与下标i同时变化，第一次循环保证B[i]当前值的来源因子不包括A[i]且只计算了A[i]的左侧乘积；第二次循环补上A[i]的右侧乘积。

#### 10、11输出结果：

<img src="image/homework-3/image-20231006193019132.png" alt="image-20231006193019132" style="zoom:67%;" />

<img src="image/homework-3/image-20231006193100733.png" alt="image-20231006193100733" style="zoom:67%;" />

​		计算程序耗时需要import time，运用库函数time.time()。

#### 12.

<img src="image/homework-3/20190924113036555.png" alt="img" style="zoom: 50%;" />

#### 13.选择排序时间复杂度和空间复杂度：

​		时间复杂度：O(n^2)

​		空间复杂度：O(1)