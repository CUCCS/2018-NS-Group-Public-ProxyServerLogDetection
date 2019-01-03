## SQL injection 检测
### 前言
SQL注入是一种代码注入，是发生于应用程序与数据库层的安全漏洞。恶意SQL语句实现SQL注入攻击包括通过从客户端到应用程序的输入数据插入或"注入"SQL查询。一个成功的SQL注入攻击可实现从数据库中读取敏感数据，修改数据库的数据,对数据库执行管理操作(如关闭DBMS)，恢复DBMS文件系统上存在的给定文件的内容，并在某些情况下向操作系统发出命令。

### 实现
主要还是根据[data_hacking](https://github.com/c4pr1c3/data_hacking)进行实验，尝试将其中的SQL注入检测算法```SQL Parse```更换成![libinjection](https://github.com/client9/libinjection)但是实验结果表明这个替换完全失败了，之后会给出比较结果

总体步骤：
- 对训练集进行处理
- 特征提取
- 随机森林模型训练
- 测试模型
- 交叉验证

#### 训练集处理
本次实验训练集实际在进行训练之前就已有```legit```和```malicious```分类，即，不需要在实验过程中通过相关算法对数据进行分类，所以处理过程如下
- 将```legit```数据和```malicious```数据进行整合
- 对数据缺失值进行处理
- 对数据进行去重和混洗

#### 特征提取
因本次实验大部分时间都用于思考为何```injection```注入检测算法无法达到预期效果，因此没有过多增加特征，特征主要如下
- SQL语句长度
- 香农熵
- 使用```G-test```计算出的```malicious_g```和```legit_g```
    这里计算的是分别由```SQL Parse```和```injection```SQL注入检测算法生成的token集和sqli及sql之间的相关性

#### 随机森林模型测试
使用sklearn中的函数
- 分割训练集和测试集
- 选取特征进行训练

#### 测试模型
在网上只找到了```sqli```数据集，所以自己手动生成了1000个```sql```数据的数据集加上原来训练使用的legit训练集进行测试

#### 交叉认证
主要针对的是```injection```SQL注入检测算法进行特征组合认证

### 实验结果
- ```injection```出来的结果很令人绝望
    - 将合法SQL语句判成非法的误判率很高，输入非法语句测试集进行测试，```SQL Parse```和```libinjection```训练模型测试正确率都达到99%以上，但是使用合法语句测试集，```SQL Parse```训练模型正确率为```0.99404489```,```injection```训练模型正确率只有```0.0132844```
    - 对```injection```训练模型进行交叉验证，结果发现```G-test```计算出来的特征完全没有用。具体结果在```jupyter notebook```中
    - 实验过程中还尝试过将一条SQL语句的token做为一个整体不分割的放入```G-test```进行计算，测试效果比分割计算好，但思考一下，这样对数据集数量要求很高
    - 使用训练集中的数据直接放到```injection```对SQL语句合法性判断的函数中，发现误判率高
由于对机器学习相关算法的认知浅薄，无法对实验结果进行理论上的分析，因此目前还没有解决为何```injection```训练模型测试出的结果与预想中的结果相差很大的问题
### 参考文献
#### 概念相关
- [SQL注入-wiki](https://zh.wikipedia.org/zh-cn/SQL%E8%B3%87%E6%96%99%E9%9A%B1%E7%A2%BC%E6%94%BB%E6%93%8A)
- [G–test of goodness-of-fit](http://www.biostathandbook.com/gtestgof.html)
- [libinjection-blackhatusa2012](#https://www.slideshare.net/nickgsuperstar/libinjection-blackhatusa2012)
#### 实现相关
- [sql parse](https://docs.python.org/3/reference/expressions.html#displays-for-lists-sets-and-dictionaries)
- [client9/libinjection](https://github.com/client9/libinjection/wiki/doc-sqli-python)
- [c4pr1c3/data_hacking](https://github.com/c4pr1c3/data_hacking)

#### 数据集来源
- [c4pr1c3/data_hacking](https://github.com/c4pr1c3/data_hacking/tree/master/sql_injection)
- [libinjection](https://github.com/client9/libinjection/tree/master/data)