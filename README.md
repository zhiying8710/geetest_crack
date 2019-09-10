[测试接口](https://github.com/zhiying8710/geetest_crack/blob/master/%E6%B5%8B%E8%AF%95%E6%8E%A5%E5%8F%A3.md)`每日限制50次, 仅供学习交流`

[使用极验的网站](https://github.com/zhiying8710/geetest_crack/issues/5)


[极验参数抓包说明](https://github.com/zhiying8710/geetest_crack/issues/6)

# geetest极验二代滑动、三代滑动和汉字点选破解

## 免责声明

本仓库仅用于学术交流, 不得用于任何商业用途!!!

## 说明

为避免不必要的纠纷, 本仓库不提供相关模型源代码、模型文件、数据集、服务源代码等.

可提issue或发邮件至zhiying8710@hotmail.com进行交流

## 通过率

- 滑动: 通过算法生成轨迹
  - 二代: 重试一次通过率99%
  - 三代: 不重试, 通过率99%
- 汉字点选: 99%

## 耗时
- 滑动: 4s以内
- 汉字点选: 10s以内(CPU上YOLO3比较耗时)

## 结果样例

二代滑动:

![gee_2](https://github.com/zhiying8710/geetest_crack/raw/master/imgs/gee_2.gif)

三代点击:

![gee_3_1](https://github.com/zhiying8710/geetest_crack/raw/master/imgs/gee_3_1.gif)

三代滑动:

![gee_3_2](https://github.com/zhiying8710/geetest_crack/raw/master/imgs/gee_3_2.gif)

三代按文字点选:

![gee](https://github.com/zhiying8710/geetest_crack/raw/master/imgs/gee_3_3.gif)

三代按语序点选:

![gee](https://github.com/zhiying8710/geetest_crack/raw/master/imgs/gee_3_4.gif)



## 开发环境

- python3.6
- tensorflow
- keras
- darknet(YOLO3)
- labelImg(YOLO3数据标签工具)
- opencv(定位滑动缺口距离)
- pyppeteer

## 算法

- YOLO3: 定位汉字位置
- CRNN: 校验文字识别
- CNN: 定位后的文字识别

## 数据集

### 汉字点选

- 4000+汉字
- 校验文字样本约50K
- YOLO3样本3.6K
- 定位后的文字样本140K



