# geetest极验二代滑动、三代滑动和汉字点选破解

## 免责声明

该项目仅用于学术交流, 不得用于任何商业用途!!!

## 通过率

- 滑动: 通过算法生成轨迹
  - 二代: 重试一次通过率99%
  - 三代: 不重试, 通过率99%
- 汉字点选: 99%

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
- selenium

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



