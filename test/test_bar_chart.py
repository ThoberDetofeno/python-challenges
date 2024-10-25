import unittest
from code_challenge.bar_chart import bar_chart

class test_bar_chart(unittest.TestCase):

    def test_values(self):
        self.assertEqual(bar_chart({'Q4': 0, 'Q3': 100, 'Q2': 0, 'Q1': 600}), "Q1|############ 600\nQ3|## 100\nQ2|0\nQ4|0")
        self.assertEqual(bar_chart({'Q4': 300, 'Q3': 150, 'Q2': 350, 'Q1': 250}), "Q2|####### 350\nQ4|###### 300\nQ1|##### 250\nQ3|### 150")
        self.assertEqual(bar_chart({'Q4': 350, 'Q3': 400, 'Q2': 400, 'Q1': 50}), "Q2|######## 400\nQ3|######## 400\nQ4|####### 350\nQ1|# 50")
        self.assertEqual(bar_chart({'Q4': 200, 'Q1': 500, 'Q2': 300, 'Q3': 300}), "Q1|########## 500\nQ2|###### 300\nQ3|###### 300\nQ4|#### 200")
        self.assertEqual(bar_chart({'Q4': 300, 'Q3': 250, 'Q2': 600, 'Q1': 350}), "Q2|############ 600\nQ1|####### 350\nQ4|###### 300\nQ3|##### 250")
        self.assertEqual(bar_chart({'Q4': 150, 'Q1': 550, 'Q2': 50, 'Q3': 600}), "Q3|############ 600\nQ1|########### 550\nQ4|### 150\nQ2|# 50")
        self.assertEqual(bar_chart({'Q4': 450, 'Q3': 0, 'Q2': 50, 'Q1': 200}), "Q4|######### 450\nQ1|#### 200\nQ2|# 50\nQ3|0")
        self.assertEqual(bar_chart({'Q4': 150, 'Q3': 0, 'Q2': 0, 'Q1': 450}), "Q1|######### 450\nQ4|### 150\nQ2|0\nQ3|0")
        self.assertEqual(bar_chart({'Q4': 0, 'Q1': 600, 'Q2': 250, 'Q3': 400}), "Q1|############ 600\nQ3|######## 400\nQ2|##### 250\nQ4|0")
        self.assertEqual(bar_chart({'Q4': 100, 'Q1': 150, 'Q2': 450, 'Q3': 0}), "Q2|######### 450\nQ1|### 150\nQ4|## 100\nQ3|0")
        self.assertEqual(bar_chart({'Q4': 150, 'Q1': 400, 'Q2': 100, 'Q3': 0}), "Q1|######## 400\nQ4|### 150\nQ2|## 100\nQ3|0")
        self.assertEqual(bar_chart({'Q4': 550, 'Q1': 600, 'Q2': 200, 'Q3': 50}), "Q1|############ 600\nQ4|########### 550\nQ2|#### 200\nQ3|# 50")
        self.assertEqual(bar_chart({'Q4': 250, 'Q3': 200, 'Q2': 500, 'Q1': 550}), "Q1|########### 550\nQ2|########## 500\nQ4|##### 250\nQ3|#### 200")
        self.assertEqual(bar_chart({'Q4': 450, 'Q3': 50, 'Q2': 500, 'Q1': 0}), "Q2|########## 500\nQ4|######### 450\nQ3|# 50\nQ1|0")
        self.assertEqual(bar_chart({'Q4': 250, 'Q3': 400, 'Q2': 150, 'Q1': 500}), "Q1|########## 500\nQ3|######## 400\nQ4|##### 250\nQ2|### 150")
        self.assertEqual(bar_chart({'Q4': 400, 'Q3': 600, 'Q2': 350, 'Q1': 600}), "Q1|############ 600\nQ3|############ 600\nQ4|######## 400\nQ2|####### 350")
        self.assertEqual(bar_chart({'Q4': 50, 'Q1': 100, 'Q2': 150, 'Q3': 50}), "Q2|### 150\nQ1|## 100\nQ3|# 50\nQ4|# 50")
        self.assertEqual(bar_chart({'Q4': 50, 'Q1': 100, 'Q2': 100, 'Q3': 300}), "Q3|###### 300\nQ1|## 100\nQ2|## 100\nQ4|# 50")
        self.assertEqual(bar_chart({'Q4': 350, 'Q3': 50, 'Q2': 600, 'Q1': 300}), "Q2|############ 600\nQ4|####### 350\nQ1|###### 300\nQ3|# 50")
        self.assertEqual(bar_chart({'Q4': 100, 'Q1': 500, 'Q2': 50, 'Q3': 200}), "Q1|########## 500\nQ3|#### 200\nQ4|## 100\nQ2|# 50")

if __name__ == "__main__":
    unittest.main()