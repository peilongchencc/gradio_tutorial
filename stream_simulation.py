"""description:流式输出模拟。
"""
import time

data_source = ["数据1", "数据2", "数据3", "数据4", "数据5"]

def data_stream_generator(data):
    result = ""
    # 模拟流式输出数据
    for data in range(100):
        time.sleep(0.5)
        result += str(data)
        yield result

# 使用生成器函数
for current_result in data_stream_generator(data_source):
    print(current_result)