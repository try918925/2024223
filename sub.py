import zmq

# ZeroMQ 上下文
context = zmq.Context()

# 创建订阅者Socket，并连接到指定地址和端口
subscriber = context.socket(zmq.SUB)
subscriber.connect("tcp://localhost:5555")  # 连接到本地发布者，端口5555

# 设置订阅过滤器（空字符串表示接收所有消息）
subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

while True:
    # 接收消息
    message = subscriber.recv_string()

    # 处理接收到的消息
    print("Received:", message)
