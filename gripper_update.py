from pymycobot import MyCobot
import time
"""
将com4改成你的机械臂串口
"""
mc = MyCobot("COM32")

# 修改
mc.set_servo_data(7,16,150)
time.sleep(0.5)
mc.set_servo_data(7,34,100)
time.sleep(0.5)
mc.set_servo_data(7,35,200)
time.sleep(0.5)

# 验证

a= mc.get_servo_data(7,16)
time.sleep(0.5)
# 显示 150 就表示成功
print(a)
b = mc.get_servo_data(7,35)
time.sleep(0.5)
# 显示200 
print(b)

c = mc.get_servo_data(7,34)
time.sleep(0.5)
# 显示100
print(c)
