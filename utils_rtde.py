
import time
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface

robot_ip = "192.168.134.100"
rtde_r = RTDEReceiveInterface(robot_ip)

while True:
    q = rtde_r.getActualQ()
    print(q)
    time.sleep(0.008)  # 125Hz
