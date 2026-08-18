import serial  # 调用串口通信库
import time    

# 寄存器地址说明，对应仿人五指灵巧手——RH56F1用户手册11页，2.4寄存器说明
regdict = {
    'ID'         : 1000,  # ID
    'baudrate'   : 1001,  # 波特率设置
    'mode'       : 1100,  # 手指控制模式（速度力保护，力闭环，阻抗，0-1-2）
    'clearErr'   : 1003,  # 清除错误
    'forceClb'   : 1007,  # 力传感器校准
    'angleSet'   : 1040,  # 各自由度的角度设置值
    'forceSet'   : 1046,  # 各自由度的力控阈值设置值
    'speedSet'   : 1052,  # 各自由度的速度设置值
    'angleAct'   : 1064,  # 各自由度的角度实际值
    'forceAct'   : 1070,  # 各手指的实际受力
    'errCode'    : 1082,  # 各自由度的电缸故障信息
    'statusCode' : 1088,  # 各自由度的状态信息
    'temp'       : 1094,  # 各自由度的电缸的温度
    'ip'         : 1700,  #ip
    'actionSeq'  : 2160,  # 当前动作序列索引号
    'actionRun'  : 2162   # 运行当前动作序列
}

# 函数说明：设置串口号和波特率并且打开串口；参数：port为串口号，baudrate为波特率
def openSerial(port, baudrate):
    ser = serial.Serial() # 调用串口通信函数
    ser.port = port
    ser.baudrate = baudrate
    ser.open()            # 打开串口
    return ser

# 函数说明：写灵巧手寄存器操作函数；参数：id为灵巧手ID号，add为起始地址，num为该帧数据的部分长度，val为所要写入寄存器的数据
def writeRegister(ser, id, add, num, val):
    bytes = [0xEB, 0x90]            # 帧头
    bytes.append(id)                # id
    bytes.append(num + 3)           # len
    bytes.append(0x12)              # cmd 写寄存器命令标志
    bytes.append(add & 0xFF)        # 寄存器起始地址低八位
    bytes.append((add >> 8) & 0xFF) # 寄存器起始地址高八位
    for i in range(num):
        bytes.append(val[i])
    checksum = 0x00                 # 校验和初始化为0
    for i in range(2, len(bytes)):
        checksum += bytes[i]        # 对数据进行加和处理
    checksum &= 0xFF                # 对校验和取低八位
    bytes.append(checksum)          # 低八位校验和
    
    print("发送到串口的指令:", [hex(b) for b in bytes])
    
    ser.write(bytes)                # 向串口写入数据
    time.sleep(0.025)                # 延时25ms
    ser.read_all()                  # 把返回帧读掉，不处理

# 函数说明：读灵巧手寄存器操作；参数：id为灵巧手ID号，add为起始地址，num为该帧数据的部分长度，mute为调试标志位
def readRegister(ser, id, add, num, mute=False):
    bytes = [0xEB, 0x90]            # 帧头
    bytes.append(id)                # id
    bytes.append(0x04)              # len 该帧数据长度
    bytes.append(0x11)              # cmd 读寄存器命令标志
    bytes.append(add & 0xFF)        # 寄存器起始地址低八位
    bytes.append((add >> 8) & 0xFF) # 寄存器起始地址高八位
    bytes.append(num)               # num 寄存器数量
    checksum = 0x00                 # 校验和赋值为0
    for i in range(2, len(bytes)):
        checksum += bytes[i]        # 对数据进行加和处理
    checksum &= 0xFF                # 对校验和取低八位
    bytes.append(checksum)          # 低八位校验和
    
    print("发送到串口的指令:", [hex(b) for b in bytes])
    
    ser.write(bytes)                # 向串口写入数据
    time.sleep(0.025)                # 延时25ms
    recv = ser.read_all()           # 从端口读字节数据
    print(recv)
    if len(recv) == 0:              # 如果返回的数据长度为0，直接返回
        return []
    num = (recv[3] & 0xFF) - 3      # 寄存器数据所返回的数量
    val = []
    for i in range(num):
        value = (recv[7 + i])
        if value > 32767:
            value -= 65536
        val.append(value)
    if not mute:
        print('读到的寄存器值依次为：', end='')
        for i in range(num):
            print(val[i], end=' ')
        print()
    return val

# 函数功能：写入灵巧手六个电缸数据函数，angleSet设置灵巧手运动角度参数、forceSet设置灵巧手抓握力度参数、speedSet设置灵巧手运动速度参数
# 参数说明：ID为灵巧手对应ID号，str为灵巧手选取设置的参数，val为设置数据
def write6(ser, id, str, val):
    if str == 'angleSet' or str == 'forceSet' or str == 'speedSet' or str == 'mode':
        val_reg = []
        for i in range(6):
            val_reg.append(val[i] & 0xFF)
            val_reg.append((val[i] >> 8) & 0xFF)
        writeRegister(ser, id, regdict[str], 12, val_reg)
    else:
        print('函数调用错误，正确方式：str的值为\'angleSet\'/\'forceSet\'/\'speedSet\'，val为长度为6的list，值为0~1000，允许使用-1作为占位符')

# 函数功能：读取灵巧手数据
# angleSet为灵巧手运动角度参数、forceSet为灵巧手抓握力度参数、speedSet为灵巧手运动速度参数、angleAct为灵巧手角度实际值、forceAct为灵巧手各手指的实际受力值
def read6(ser, id, str):
    if str == 'angleSet' or str == 'forceSet' or str == 'speedSet' or str == 'angleAct' or str == 'forceAct' or str == 'temp'  or str == 'errCode'  or str == 'ip' or str == 'statusCode' :
        val = readRegister(ser, id, regdict[str], 12, True) # 读取
        if len(val) < 12:         # 读取到的数据小于12直接舍弃
            print('没有读到数据')
            return
        val_act = []
        for i in range(6):
            value_act = ((val[2*i] & 0xFF) + (val[1 + 2*i] << 8))
            if value_act > 32767:
                value_act -= 65535
            val_act.append(value_act)
        print('读到的值依次为：', end='')
        for i in range(6):
            print(val_act[i], end=' ')
        print()
        return val_act
    else:
        print('函数调用错误，正确方式：str的值为\'angleSet\'/\'forceSet\'/\'speedSet\'/\'angleAct\'/\'forceAct\'/\'errCode\'/\'statusCode\'/\'ip\'')

def readTouchData(ser):#（法向力、切向力、切向力方向（顺时针360度，无接触时返回65535）、接近觉（0-16000000））
    # 发送指令
    cmd = bytes([0xEB, 0x90, 0x01, 0x04, 0x11, 0xB8, 0x0B, 0x44, 0x1D])
    print("发送触觉读取指令:", [hex(b) for b in cmd])
    ser.write(cmd)
    time.sleep(0.025) #延时25ms接收数据
    recv = ser.read_all()
    
    # 打印原始响应
    print("原始响应数据:", recv)
    
    # 定位触觉数据起始位置（0xB8 0x0B）
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            start_idx = recv.index(0xB8)
            if recv[start_idx + 1] == 0x0B:
                data_start = start_idx + 2
                break  # 找到数据，退出循环
            else:
                print(f"定位尝试 {attempt + 1}: 未找到预期的B8 0B")
        except ValueError:
            print(f"定位尝试 {attempt + 1}: 未找到触觉数据地址 B8 0B")

        time.sleep(0.01)  # 延迟后重试

    else:  # 如果在所有尝试中都未找到，返回 None
        print("未找到触觉数据地址 B8 0B!!")
        return None, None

    # 五个手指名称
    fingers = ['little', 'ring', 'middle', 'index', 'thumb']
    finger_results = {}

    for i, finger in enumerate(fingers):
        base_idx = data_start + i * 10  # 每组10个字节

        # 读取10个字节
        bytes_data = recv[base_idx:base_idx + 10]  

        # 组合手指数据
        data_bytes = [
            (bytes_data[j] | (bytes_data[j + 1] << 8)) for j in range(0, 6, 2)  # 低字节在前
        ]
    
        # 24位接近觉数据
        combined_value = (bytes_data[6] | (bytes_data[7] << 8) | (bytes_data[8] << 16))
        data_bytes.append(combined_value)

        print(f"{finger} 数据:", data_bytes)

        finger_results[finger] = data_bytes

    # 提取掌心部分的18个字节，组成9个数据
    plam_results = {}
    plam_start_idx = data_start + len(fingers) * 10  

    if plam_start_idx + 17 < len(recv):  # 确保有足够的字节
        plam_data = []
        for j in range(18):
            plam_byte = recv[plam_start_idx + j]
            plam_data.append(plam_byte)

        # 将18个字节组合成9个数据，每2个字节组合成1个数据
        for j in range(9):
            b0 = plam_data[j * 2]      # 低字节
            b1 = plam_data[j * 2 + 1]  # 高字节
            plam_value = (b0 | (b1 << 8))  # 组合为一个16位整数
            plam_results[f'plam_data_{j + 1}'] = plam_value  # plam_data_1-3(掌左)，plam_data_4-6(掌中)，plam_data_7-9(掌右)。

    else:
        print("掌心数据超出边界，无法读取")

    # 返回数据字典，包括五指和掌心
    ser.read_all() #清除残余数据
    return finger_results, plam_results

def main():
    # 初始化串口（请根据实际情况修改端口和波特率）
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1)
    
    print('设置灵巧手控制模式，-1为不设置！')
    write6(ser, 1, 'mode', [0, 0, 0, 0, 0, 0]) # ID号改为对应灵巧手的ID，0-速度力保护，1-力闭环，2-阻抗
    time.sleep(0.1)                   # 延时0.1s
        
    print('设置灵巧手运动速度参数，-1为不设置该运动速度！')
    write6(ser, 1, 'speedSet', [4000, 4000, 4000, 4000, 4000, 4000]) # ID号改为对应灵巧手的ID，val对应的电缸ID为1,2,3,4,5,6;对应的速度值为0-4000，4000为最大值，0不运动，如果val设置为-1，相应的手指无反应
    time.sleep(0.1)                   # 延时0.1s
    print('设置灵巧手抓握力度参数！')
    write6(ser, 1, 'forceSet', [1500, 1500, 1500, 1500, 2000, 2000])# ID号改为对应灵巧手的ID，val对应的电缸ID为1,2,3,4,5,6;四指对应的力度值为0-1500，1500为最大力，0不运动，如果val设置为-1，相应的手指无反应，大拇指两个自由度对应的力度值为0-2000，2000为最大力，0不运动，如果val设置为-1，相应的手指无反应
    time.sleep(0.1)                   # 延时0.1s
    print('设置灵巧手运动角度参数0，-1为不设置该运动角度！')
    #write6(ser, 1, 'angleSet', [900, 900, 1740, 1740, 900, 1750])# ID号改为对应灵巧手的ID，val对应的电缸ID为1,2,3,4,5,6;对应的角度值四指为900-1740，1740为最大角度（174度），900（90度）为最小角度，大拇指弯曲为1100-1350，1350为最大角度（135度），1100（110度）为最小角度，大拇指侧摆为600-1750，1750为最大角度（175度），600（60度）为最小角度,如果设置为-1，相应的手指无反应
    #time.sleep(1)             # 延时1s
    #write6(ser, 1, 'angleSet', [900, 900, 900, 900, 1350, 1750])# ID号改为对应灵巧手的ID，val对应的电缸ID为1,2,3,4,5,6;对应的角度值四指为900-1740，1740为最大角度（174度），900（90度）为最小角度，大拇指弯曲为1100-1350，1350为最大角度（135度），1100（110度）为最小角度，大拇指侧摆为600-1750，1750为最大角度（175度），600（60度）为最小角度,如果设置为-1，相应的手指无反应
    #time.sleep(1)                   # 延时1s
    
    try:
        while True:
            finger_data, plam_data = readTouchData(ser)
            if finger_data is not None and plam_data is not None:
                # 打印触觉数据（法向力、切向力、切向力方向（顺时针360度，无接触时返回65535）、接近觉（0-16000000））
                print("手指数据:", finger_data)
                print("掌心数据:", plam_data)
            else:
                print("数据读取失败！")

            time.sleep(0.02)  # 50Hz，间隔20毫秒

    except KeyboardInterrupt:
        print("程序退出")
    finally:
        ser.close()  # 关闭串口

if __name__ == "__main__":
    main()
