
# coding: utf-8

# # SimpleBGC controller
# 
# Implemented using the serial API specified in www.basecamelectronics.com/serialapi/

# In[2]:


import numpy as np, serial, time


# In[356]:


ser = serial.Serial('/dev/serial0',  # has been mapped to ttyAMA0
                    baudrate=115200,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,  # serial.PARITY_NONE if this doesn't work
                    stopbits=serial.STOPBITS_ONE,
                    timeout=5)
#serial0 == '/dev/ttyS0', connected on physical pins 8 & 10.
#serial1 == '/dev/ttyAMA0', connected internally to the Bluetooth hardware.
# parity issues on ttyS0:
#     https://github.com/pyserial/pyserial/issues/196#issuecomment-323382976


# In[304]:


L = {
    # menu items - so far, these all work
    'CMD_EXECUTE_MENU': 69,
    'SBGC_MENU_MOTOR_ON': 11,
    'SBGC_MENU_MOTOR_OFF': 12,
    
    # cmd items - so far, only 'control' works
    'CMD_CONTROL': 67,  # works
    'CMD_REALTIME_DATA': 68,
    'CMD_GET_ANGLES': 73,  #??
    'CMD_MOTORS_ON': 77,
    'CMD_MOTORS_OFF': 109
}


# In[75]:


def as_int(b):
    """ converts b (a bytearray) into a signed integer.
        assumes little endian """
    return int.from_bytes(b, byteorder='little', signed=True)

def as_uint(b):
    """ converts b (a bytearray) into an unsigned integer.
        assumes little endian """
    return int.from_bytes(b, byteorder='little', signed=False)

def print_binary_padded(hex_data, num_bits=8, reversed=False):
    """ prints the hex value in binary, with zero padding
    
        eg: print_binary_padded(0x03)
            print_binary_padded(0x0421, num_bits=16)
    """
    if reversed is True:
        print('LSB first: 0b' + bin(hex_data)[2:].zfill(num_bits)[::-1])
    else:
        print('0b' + bin(hex_data)[2:].zfill(num_bits))  # [2:] to get rid of the '0b' part

def print_hex_nicely(msg):
    """ hex string, printed in groups of bytes"""
    [print(msg[i*2:i*2+2] + ' ', end='') for i in range(len(msg)//2)]
    print()


# In[305]:


def send_data(command_ID, data=None, data_size=0):
    """ Compose and send a command to the SimpleBGC
    Inputs:
        command_ID: a string, such as 'CMD_GET_ANGLES'
        data: a list of numpy numbers
        data_size: the number of bytes in data

    Format:
        head:
            start_char = ord('>') = 0x3E. 1u
            command_ID. 1u
            data_size. 1u. Can be 0
            header_checksum =  (command ID + data_size) % 255. 1u

        body:
            array_of_bytes_of_length_data_size
            body_checksum. 1u
    """
    # compose head:
    start_char = np.uint8(ord('>'))
#    header_checksum = np.uint8((L[command_ID] + data_size) % 0xFF)
    header_checksum = np.uint8(L[command_ID] + data_size)
    
    message = bytearray()
    message.append(start_char)
    message.append(L[command_ID])
    message.append(np.uint8(data_size))
    message.append(header_checksum)

    if data_size > 0:
        for d in data:
            if d.nbytes == 1:
                message.append(d)
            elif d.nbytes == 2:
                d_bytes = d.tobytes()
                # print('%s\t-> %s'% (d, d_bytes))
                message.append(d_bytes[0])  # working with little endian
                message.append(d_bytes[1])

        body_checksum = np.uint8(sum(data) % 0xFF)
        print([x for x in data])
        print('body_checksum = %i' % body_checksum)
        
        # have to subtract 1 for each negative value. Not sure why!
        for d in data:
            if d < 0:
                body_checksum -= 1
                break
        
        print('body_checksum after subtracting = %i' % body_checksum)
        message.append(np.uint8(body_checksum))
    
    ser.write(message)
    print('message sent: ' + message.hex())
    
    return message


# In[306]:


def send_angle_command(roll, pitch, yaw):
    """ send an angle command to the gimbal
        roll, pitch and yaw are in degrees """

    scaling = 0.02197265625  # units/degree
    message = [
        np.uint8(2),                # CONTROL_MODE = MODE_ANGLE. do this three times for three axis??
        np.int16(0),                # roll speed
        np.int16(roll/scaling),     # roll angle
        np.int16(0),                # pitch speed
        np.int16(pitch/scaling),    # pitch angle
        np.int16(0),                # yaw speed
        np.int16(yaw/scaling)       # yaw angle
    ]
    return send_data('CMD_CONTROL', data=message, data_size=13)


# In[352]:


def get_motor_angles():
    """ get the gimbal angle, as measured by the IMU
        units are in degrees """
    msg = send_data('CMD_GET_ANGLES')
    
    # the gimbal returns 18 bytes of data
    # 3 axis, each having an angle, target angle and target speed
    gimbal_state = ser.read(18)  # read (up to) 18 bytes

    if gimbal_state == b'':
        print('no response from controller')
        return
    
    # only interested in the angles
    # they each arrive in 14-bit resolution and require scaling
    scaling = 0.02197265625  # scales to degrees
    IMU = {
        'roll': as_int(gimbal_state[0:2]) * scaling,
        'pitch': as_int(gimbal_state[6:8]) * scaling,
        'yaw': as_int(gimbal_state[12:14] * scaling)
    }
    
    return IMU


# In[361]:


# # send_data('CMD_REALTIME_DATA_3')
# # print(ser.read(5))

# send_data('CMD_GET_ANGLES')
# print(ser.read(5))

# send_data('CMD_REALTIME_DATA')
# print(ser.read(5))

# get_motor_angles()
# print(ser.read(5))

send_data('CMD_EXECUTE_MENU',
          [np.uint8(L['CMD_GET_ANGLES'])], 1)
print(ser.read(5))

# message = [np.uint8(L['CMD_REALTIME_DATA'])]
# send_data('CMD_EXECUTE_MENU', message, 1)
# print(ser.read(5))


# In[348]:


def turn_off_motors():
    message = [
        np.uint8(L['SBGC_MENU_MOTOR_OFF'])
    ]
    return send_data('CMD_EXECUTE_MENU', message, 1)

def turn_on_motors():
    message = [
        np.uint8(L['SBGC_MENU_MOTOR_ON'])
    ]
    return send_data('CMD_EXECUTE_MENU', message, 1)


# # Debugging section

# In[230]:


i = 0
while True:
    #turn_off_motors()
    send_angle_command(0, 0.02197265625*2, -0.02197265625*2)
#     #turn_on_motors()
#     x = bytearray()
# #     x.append(np.uint8(ord('>')))
# #     x.append(np.uint8(69))
# #     x.append(np.uint8(1))
# #     x.append(np.uint8(70))
# #     x.append(np.uint8(12-i))
# #     x.append(np.uint8(12-i))
# #     ser.write(x)
#     x.append(np.uint8(ord('>')))
#     x.append(np.uint8(67))
#     x.append(np.uint8(13))
#     x.append(np.uint8(80))
    
#     x.append(np.uint8(2))
#     for ii in range(12):
#         x.append(np.uint8(i))
#     x.append(np.uint8(2 + 12*i))
#     ser.write(x)
    
#     if (i == 0):
#         i = 30
#     else:
#         i = 0
#     time.sleep(0.5)
#     ser.flushOutput()
#     ser.flushInput()
    time.sleep(0.1)
    #print(x.hex())
    print(i)
    i += 1


# In[338]:


x = 20
send_angle_command(0, -x, x)


# In[363]:


chr(0b00111110)


# In[365]:


0b01000101


# In[366]:


0b00000001


# In[367]:


0b01000110


# In[368]:


0b01001001


# In[371]:


0b01001001


# In[ ]:


print(chr(0b00111110))  # start char >
print(0b01000101)       # command id = 69 = start menu
print(0b00000001)       # data size = 1 = size of data
print(0b01000110)       # header checksum = 70
print(0b00001100)       # data = 12 = motor off
print(0b00001100)       # data checksum = 12


# In[ ]:


print(chr(0b00111110))  # start char >
print(0b01000011)       # command id = 67 = CMD_control
print(0b00001101)       # data size = 13
print(0b01010000)       # header checksum = 80
print(0b00000010)       # data = 2
print(0b00001100)       # data = 0


# In[ ]:


import time

arr = [0, 1, 2, 4, 8, 16, 32, 64, 128]
i = 0
while True:
    x = bytearray()
    x.append(np.uint8(arr[i]))
    x.append(np.uint8(arr[i]))
    ser.write(x)
    print('i = %i, num = %i' % (i, arr[i]))
    if i >= len(arr)-1:
        i = 0
    else:
        i += 1
    time.sleep(1)


# In[ ]:


time_width = 50 * 10**(-6)  # seconds
baud_rate = 115200  # Hz
print('time in seconds per bit = %f us' % (10**6 * 1/baud_rate))
print('divisions per bit = %f divs/bit' % (1/baud_rate / time_width))
print('divs per byte (9 bits) = %f' % (9/baud_rate/time_width))


# In[ ]:


reverse = True
print_binary_padded(0x3e, reversed=reverse)
print_binary_padded(0x6d, reversed=reverse)
print_binary_padded(0x00, reversed=reverse)
print_binary_padded(0x6d, reversed=reverse)


# In[ ]:


L = {
    'CMD_READ_PARAMS': 82,
    'CMD_WRITE_PARAMS': 87,
    'CMD_REALTIME_DATA': 68,
    'CMD_BOARD_INFO': 86,
    'CMD_CALIB_ACC': 65,
    'CMD_CALIB_GYRO': 103,
    'CMD_CALIB_EXT_GAIN': 71,
    'CMD_USE_DEFAULTS': 70,
    'CMD_CALIB_POLES': 80,
    'CMD_RESET': 114,
    'CMD_HELPER_DATA': 72,
    'CMD_CALIB_OFFSET': 79,
    'CMD_CALIB_BAT': 66,
    'CMD_MOTORS_ON': 77,
    'CMD_MOTORS_OFF': 109,
    'CMD_CONTROL': 67,
    'CMD_TRIGGER_PIN': 84,
    'CMD_EXECUTE_MENU': 69,
    'CMD_GET_ANGLES': 73,
    #'CMD_CONFIRM': 67,
    'CMD_BOARD_INFO_3': 20,
    'CMD_READ_PARAMS_3': 21,
    'CMD_WRITE_PARAMS_3': 22,
    'CMD_REALTIME_DATA_3': 23,
    'CMD_REALTIME_DATA_4': 25,
    'CMD_SELECT_IMU_3': 24,
    'CMD_READ_PROFILE_NAMES': 28,
    'CMD_WRITE_PROFILE_NAMES': 29,
    'CMD_QUEUE_PARAMS_INFO_3': 30,
    'CMD_SET_ADJ_VARS_VAL': 31,
    'CMD_SAVE_PARAMS_3': 32,
    'CMD_READ_PARAMS_EXT': 33,
    'CMD_WRITE_PARAMS_EXT': 34,
    'CMD_AUTO_PID': 35,
    'CMD_SERVO_OUT': 36,
    'CMD_I2C_WRITE_REG_BUF': 39,
    'CMD_I2C_READ_REG_BUF': 40,
    'CMD_WRITE_EXTERNAL_DATA': 41,
    'CMD_READ_EXTERNAL_DATA': 42,
    'CMD_READ_ADJ_VARS_CFG': 43,
    'CMD_WRITE_ADJ_VARS_CFG': 44,
    'CMD_API_VIRT_CH_CONTROL': 45,
    'CMD_ADJ_VARS_STATE': 46,
    'CMD_EEPROM_WRITE': 47,
    'CMD_EEPROM_READ': 48,
    'CMD_CALIB_INFO': 49,
    'CMD_BOOT_MODE_3': 51,
    'CMD_SYSTEM_STATE': 52,
    'CMD_READ_FILE': 53,
    'CMD_WRITE_FILE': 54,
    'CMD_FS_CLEAR_ALL': 55,
    'CMD_AHRS_HELPER': 56,
    'CMD_RUN_SCRIPT': 57,
    'CMD_SCRIPT_DEBUG': 58,
    'CMD_CALIB_MAG': 59,
    'CMD_GET_ANGLES_EXT': 61,
    'CMD_READ_PARAMS_EXT2': 62,
    'CMD_WRITE_PARAMS_EXT2': 63,
    'CMD_GET_ADJ_VARS_VAL': 64,
    'CMD_CALIB_MOTOR_MAG_LINK': 74,
    'CMD_GYRO_CORRECTION': 75,
    'CMD_DATA_STREAM_INTERVAL': 85,
    'CMD_REALTIME_DATA_CUSTOM': 88,
    'CMD_BEEP_SOUND': 89,
    'CMD_ENCODERS_CALIB_OFFSET_4': 26,
    'CMD_ENCODERS_CALIB_FLD_OFFSET_4': 27,
    'CMD_CONTROL_CONFIG': 90,
    'CMD_CALIB_ORIENT_CORR': 91,
    'CMD_COGGING_CALIB_INFO': 92,
    'CMD_CALIB_COGGING': 93,
    'CMD_CALIB_ACC_EXT_REF': 94,
    'CMD_PROFILE_SET': 95,
    'CMD_CAN_DEVICE_SCAN': 96,
    'CMD_CAN_DRV_HARD_PARAMS': 97,
    'CMD_CAN_DRV_STATE': 98,
    'CMD_CAN_DRV_CALIBRATE': 99,
    'CMD_READ_RC_INPUTS': 100,
    'CMD_REALTIME_DATA_CAN_DRV': 101,
    'CMD_EVENT': 102,
    'CMD_READ_PARAMS_EXT3': 104,
    'CMD_WRITE_PARAMS_EXT3': 105,
    'CMD_EXT_IMU_DEBUG_INFO': 106,
    'CMD_SET_DEBUG_PORT': 249,
    'CMD_MAVLINK_INFO': 250,
    'CMD_MAVLINK_DEBUG': 251,
    'CMD_DEBUG_VARS_INFO_3': 253,
    'CMD_DEBUG_VARS_3': 254,
    'CMD_ERROR': 255
}

