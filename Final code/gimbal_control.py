
# coding: utf-8

# # SimpleBGC controller
# 
# Implemented using the serial API specified in www.basecamelectronics.com/serialapi/ and by looking at the source code for the arduino library, found at https://github.com/alexmos/sbgc-api-examples. It is very incomplete (I only added the commands I need) but should be easy enough to follow that any other commands can be added without much extra work.
# 
# Main commands (prefixed with `gimbal_control.`):
# 1. `turn_on_motors()`
# 2. `send_angle_command(roll, pitch, yaw)`
# 3. `angles_dict = get_motor_angles()`
# 4. `turn_off_motors()`
# 
# All angles are in degrees
# 
# ---
# 
# I find development using a notebook to be quite a bit easier than developing using a regular python file. Unfortunately, you can't import a `.ipynb` as a module. So, here's the workflow:
# 1. Use this file to understand the code and make changes
# 2. When you want to commit a change, click `Kernal > Restart and Clear Output` to remove your outputs + make the file a bit smaller (shows up as fewer lines in the git commit)
# 3. Run the command `jupyter nbconvert --to=python gimbal_control.ipynb` to generate a `.py` file which can be imported as a module. Just make sure to remove your debugging code beforehand! Lines like the `ser = serial.Serial(...)` should be kept in.

# In[ ]:


import numpy as np, serial, time


# In[ ]:


ser = serial.Serial('/dev/serial0',  # pin 8 = Tx, pin 10 = Rx, has been mapped to ttyAMA0
                    baudrate=115200,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=0.08)  # 80 ms timeout

#serial0 == '/dev/ttyS0', connected on physical pins 8 & 10.
# parity issues on ttyS0:
#     https://github.com/pyserial/pyserial/issues/196#issuecomment-323382976

# therefore, these pins have been mapped to ttyAMA0 which has better support
#serial1 == '/dev/ttyAMA0', connected internally to the Bluetooth hardware.


# In[ ]:


# incomplete lookup table - see the note in the second to last cell
L = {
    # menu items
    'CMD_EXECUTE_MENU': 69,
    'SBGC_MENU_MOTOR_ON': 11,
    'SBGC_MENU_MOTOR_OFF': 12,
    
    # cmd items
    'CMD_CONTROL': 67,
    'CMD_REALTIME_DATA_CUSTOM': 88
}


# ## Utils

# In[ ]:


def as_int(b):
    """ Converts b (a bytearray) into a signed integer.
        Assumes little endian """
    return int.from_bytes(b, byteorder='little', signed=True)

def as_uint(b):
    """ Converts b (a bytearray) into an unsigned integer.
        Assumes little endian """
    return int.from_bytes(b, byteorder='little', signed=False)

def print_binary_padded(hex_data, num_bits=8, reverse=False):
    """ Prints the hex value in binary, with zero padding
    
        eg: print_binary_padded(0x03) # --> 0b00000011
            print_binary_padded(0x0421, num_bits=16) # --> 0b0000010000100001
    """
    if reverse is True:
        print('LSB first: 0b' + bin(hex_data)[2:].zfill(num_bits)[::-1])
    else:
        print('0b' + bin(hex_data)[2:].zfill(num_bits))  # [2:] to get rid of the '0b' part

def print_hex_nicely(hex_string):
    """ Prints a hex string in groups of bytes
        Useful when you want to decode a bytearray by eye
    
        eg. print('Response:   ', end='')
            print_hex_nicely(b'>X\x08`\x0c\xf3z\xffr\xfc\x8e\x03w'.hex())
    """
    [print(hex_string[i*2:i*2+2] + ' ', end='') for i in range(len(hex_string)//2)]
    print()

def print_twos_complement(number, num_bits=16):
    """ Another function made while debugging.
    >>> print_twos_complement(-4, num_bits=8)  # --> 11111100
    """
    print(format(number % (1 << num_bits), '0' + str(num_bits) + 'b'))


# ## Main command functions

# In[ ]:


def send_data(command_ID, data=None, data_size=0, debug=False):
    """ Compose and send a command to the SimpleBGC
    Inputs:
        command_ID:     a string, such as 'CMD_GET_ANGLES'
        data:           a list of numpy scalars
        data_size:      the number of bytes in 'data'

    Format:
        head:
            start_char = ord('>') = 0x3E. 1u
            command_ID. 1u
            data_size. 1u. Can be 0
            header_checksum = (command ID + data_size) % 255. 1u

        body:
            array_of_bytes_of_length_data_size
            body_checksum. 1u
    """
    # compose head:
    start_char = np.uint8(ord('>'))
    header_checksum = np.uint8(L[command_ID] + data_size)
    
    message = bytearray()
    message.append(start_char)
    message.append(L[command_ID])
    message.append(np.uint8(data_size))
    message.append(header_checksum)

    # compose body:
    body_checksum = 0
    if data_size > 0:
        for d in data:
            if d.nbytes == 1:
                message.append(d)
                body_checksum += d
            elif d.nbytes == 2:
                d_bytes = d.tobytes()
                message.append(d_bytes[0])  # working with little endian
                message.append(d_bytes[1])
                body_checksum += d_bytes[0] + d_bytes[1]
            else:
                print('Haven\'t yet built in functionality for 3 or more bytes')
        
        if debug: print('body_checksum = %i' % body_checksum)

        message.append(np.uint8(body_checksum))
    
    ser.flushInput()
    ser.flushOutput()
    ser.write(message)
    
    if debug:
        print('message sent:\t\t', end='')
        print_hex_nicely(message.hex())
    
    return message


# In[ ]:


def send_angle_command(roll, pitch, yaw, debug=False):
    """ send an angle command to the gimbal
        roll, pitch and yaw are in degrees """
    
    # the spec sheet says to send CONTROL_MODE three times (one for each axis)
    # BUT it doesn't seem to work when I do that. Not sure why

    scaling = 0.02197265625  # units/degree. 2**15 * 0.02197265625 == 720
    message = [
        np.uint8(2),                # CONTROL_MODE = MODE_ANGLE
        np.int16(0),                # roll speed
        np.int16(roll/scaling),     # roll angle
        np.int16(0),                # pitch speed
        np.int16(pitch/scaling),    # pitch angle
        np.int16(0),                # yaw speed
        np.int16(yaw/scaling)       # yaw angle
    ]

    return send_data('CMD_CONTROL', data=message, data_size=13, debug=debug)


# In[ ]:


def send_speed_command(roll, pitch, yaw, debug=False):
    """ send a speed command to the gimbal
        roll, pitch and yaw are in degrees/second """
    
    # the spec sheet says to send CONTROL_MODE three times (one for each axis)
    # BUT it doesn't seem to work when I do that. Not sure why
    
    scaling = 0.1220740379          # units/degree/s
    message = [
        np.uint8(1),                # CONTROL_MODE = MODE_SPEED
        np.int16(roll/scaling),     # roll speed
        np.int16(0),                # roll angle
        np.int16(pitch/scaling),    # pitch speed
        np.int16(0),                # pitch angle
        np.int16(yaw/scaling),      # yaw speed
        np.int16(0)                 # yaw angle
    ]

    return send_data('CMD_CONTROL', data=message, data_size=13, debug=debug)


# In[ ]:


def get_motor_angles(debug=False):
    """ Get the gimbal's angles, as measured by the IMU and estimated by the SBGC EKF
        Units are in degrees.
        
        One could extend this to fetch other useful data as well - simple bit shift
        and then correct extraction of the data. See page 47 of the serial api doc.
    """

    msg = [np.uint8(1),  # 1 = activated bit 0
           np.uint8(0), np.uint8(0), np.uint8(0), np.uint8(0), np.uint8(0),
           np.uint8(0), np.uint8(0), np.uint8(0), np.uint8(0)]#np.uint64(0)]  # empty flag stuff/reserved
    send_data('CMD_REALTIME_DATA_CUSTOM', msg, 10, debug)
    
    SBGC_response = ser.read(13)  # read (up to) 13 bytes

    if response_is_not_valid(SBGC_response):
        return 0
    
    # example responses:
    # 3e 58 08 60   f6 89   1a ff   fe ff   fe ff   92   # 0 deg angles
    # 3e 58 08 60   66 a5   76 ff   74 fc   90 03   83   # 20 deg angles

    # they each arrive in 14-bit resolution and require scaling
    scaling = 0.02197265625  # scales to degrees
    angles = {
        'timestamp_ms': as_uint(SBGC_response[4:6]),
        'roll':         as_int(SBGC_response[6:8]) * scaling,
        'pitch':        as_int(SBGC_response[8:10]) * scaling,
        'yaw':          as_int(SBGC_response[10:12]) * scaling
    }
    
    return angles


# In[ ]:


def turn_off_motors(debug=False):
    message = [np.uint8(L['SBGC_MENU_MOTOR_OFF'])]
    return send_data('CMD_EXECUTE_MENU', message, 1, debug)

def turn_on_motors(debug=False):
    message = [np.uint8(L['SBGC_MENU_MOTOR_ON'])]
    return send_data('CMD_EXECUTE_MENU', message, 1, debug)


# In[ ]:


def response_is_not_valid(SBGC_response):
    """ should be used as,
        if response_is_not_valid(SBGC_response):
            return -1
    """
    if SBGC_response == b'':
        print('No response response from controller')
        return True
    
    elif chr(SBGC_response[0]) != '>':
        print('Invalid start of response. Got:', SBGC_response)
        return True
    
    elif (SBGC_response[1] + SBGC_response[2] != SBGC_response[3]):
        print('Invalid header checksum in response. Got:', SBGC_response)
        return True
    
    # maybe look at the checksum for the body? though not knowing the datatype would be an issue
    # elif ...
    
    else:
        return False


# # Debugging section

# In[ ]:


# import time

# turn_off_motors()
# time.sleep(3)
# turn_off_motors()
# time.sleep(3)

# roll = 0
# for pitch in [45, 30, 15, 0, -10]:
#     for yaw in [-60, -30, 0, 30, 60]:
#         send_angle_command(roll, pitch, yaw)
#         time.sleep(2)
#         angles = get_motor_angles()
#         print('Roll = %d, pitch = %d, yaw = %d' % (angles['roll'], angles['pitch'], angles['yaw']))


# ## Complete lookup table
# Not used as implementing a command isn't always as simple as sending the command (need a special receive syntax, etc) so it's a bit misleading to show the whole table. Add stuff to the other `L` thing if needed

# In[ ]:


# L = {
#     'CMD_READ_PARAMS': 82,
#     'CMD_WRITE_PARAMS': 87,
#     'CMD_REALTIME_DATA': 68,
#     'CMD_BOARD_INFO': 86,
#     'CMD_CALIB_ACC': 65,
#     'CMD_CALIB_GYRO': 103,
#     'CMD_CALIB_EXT_GAIN': 71,
#     'CMD_USE_DEFAULTS': 70,
#     'CMD_CALIB_POLES': 80,
#     'CMD_RESET': 114,
#     'CMD_HELPER_DATA': 72,
#     'CMD_CALIB_OFFSET': 79,
#     'CMD_CALIB_BAT': 66,
#     'CMD_MOTORS_ON': 77,
#     'CMD_MOTORS_OFF': 109,
#     'CMD_CONTROL': 67,
#     'CMD_TRIGGER_PIN': 84,
#     'CMD_EXECUTE_MENU': 69,
#     'CMD_GET_ANGLES': 73,
#     'CMD_CONFIRM': 67,  # repeat of 67?
#     'CMD_BOARD_INFO_3': 20,
#     'CMD_READ_PARAMS_3': 21,
#     'CMD_WRITE_PARAMS_3': 22,
#     'CMD_REALTIME_DATA_3': 23,
#     'CMD_REALTIME_DATA_4': 25,
#     'CMD_SELECT_IMU_3': 24,
#     'CMD_READ_PROFILE_NAMES': 28,
#     'CMD_WRITE_PROFILE_NAMES': 29,
#     'CMD_QUEUE_PARAMS_INFO_3': 30,
#     'CMD_SET_ADJ_VARS_VAL': 31,
#     'CMD_SAVE_PARAMS_3': 32,
#     'CMD_READ_PARAMS_EXT': 33,
#     'CMD_WRITE_PARAMS_EXT': 34,
#     'CMD_AUTO_PID': 35,
#     'CMD_SERVO_OUT': 36,
#     'CMD_I2C_WRITE_REG_BUF': 39,
#     'CMD_I2C_READ_REG_BUF': 40,
#     'CMD_WRITE_EXTERNAL_DATA': 41,
#     'CMD_READ_EXTERNAL_DATA': 42,
#     'CMD_READ_ADJ_VARS_CFG': 43,
#     'CMD_WRITE_ADJ_VARS_CFG': 44,
#     'CMD_API_VIRT_CH_CONTROL': 45,
#     'CMD_ADJ_VARS_STATE': 46,
#     'CMD_EEPROM_WRITE': 47,
#     'CMD_EEPROM_READ': 48,
#     'CMD_CALIB_INFO': 49,
#     'CMD_BOOT_MODE_3': 51,
#     'CMD_SYSTEM_STATE': 52,
#     'CMD_READ_FILE': 53,
#     'CMD_WRITE_FILE': 54,
#     'CMD_FS_CLEAR_ALL': 55,
#     'CMD_AHRS_HELPER': 56,
#     'CMD_RUN_SCRIPT': 57,
#     'CMD_SCRIPT_DEBUG': 58,
#     'CMD_CALIB_MAG': 59,
#     'CMD_GET_ANGLES_EXT': 61,
#     'CMD_READ_PARAMS_EXT2': 62,
#     'CMD_WRITE_PARAMS_EXT2': 63,
#     'CMD_GET_ADJ_VARS_VAL': 64,
#     'CMD_CALIB_MOTOR_MAG_LINK': 74,
#     'CMD_GYRO_CORRECTION': 75,
#     'CMD_DATA_STREAM_INTERVAL': 85,
#     'CMD_REALTIME_DATA_CUSTOM': 88,
#     'CMD_BEEP_SOUND': 89,
#     'CMD_ENCODERS_CALIB_OFFSET_4': 26,
#     'CMD_ENCODERS_CALIB_FLD_OFFSET_4': 27,
#     'CMD_CONTROL_CONFIG': 90,
#     'CMD_CALIB_ORIENT_CORR': 91,
#     'CMD_COGGING_CALIB_INFO': 92,
#     'CMD_CALIB_COGGING': 93,
#     'CMD_CALIB_ACC_EXT_REF': 94,
#     'CMD_PROFILE_SET': 95,
#     'CMD_CAN_DEVICE_SCAN': 96,
#     'CMD_CAN_DRV_HARD_PARAMS': 97,
#     'CMD_CAN_DRV_STATE': 98,
#     'CMD_CAN_DRV_CALIBRATE': 99,
#     'CMD_READ_RC_INPUTS': 100,
#     'CMD_REALTIME_DATA_CAN_DRV': 101,
#     'CMD_EVENT': 102,
#     'CMD_READ_PARAMS_EXT3': 104,
#     'CMD_WRITE_PARAMS_EXT3': 105,
#     'CMD_EXT_IMU_DEBUG_INFO': 106,
#     'CMD_SET_DEBUG_PORT': 249,
#     'CMD_MAVLINK_INFO': 250,
#     'CMD_MAVLINK_DEBUG': 251,
#     'CMD_DEBUG_VARS_INFO_3': 253,
#     'CMD_DEBUG_VARS_3': 254,
#     'CMD_ERROR': 255
# }

