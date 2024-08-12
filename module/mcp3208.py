import spidev
from enum import IntEnum

spi = spidev.SpiDev()

# ADCの読み取りモード
class conv(IntEnum):
    """
    MCP3208の変換タイプを表す列挙型です。

    - diff: 差動変換モード
    - sgl: 単一変換モード
    """
    diff = 0
    sgl = 1

# ADC初期化
def open(port, cs, speed):
    """
    MCP3208 ADCをSPI経由で接続するための接続を開きます。

    Args:
        port (int): SPIポート番号。
        cs (int): チップセレクト（CS）ピン番号。
        speed (int): SPIクロック速度（Hz）。

    Returns:
        spidev.SpiDev: 接続を表すSPIデバイスオブジェクト。

    """
    spi.open(port, cs) #port 0,cs 0
    spi.max_speed_hz = speed
    return spi


# ADC読み取り関数
def read_data(channel, convtype = 1):
    """
    MCP3208の電圧を読み取る.
    
    Parameters:
    channel (int) :
        読み取るチャンネル番号(DIFFモードのときはIN+の番号)
    convtype (int) :
        読み取りモード(0:差分(DIFF) 1:絶対値(SGL))
    
    Returns:
    adc_convert (float) :
        読み取り値.
    """
    adc_send = [0b00000100 | (convtype << 1) | (channel >> 2),
                0b11000000 & (channel << 6),
                0b00000000]
    # print(bin(adc_send[0]), bin(adc_send[1]), bin(adc_send[2]))
    adc_receive = spi.xfer2(adc_send)
    adc_convert = ((0b00001111 & adc_receive[1]) << 8) | adc_receive[2]
    return(adc_convert)

# ADCのデータを電圧に変換
def read_voltage(channel, convtype=1, vref=5.0):
    """
    指定されたチャンネルからアナログデータを読み取り、電圧に変換して返す関数です。

    Parameters:
    channel (int): チャンネル番号
    convtype (int, optional): 変換タイプ (デフォルト値: 1)
    vref (float, optional): 参照電圧 (デフォルト値: 5.0)

    Returns:
    float: 変換された電圧値
    """
    data = read_data(channel, convtype)
    d_to_v = vref * data / 4096.0
    return d_to_v

# ADC終了関数
def close():
    """
    SPI通信を終了します。
    """
    spi.close()