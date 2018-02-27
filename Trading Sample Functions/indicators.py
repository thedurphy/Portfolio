import numpy  
import pandas as pd
import numpy as np
import math as m
import glob
import os
from keras.models import load_model
import forexnn




#Moving Average  
def MA(df, n):  
    MA = pd.Series(pd.rolling_mean(df['Close'], n), name = 'MA_' + str(n))  
    df = df.join(MA)  
    return df

#Exponential Moving Average  
def EMA(df, n):  
    EMA = pd.Series(pd.ewma(df['Close'], span = n, min_periods = n - 1), name = 'EMA_' + str(n))  
    df = df.join(EMA)  
    return df

#Momentum  
def MOM(df, n):  
    M = pd.Series(df['Close'].diff(n), name = 'Momentum_' + str(n))  
    df = df.join(M)  
    return df

#Rate of Change  
def ROC(df, n):  
    M = df['Close'].diff(n - 1)  
    N = df['Close'].shift(n - 1)  
    ROC = pd.Series(M / N, name = 'ROC_' + str(n))  
    df = df.join(ROC)  
    return df

#Average True Range  
def ATR(df, n):  
    i = 0  
    TR_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = pd.Series(pd.ewma(TR_s, span = n, min_periods = n), name = 'ATR_' + str(n))  
    df = df.join(ATR)  
    return df

#Bollinger Bands  
def BBANDS(df, n):  
    MA = pd.Series(df['Close'].rolling(n).mean())  
    MSD = pd.Series(df['Close'].rolling(n).std())  
    b1 = 4 * MSD / MA  
    B1 = pd.Series(b1, name = 'BollingerB_' + str(n))  
    df = df.join(B1)  
    b2 = (df['Close'] - MA + 2 * MSD) / (4 * MSD)  
    B2 = pd.Series(b2, name = 'Bollingerb_' + str(n))  
    df = df.join(B2)  
    return df

#Pivot Points, Supports and Resistances  
def PPSR(df):  
    PP = pd.Series((df['High'] + df['Low'] + df['Close']) / 3)  
    R1 = pd.Series(2 * PP - df['Low'])  
    S1 = pd.Series(2 * PP - df['High'])  
    R2 = pd.Series(PP + df['High'] - df['Low'])  
    S2 = pd.Series(PP - df['High'] + df['Low'])  
    R3 = pd.Series(df['High'] + 2 * (PP - df['Low']))  
    S3 = pd.Series(df['Low'] - 2 * (df['High'] - PP))  
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}  
    PSR = pd.DataFrame(psr)  
    df = df.join(PSR)  
    return df

#Stochastic oscillator %K  
def STOK(df):  
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name = 'SO%k')  
    df = df.join(SOk)  
    return df

#Stochastic oscillator %D  
def STO(df, n):  
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name = 'SO%k')  
    SOd = pd.Series(SOk.ewm(com = n, min_periods = n - 1).mean(), name = 'SO%d_' + str(n))  
    so = {'SOk' : SOk, 'SOd' : SOd}
    so = pd.DataFrame(so)
    df = df.join(so)  
    return df

#Trix  
def TRIX(df, n):  
    EX1 = pd.ewma(df['Close'], span = n, min_periods = n - 1)  
    EX2 = pd.ewma(EX1, span = n, min_periods = n - 1)  
    EX3 = pd.ewma(EX2, span = n, min_periods = n - 1)  
    i = 0  
    ROC_l = [0]  
    while i + 1 <= df.index[-1]:  
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]  
        ROC_l.append(ROC)  
        i = i + 1  
    Trix = pd.Series(ROC_l, name = 'Trix_' + str(n))  
    df = df.join(Trix)  
    return df

#Average Directional Movement Index  
def ADX(df, n, n_ADX):  
    i = 0  
    UpI = []  
    DoI = []  
    while i + 1 <= df.index[-1]:  
        UpMove = df.get_value(i + 1, 'High') - df.get_value(i, 'High')  
        DoMove = df.get_value(i, 'Low') - df.get_value(i + 1, 'Low')  
        if UpMove > DoMove and UpMove > 0:  
            UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:  
            DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    i = 0  
    TR_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = pd.Series(pd.ewma(TR_s, span = n, min_periods = n))  
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI)  
    PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1) / ATR)  
    NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1) / ATR)  
    ADX = pd.Series(pd.ewma(abs(PosDI - NegDI) / (PosDI + NegDI), span = n_ADX, min_periods = n_ADX - 1), name = 'ADX_' + str(n) + '_' + str(n_ADX))  
    df = df.join(ADX)  
    return df

#MACD, MACD Signal and MACD difference  
def MACD(df, n_fast, n_slow):  
    EMAfast = pd.Series(df['Close'].ewm(com = n_fast, min_periods = n_slow - 1).mean())
    EMAslow = pd.Series(df['Close'].ewm(com = n_slow, min_periods = n_slow - 1).mean())
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))  
    MACDsign = pd.Series(MACD.ewm(com = 9, min_periods = 8).mean(), name = 'MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))  
    df = df.join(MACD)  
    df = df.join(MACDsign)  
    df = df.join(MACDdiff)  
    return df

#Mass Index  
def MassI(df):  
    Range = df['High'] - df['Low']  
    EX1 = pd.ewma(Range, span = 9, min_periods = 8)  
    EX2 = pd.ewma(EX1, span = 9, min_periods = 8)  
    Mass = EX1 / EX2  
    MassI = pd.Series(pd.rolling_sum(Mass, 25), name = 'Mass Index')  
    df = df.join(MassI)  
    return df

#Vortex Indicator: http://www.vortexindicator.com/VFX_VORTEX.PDF  
def Vortex(df, n):  
    i = 0  
    TR = [0]  
    while i < df.index[-1]:  
        Range = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        TR.append(Range)  
        i = i + 1  
    i = 0  
    VM = [0]  
    while i < df.index[-1]:  
        Range = abs(df.get_value(i + 1, 'High') - df.get_value(i, 'Low')) - abs(df.get_value(i + 1, 'Low') - df.get_value(i, 'High'))  
        VM.append(Range)  
        i = i + 1  
    VI = pd.Series(pd.rolling_sum(pd.Series(VM), n) / pd.rolling_sum(pd.Series(TR), n), name = 'Vortex_' + str(n))  
    df = df.join(VI)  
    return df





#KST Oscillator  
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):  
    M = df['Close'].diff(r1 - 1)  
    N = df['Close'].shift(r1 - 1)  
    ROC1 = M / N  
    M = df['Close'].diff(r2 - 1)  
    N = df['Close'].shift(r2 - 1)  
    ROC2 = M / N  
    M = df['Close'].diff(r3 - 1)  
    N = df['Close'].shift(r3 - 1)  
    ROC3 = M / N  
    M = df['Close'].diff(r4 - 1)  
    N = df['Close'].shift(r4 - 1)  
    ROC4 = M / N  
    KST = pd.Series(pd.rolling_sum(ROC1, n1) + pd.rolling_sum(ROC2, n2) * 2 + pd.rolling_sum(ROC3, n3) * 3 + pd.rolling_sum(ROC4, n4) * 4, name = 'KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(n2) + '_' + str(n3) + '_' + str(n4))  
    df = df.join(KST)  
    return df

#Relative Strength Index  
def RSI(df, n):  
    i = 0  
    UpI = [0]  
    DoI = [0]  
    while i + 1 <= df.index[-1]:  
        UpMove = df.at[i + 1, 'High'] - df.at[i, 'High']
        DoMove = df.at[i, 'Low'] - df.at[i + 1, 'Low']
        if UpMove > DoMove and UpMove > 0:  
            UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:  
            DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI)  
    PosDI = pd.Series(UpI.ewm(com = n, min_periods = n - 1).mean())  
    NegDI = pd.Series(DoI.ewm(com = n, min_periods = n - 1).mean())  
    RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI_' + str(n))  
    df = df.join(RSI)  
    return df

#True Strength Index  
def TSI(df, r, s):  
    M = pd.Series(df['Close'].diff(1))  
    aM = abs(M)
    EMA1 = pd.Series(pd.ewma(M, span = r, min_periods = r - 1))  
    aEMA1 = pd.Series(pd.ewma(aM, span = r, min_periods = r - 1))  
    EMA2 = pd.Series(pd.ewma(EMA1, span = s, min_periods = s - 1))  
    aEMA2 = pd.Series(pd.ewma(aEMA1, span = s, min_periods = s - 1))  
    TSI = pd.Series(EMA2 / aEMA2, name = 'TSI_' + str(r) + '_' + str(s))  
    df = df.join(TSI)  
    return df

#Accumulation/Distribution  
def ACCDIST(df, n):  
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']  
    M = ad.diff(n - 1)  
    N = ad.shift(n - 1)  
    ROC = M / N  
    AD = pd.Series(ROC, name = 'Acc/Dist_ROC_' + str(n))  
    df = df.join(AD)  
    return df

#Chaikin Oscillator  
def Chaikin(df):  
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']  
    Chaikin = pd.Series(pd.ewma(ad, span = 3, min_periods = 2) - pd.ewma(ad, span = 10, min_periods = 9), name = 'Chaikin')  
    df = df.join(Chaikin)  
    return df

#Money Flow Index and Ratio  
def MFI(df, n):  
    PP = (df['High'] + df['Low'] + df['Close']) / 3  
    i = 0  
    PosMF = [0]  
    while i < df.index[-1]:  
        if PP[i + 1] > PP[i]:  
            PosMF.append(PP[i + 1] * df.get_value(i + 1, 'Volume'))  
        else:  
            PosMF.append(0)  
        i = i + 1  
    PosMF = pd.Series(PosMF)  
    TotMF = PP * df['Volume']  
    MFR = pd.Series(PosMF / TotMF)  
    MFI = pd.Series(pd.rolling_mean(MFR, n), name = 'MFI_' + str(n))  
    df = df.join(MFI)  
    return df

#On-balance Volume  
def OBV(df, n):  
    i = 0  
    OBV = [0]  
    while i < df.index[-1]:  
        if df.at[i + 1, 'Close'] - df.at[i, 'Close'] > 0:  
            OBV.append(df.at[i + 1, 'Volume'])  
        if df.at[i + 1, 'Close'] - df.at[i, 'Close'] == 0:  
            OBV.append(0)  
        if df.at[i + 1, 'Close'] - df.at[i, 'Close'] < 0:  
            OBV.append(-df.at[i + 1, 'Volume'])  
        i = i + 1  
    OBV = pd.Series(OBV)
    OBV_ma = pd.Series(OBV.rolling(n).mean(), name = 'OBV_' + str(n))  
    df = df.join(OBV_ma)  
    return df

#Force Index  
def FORCE(df, n):  
    F = pd.Series(df['Close'].diff(n) * df['Volume'].diff(n), name = 'Force_' + str(n))  
    df = df.join(F)  
    return df

#Ease of Movement  
def EOM(df, n):  
    EoM = (df['High'].diff(1) + df['Low'].diff(1)) * (df['High'] - df['Low']) / (2 * df['Volume'])  
    Eom_ma = pd.Series(pd.rolling_mean(EoM, n), name = 'EoM_' + str(n))  
    df = df.join(Eom_ma)  
    return df

#Commodity Channel Index  
def CCI(df, n):  
    PP = (df['High'] + df['Low'] + df['Close']) / 3  
    CCI = pd.Series((PP - PP.rolling(n).mean()) / PP.rolling(n).std(), name = 'CCI_' + str(n))  
    df = df.join(CCI)  
    return df

#Coppock Curve  
def COPP(df, n):  
    M = df['Close'].diff(int(n * 11 / 10) - 1)  
    N = df['Close'].shift(int(n * 11 / 10) - 1)  
    ROC1 = M / N  
    M = df['Close'].diff(int(n * 14 / 10) - 1)  
    N = df['Close'].shift(int(n * 14 / 10) - 1)  
    ROC2 = M / N  
    Copp = pd.Series(pd.ewma(ROC1 + ROC2, span = n, min_periods = n), name = 'Copp_' + str(n))  
    df = df.join(Copp)  
    return df

#Keltner Channel  
def KELCH(df, n):  
    KelChM = pd.Series(pd.rolling_mean((df['High'] + df['Low'] + df['Close']) / 3, n), name = 'KelChM_' + str(n))  
    KelChU = pd.Series(pd.rolling_mean((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3, n), name = 'KelChU_' + str(n))  
    KelChD = pd.Series(pd.rolling_mean((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3, n), name = 'KelChD_' + str(n))  
    df = df.join(KelChM)  
    df = df.join(KelChU)  
    df = df.join(KelChD)  
    return df

#Ultimate Oscillator  
def ULTOSC(df):  
    i = 0  
    TR_l = [0]  
    BP_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.at[i + 1, 'High'], df.at[i, 'Close']) - min(df.at[i + 1, 'Low'], df.at[i, 'Close'])  
        TR_l.append(TR)  
        BP = df.at[i + 1, 'Close'] - min(df.at[i + 1, 'Low'], df.at[i, 'Close'])  
        BP_l.append(BP)  
        i = i + 1 
    TR_l = pd.Series(TR_l)
    BP_l = pd.Series(BP_l)
    UltO = pd.Series((4 * BP_l.rolling(7).sum() / TR_l.rolling(7).sum()) + (2 * BP_l.rolling(14).sum() / TR_l.rolling(14).sum()) + (BP_l.rolling(28).sum() / BP_l.rolling(28).sum()), name = 'Ultimate_Osc')  
    df = df.join(UltO)  
    return df

#Donchian Channel  
def DONCH(df, n):  
    i = 0  
    DC_l = []  
    while i < n - 1:  
        DC_l.append(0)  
        i = i + 1  
    i = 0  
    while i + n - 1 < df.index[-1]:  
        DC = max(df['High'].ix[i:i + n - 1]) - min(df['Low'].ix[i:i + n - 1])  
        DC_l.append(DC)  
        i = i + 1  
    DonCh = pd.Series(DC_l, name = 'Donchian_' + str(n))  
    DonCh = DonCh.shift(n - 1)  
    df = df.join(DonCh)  
    return df

#Standard Deviation  
def STDDEV(df, n):  
    df = df.join(pd.Series(pd.rolling_std(df['Close'], n), name = 'STD_' + str(n)))  
    return df  


'''
My Custom indicators
'''

## Neural Net Based Indicator
def nnind(df, lookback, normalize_col):
    latest_model = max(glob.glob('nnind/*'), key=os.path.getctime)
    latest_model = load_model(latest_model)
    s, t = forexnn.pd2seq(df, lookback, -2, normalize_col)
    pp = latest_model.predict(s)
    pp = pp[:, 0]
    pp = np.insert(pp, 0, [None for _ in range(lookback)])
    pp
    df.insert(loc = df.shape[1]-2, column = 'UPP', value = pp)
    return df.dropna().reset_index(drop=True)

def heikin_ashi(df):
    o = [] #OPEN
    h = [] #HIGH
    l = [] #LOW
    c = [] #CLOSE
    for index, row in df.iterrows():
        if index == 0:
            c.append(row[['Open', 'High', 'Low', 'Close']].mean())
            o.append(row[['Open', 'Close']].mean())
            l.append(row['Low'])            
            h.append(row['High'])
        else:
            ha_close = row[['Open', 'High', 'Low', 'Close']].mean()
            ha_open = np.mean([o[index-1], c[index-1]])
            c.append(ha_close)
            o.append(ha_open)
            l.append(np.min([row['Low'], ha_open, ha_close]))
            h.append(np.max([row['High'], ha_open, ha_close]))
    
    df['ha_Open'] = o
    df['ha_High'] = h
    df['ha_Low'] = l
    df['ha_Close'] = c
    return df

## Adaptive Trend and Cycle Following
def atcf(df):
    import numpy as np
    FATLBuffer=np.array([0.0040364019,
     0.0130129076,
     0.000786016,
     0.0005541115,
     -0.004771771,
     -0.00720034,
     -0.0067093714,
     -0.0023824623,
     0.0040444064,
     0.0095711419,
     0.0110573605,
     0.0069480557,
     -0.0016060704,
     -0.0108597376,
     -0.0160483392,
     -0.013674485,
     -0.0036771622,
     0.0100299086,
     0.0208778257,
     0.0226522218,
     0.0128149838,
     -0.0055774838,
     -0.0244141482,
     -0.0338917071,
     -0.0272432537,
     -0.0047706151,
     0.0249252327,
     0.0477818607,
     0.0502044896,
     0.0259609206,
     -0.0190795053,
     -0.0670110374,
     -0.0933058722,
     -0.0760367731,
     -0.0054034585,
     0.1104506886,
     0.2460452079,
     0.3658689069,
     0.436040945])

    SATLBuffer=np.array([0.0161380976,
     0.0049516078,
     0.0056078229,
     0.0062325477,
     0.0068163569,
     0.0073260526,
     0.007754382,
     0.0080741359,
     0.0082901022,
     0.0083694798,
     0.0083037666,
     0.0080376628,
     0.0076266453,
     0.0070340085,
     0.0062194591,
     0.0052380201,
     0.0040471369,
     0.0026845693,
     0.0011421469,
     -0.000553518,
     -0.0023956944,
     -0.0043466731,
     -0.006384185,
     -0.008473677,
     -0.0105938331,
     -0.0126796776,
     -0.0147139428,
     -0.0166377699,
     -0.0184126992,
     -0.0199924534,
     -0.0213300463,
     -0.02237969,
     -0.0231017777,
     -0.0234566315,
     -0.0234080863,
     -0.0229204861,
     -0.0219739146,
     -0.0205446727,
     -0.0186164872,
     -0.0161875265,
     -0.0132507215,
     -0.0098190256,
     -0.0059060082,
     -0.0015350359,
     0.0032639979,
     0.0084512448,
     0.0139807863,
     0.0198005183,
     0.0258537721,
     0.0320735368,
     0.038395995,
     0.0447468229,
     0.0510534242,
     0.0572428925,
     0.0632381578,
     0.0689666682,
     0.0743569346,
     0.079340635,
     0.0838544303,
     0.0878391006,
     0.091243709,
     0.0940230544,
     0.0961401078,
     0.0975682269,
     0.0982862174])

    RFTLBuffer= np.array([0.0018747783,
     0.0060440751,
     0.000365079,
     0.0002573669,
     -0.0022163335,
     -0.0033443253,
     -0.0031162862,
     -0.0011065767,
     0.0018784961,
     0.0044454862,
     0.0051357867,
     0.0032271474,
     -0.0007459678,
     -0.0050439973,
     -0.007453935,
     -0.0063513565,
     -0.001707923,
     0.0046585685,
     0.0096970755,
     0.0105212252,
     0.0059521459,
     -0.002590561,
     -0.011339583,
     -0.0157416029,
     -0.0126536111,
     -0.0022157966,
     0.0115769653,
     0.0221931304,
     0.0233183633,
     0.0120580088,
     -0.0088618137,
     -0.0311244617,
     -0.0433375629,
     -0.0353166244,
     -0.0025097319,
     0.0513007762,
     0.1142800493,
     0.169934286,
     0.2025269304,
     0.2025269304,
     0.169934286,
     0.1142800493,
     0.0513007762,
     -0.0025097319])

    RSTLBuffer= np.array([0.0073925495,
     0.0022682355,
     0.0025688349,
     0.0028550092,
     0.0031224409,
     0.0033559226,
     0.003552132,
     0.0036986051,
     0.003797535,
     0.0038338964,
     0.0038037944,
     0.0036818974,
     0.0034936183,
     0.0032221429,
     0.0028490136,
     0.0023994354,
     0.0018539149,
     0.0012297491,
     0.0005231953,
     -0.0002535559,
     -0.0010974211,
     -0.0019911267,
     -0.0029244713,
     -0.0038816271,
     -0.0048528295,
     -0.0058083144,
     -0.0067401718,
     -0.0076214397,
     -0.0084345004,
     -0.0091581551,
     -0.0097708805,
     -0.0102517019,
     -0.0105824763,
     -0.010745028,
     -0.0107227904,
     -0.0104994302,
     -0.0100658241,
     -0.0094111161,
     -0.0085278517,
     -0.0074151919,
     -0.0060698985,
     -0.0044979052,
     -0.0027054278,
     -0.0007031702,
     0.0014951741,
     0.0038713513,
     0.0064043271,
     0.0090702334,
     0.0118431116,
     0.0146922652,
     0.0175884606,
     0.0204976517,
     0.0233865835,
     0.0262218588,
     0.0289681736,
     0.0315922931,
     0.0340614696,
     0.0363444061,
     0.0384120882,
     0.0402373884,
     0.0417969735,
     0.0430701377,
     0.0440399188,
     0.0446941124,
     0.04502301,
     0.04502301,
     0.0446941124,
     0.0440399188,
     0.0430701377,
     0.0417969735,
     0.0402373884,
     0.0384120882,
     0.0363444061,
     0.0340614696,
     0.0315922931,
     0.0289681736,
     0.0262218588,
     0.0233865835,
     0.0204976517,
     0.0175884606,
     0.0146922652,
     0.0118431116,
     0.0090702334,
     0.0064043271,
     0.0038713513,
     0.0014951741,
     -0.0007031702,
     -0.0027054278,
     -0.0044979052,
     -0.0060698985,
     -0.0074151919])

    PCCIBuffer = np.array([-0.00022625,
     0.01047723,
     0.00373774,
     0.00144873,
     -0.00198138,
     -0.00509042,
     -0.00610944,
     -0.00405186,
     0.00059669,
     0.00583939,
     0.00897632,
     0.0079338,
     0.0024875,
     -0.00527241,
     -0.01164808,
     -0.01301467,
     -0.00775517,
     0.00252297,
     0.01328613,
     0.01895659,
     0.01565175,
     0.00350063,
     -0.0127291,
     -0.02525534,
     -0.02678947,
     -0.01442877,
     0.00816037,
     0.03120441,
     0.04267885,
     0.03382809,
     0.00422528,
     -0.03611022,
     -0.06903411,
     -0.07451349,
     -0.03871426,
     0.03934469,
     0.14548806,
     0.25372851,
     0.33441085,
     0.3642399])

    # Negative of the entire summation
    RBCIBuffer= np.array([-1.615617397,
     -1.377516078,
     -1.513691845,
     -1.276670767,
     -0.6386689841,
     0.3089253193,
     1.353679243,
     2.228994128,
     2.697374234,
     2.627040982,
     2.057741075,
     1.188784148,
     0.3278853523,
     -0.2245901578,
     -0.2797065802,
     0.1561848839,
     0.8771442423,
     1.54127228,
     1.796998735,
     1.420216677,
     0.4132650195,
     -0.9760510577,
     -2.332625794,
     -3.221651455,
     -3.358959682,
     -2.732295856,
     -1.62749164,
     -0.5359717954,
     0.0260722294,
     -0.2740437883,
     -1.431012658,
     -3.067145982,
     -4.54225353,
     -5.180855695,
     -4.535883446,
     -2.591938701,
     0.1815496232,
     2.960440887,
     4.851086292,
     5.234224328,
     4.04333043,
     1.861734281,
     -0.2191111431,
     -0.9559211961,
     0.5817527531,
     4.596423992,
     10.35240127,
     16.27023906,
     20.32661158,
     20.65662104,
     16.17628156,
     7.023163695,
     -5.341847567,
     -18.42774496,
     -29.33398965,
     -35.52418194])    
    FATL = [0 for _ in range(len(FATLBuffer)-1)]
    SATL = [0 for _ in range(len(SATLBuffer)-1)]
    RFTL = [0 for _ in range(len(RFTLBuffer)-1)]
    RSTL = [0 for _ in range(len(RSTLBuffer)-1)]
    RBCI = [0 for _ in range(len(RBCIBuffer)-1)]
    PCCI = [0 for _ in range(len(PCCIBuffer)-1)]
    temp_mat = df.copy()[['High', 'Low', 'Close']].as_matrix()
    for i in range(len(FATLBuffer)-1, len(temp_mat)):
        # FATL equation BufferSummation
        FATL.append(np.sum(temp_mat[i-(len(FATLBuffer)-1):i+1,-1]*FATLBuffer))
        if i >= len(PCCIBuffer)-1:
            # PCCI equation (High[i]-Low[i])/2-BufferSummation
            pccibuffersum = (np.sum(temp_mat[i-(len(PCCIBuffer)-1):i+1,-1]*PCCIBuffer))
            PCCI.append((temp_mat[i,0]-temp_mat[i,1])/2-pccibuffersum)
            if i >= len(RFTLBuffer)-1:
                # RFTL equation BufferSummation
                RFTL.append(np.sum(temp_mat[i-(len(RFTLBuffer)-1):i+1,-1]*RFTLBuffer))
                # FTLM equation FATL-RFTL
                if i >= len(RBCIBuffer)-1:
                    #RBCI equation -1*(BufferSummation)
                    RBCI.append(-1*np.sum(temp_mat[i-(len(RBCIBuffer)-1):i+1,-1]*RBCIBuffer))
                    if i >= len(SATLBuffer)-1:
                        # SATL equation BufferSummation
                        SATL.append(np.sum(temp_mat[i-(len(SATLBuffer)-1):i+1,-1]*SATLBuffer))
                        if i >= len(RSTLBuffer)-1:
                            # RSTL equation BufferSummation
                            RSTL.append(np.sum(temp_mat[i-(len(RSTLBuffer)-1):i+1,-1]*RSTLBuffer))
                            # STLM equation SATL-RSTL
    FATL = np.array(FATL)          
    SATL = np.array(SATL)
    RFTL = np.array(RFTL)
    RSTL = np.array(RSTL)
    FTLM = FATL-RFTL
    STLM = SATL-RSTL
    RBCI = np.array(RBCI)
    PCCI = np.array(PCCI)
    return np.array([FATL, SATL, RFTL, RSTL, FTLM, STLM, RBCI, PCCI]).T