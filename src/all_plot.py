from tkinter import E
import pandas as pd
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import matplotlib
import numpy as np


def allplot(A,B,C):
    file_path = '.\dat\CSV\%s*.csv'%C                                      
    csv = []
    for filename in glob.glob(file_path, recursive=True):
        csv.append(filename)
    
    csv_tqdm = tqdm(csv)
    
    voltage = []
    capacitance = []
    conductunce = []
    for i in csv_tqdm:
        # filename = i.split('\\')[3]
        filename_1 = i.split('\\')[-1][:-4]
        csv_tqdm.set_description(f'Processing {filename_1}')
    

        Data = pd.read_csv(".\dat\CSV\%s" % filename_1 + '.csv')

        # Value = "DataValue"

        # find_row = Data.loc[(Data['Value'] == Value)]
        # find_row = find_row.iloc[:,1:3]
        # print(find_row)

        # vol = csv['Voltage (V)']
        # cap = csv['Cp']
        # con = csv['G']


        find_columns_v = Data.loc[:,'Voltage (V)']
        find_columns_c = Data.loc[:,'Cp(F)']
        find_columns_g = Data.loc[:,' G(s)']
        # print(find_columns_v)
        # print(cap.values.tolist())
        Voltage = list(map(float, find_columns_v.tolist()))
        voltage.append(Voltage)
        Capacitance = list(map(float, find_columns_c.tolist()))
        capacitance.append(Capacitance)
        Conductunce = list(map(float, find_columns_g.tolist()))                # 배열을 리스트로 바꾸기
        conductunce.append(Conductunce)
        


    plt.rcParams['font.size'] = 20
    # plt.rc('font', size=15)
    plt.figure(figsize=(12, 9))

    # line1 = ax1.plot(VBias[0:round((len(VBias)/2))+1], CMeas[0:round((len(VBias)/2))+1], color='blue', linestyle='-',marker = matplotlib.markers.CARETRIGHT, markersize = 10,label = 'Capacitance')
    # ax1.plot(VBias[round((len(VBias)/2)):-1], CMeas[round((len(VBias)/2)):-1], color='blue', linestyle='-',marker = matplotlib.markers.CARETLEFT, markersize = 10)
    # ax1.set_ylabel('Capacitance ', color='blue', fontsize=20, labelpad = 10)
    # ax1.tick_params('y', colors='blue')

    Volt_1 = voltage[0]
    Volt_2 = voltage[1]
    Volt_3 = voltage[2]
    Volt_4 = voltage[3]

    Cap_1 = capacitance[0]
    Cap_2 = capacitance[1]
    Cap_3 = capacitance[2]
    Cap_4 = capacitance[3]
    plt.plot(Volt_1[0:round((len(Volt_1)/2))+1], Cap_1[0:round((len(Volt_1)/2))+1], color='blue', linestyle='-',marker = matplotlib.markers.CARETRIGHT, markersize = 10,label ='%s' % csv[0].split('\\')[-1][:-18].split('_')[-1])
    plt.plot(Volt_1[round((len(Volt_1)/2)):-1], Cap_1[round((len(Volt_1)/2)):-1], color='blue', linestyle='-',marker = matplotlib.markers.CARETLEFT, markersize = 10)
    
    plt.plot(Volt_2[0:round((len(Volt_1)/2))+1], Cap_2[0:round((len(Volt_1)/2))+1], color='orange', linestyle='-',marker = matplotlib.markers.CARETRIGHT, markersize = 10,label = '%s' % csv[1].split('\\')[-1][:-18].split('_')[-1])
    plt.plot(Volt_2[round((len(Volt_1)/2)):-1], Cap_2[round((len(Volt_1)/2)):-1], color='orange', linestyle='-',marker = matplotlib.markers.CARETLEFT, markersize = 10)

    plt.plot(Volt_3[0:round((len(Volt_1)/2))+1], Cap_3[0:round((len(Volt_1)/2))+1], color='green', linestyle='-',marker = matplotlib.markers.CARETRIGHT, markersize = 10,label = '%s' % csv[2].split('\\')[-1][:-18].split('_')[-1])
    plt.plot(Volt_3[round((len(Volt_1)/2)):-1], Cap_3[round((len(Volt_1)/2)):-1], color='green', linestyle='-',marker = matplotlib.markers.CARETLEFT, markersize = 10)

    plt.plot(Volt_4[0:round((len(Volt_1)/2))+1], Cap_4[0:round((len(Volt_1)/2))+1], color='red', linestyle='-',marker = matplotlib.markers.CARETRIGHT, markersize = 10,label = '%s' % csv[3].split('\\')[-1][:-18].split('_')[-1])
    plt.plot(Volt_4[round((len(Volt_1)/2)):-1], Cap_4[round((len(Volt_1)/2)):-1], color='red', linestyle='-',marker = matplotlib.markers.CARETLEFT, markersize = 10)
        # ax1.set_xlabel("Voltage [V]", fontsize=20)
        # lines = line1 + line2

        # labels = [l.get_label() for l in lines]
        # ax1.legend(lines, labels, loc='upper right')
        # plt.tight_layout()
            

    # plt.plot(voltage[0], capacitance[0], label='%s' % csv[0].split('\\')[-1][:-18].split('_')[-1])
    # plt.plot(voltage[1], capacitance[1], label='%s' % csv[1].split('\\')[-1][:-18].split('_')[-1])
    # plt.plot(voltage[2], capacitance[2], label='%s' % csv[2].split('\\')[-1][:-18].split('_')[-1])
    # # plt.plot(voltage[3], capacitance[3], label='%s' % csv[3].split('\\')[-1][:-18])
    # # plt.plot(voltage[4], capacitance[4], label='%s' % csv[4].split('\\')[-1][:-18])
        
    plt.title('%s'%C)
    plt.xlabel('Voltage [V]', labelpad=10)
    plt.ylabel('Capacitance [F]', labelpad=10)
    plt.legend()
    plt.grid(True)

    if A == 'T':
        plt.savefig('.\\res\\%s.png' % (C +' ALL'))

    if B == 'T':
        plt.show(block=False)
        plt.pause(1)
        plt.close()

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def standardplot_2(A,B,C):                                         # CV 측정 (1k, 10k, 100k, 1M  주파수 4개 같이 플랏) 4개 고정 
    file_path = '.\dat\CSV\%s*.csv'%C                                      
    csv = []
    for filename in glob.glob(file_path, recursive=True):
        csv.append(filename)
    
    csv_tqdm = tqdm(csv)
    
    voltage = []
    capacitance = []
    conductunce = []
    for i in csv_tqdm:
        # filename = i.split('\\')[3]
        filename_1 = i.split('\\')[-1][:-4]
        csv_tqdm.set_description(f'Processing {filename_1}')
    
        
        Data = pd.read_csv(".\dat\CSV\%s" % filename_1 + '.csv')
        
        # Value = "DataValue"

        # find_row = Data.loc[(Data['Value'] == Value)]
        # find_row = find_row.iloc[:,1:3]
        # print(find_row)

        # vol = csv['Voltage (V)']
        # cap = csv['Cp']
        # con = csv['G']


        find_columns_v = Data.loc[:,'Voltage (V)']
        find_columns_c = Data.loc[:,'Cp(F)']
        find_columns_g = Data.loc[:,' G(s)']
        # print(find_columns_v)
        # print(cap.values.tolist())
        Voltage = list(map(float, find_columns_v.tolist()))
        voltage.append(Voltage)
        Capacitance = list(map(float, find_columns_c.tolist()))
        capacitance.append(Capacitance)
        Conductunce = list(map(float, find_columns_g.tolist()))                # 배열을 리스트로 바꾸기
        conductunce.append(Conductunce)
        


    plt.rcParams['font.size'] = 27
    # plt.rc('font', size=15)
    plt.figure(figsize=(12, 9))

    # line1 = ax1.plot(VBias[0:round((len(VBias)/2))+1], CMeas[0:round((len(VBias)/2))+1], color='blue', linestyle='-',marker = matplotlib.markers.CARETRIGHT, markersize = 10,label = 'Capacitance')
    # ax1.plot(VBias[round((len(VBias)/2)):-1], CMeas[round((len(VBias)/2)):-1], color='blue', linestyle='-',marker = matplotlib.markers.CARETLEFT, markersize = 10)
    # ax1.set_ylabel('Capacitance ', color='blue', fontsize=20, labelpad = 10)
    # ax1.tick_params('y', colors='blue')
    
    Volt_1 = voltage[0]
    Volt_2 = voltage[1]
    Volt_3 = voltage[2]
    Volt_4 = voltage[3]

    Cap_1 = capacitance[0]
    Cap_2 = capacitance[1]
    Cap_3 = capacitance[2]
    Cap_4 = capacitance[3]

    # r1 = float(r)
    # r2 = float(r)
    # r3 = float(r)
    # r4 = float(r)
    
    r1= float(csv[0].split('\\')[-1][:-18].split('_')[-1].split(' ')[1])                        # 반지름 바꾸기 이거 쓰기
    r2= float(csv[1].split('\\')[-1][:-18].split('_')[-1].split(' ')[1])
    r3= float(csv[2].split('\\')[-1][:-18].split('_')[-1].split(' ')[1])
    r4= float(csv[3].split('\\')[-1][:-18].split('_')[-1].split(' ')[1])



    cap_standard_1 = []
    for i in Cap_1:
        x= i*1e6/(np.pi * (pow(r1,2)) * 1e-8)
        cap_standard_1.append(round(x,3))
    cap_standard_2 = []
    for z in Cap_2:
        x= z*1e6/(np.pi * (pow(r2,2)) * 1e-8)
        cap_standard_2.append(round(x,3))
    cap_standard_3 = []
    for k in Cap_3:
        x= k*1e6/(np.pi * (pow(r3,2)) * 1e-8)
        cap_standard_3.append(round(x,3))
    cap_standard_4 = []
    for k in Cap_4:
        x= k*1e6/(np.pi * (pow(r4,2)) * 1e-8)
        cap_standard_4.append(round(x,3))    

    plt.plot(Volt_1[0:round((len(Volt_1)/2))+1], cap_standard_1[0:round((len(Volt_1)/2))+1], color='blue', linestyle='-',marker = matplotlib.markers.CARETRIGHT, markersize = 10,label ='%s KHz' % csv[0].split('\\')[-1][:-18].split('_')[-1].split(' ')[2])
    plt.plot(Volt_1[round((len(Volt_1)/2)):-1], cap_standard_1[round((len(Volt_1)/2)):-1], color='blue', linestyle='-',marker = matplotlib.markers.CARETLEFT, markersize = 10)
    

    plt.plot(Volt_3[0:round((len(Volt_1)/2))+1], cap_standard_3[0:round((len(Volt_1)/2))+1], color='green', linestyle='-',marker = matplotlib.markers.CARETRIGHT, markersize = 10,label = '%s KHz' % csv[2].split('\\')[-1][:-18].split('_')[-1].split(' ')[2])
    plt.plot(Volt_3[round((len(Volt_1)/2)):-1], cap_standard_3[round((len(Volt_1)/2)):-1], color='green', linestyle='-',marker = matplotlib.markers.CARETLEFT, markersize = 10)

    plt.plot(Volt_4[0:round((len(Volt_1)/2))+1], cap_standard_4[0:round((len(Volt_1)/2))+1], color='red', linestyle='-',marker = matplotlib.markers.CARETRIGHT, markersize = 10,label = '%s KHz' % csv[3].split('\\')[-1][:-18].split('_')[-1].split(' ')[2])
    plt.plot(Volt_4[round((len(Volt_1)/2)):-1], cap_standard_4[round((len(Volt_1)/2)):-1], color='red', linestyle='-',marker = matplotlib.markers.CARETLEFT, markersize = 10)

    plt.plot(Volt_2[0:round((len(Volt_1)/2))+1], cap_standard_2[0:round((len(Volt_1)/2))+1], color='orange', linestyle='-',marker = matplotlib.markers.CARETRIGHT, markersize = 10,label = '%s MHz' % csv[1].split('\\')[-1][:-18].split('_')[-1].split(' ')[2])
    plt.plot(Volt_2[round((len(Volt_1)/2)):-1], cap_standard_2[round((len(Volt_1)/2)):-1], color='orange', linestyle='-',marker = matplotlib.markers.CARETLEFT, markersize = 10)
        # ax1.set_xlabel("Voltage [V]", fontsize=20)
        # lines = line1 + line2

        # labels = [l.get_label() for l in lines]
        # ax1.legend(lines, labels, loc='upper right')
        # plt.tight_layout()
            

    # plt.plot(voltage[0], capacitance[0], label='%s' % csv[0].split('\\')[-1][:-18].split('_')[-1])
    # plt.plot(voltage[1], capacitance[1], label='%s' % csv[1].split('\\')[-1][:-18].split('_')[-1])
    # plt.plot(voltage[2], capacitance[2], label='%s' % csv[2].split('\\')[-1][:-18].split('_')[-1])
    # # plt.plot(voltage[3], capacitance[3], label='%s' % csv[3].split('\\')[-1][:-18])
    # # plt.plot(voltage[4], capacitance[4], label='%s' % csv[4].split('\\')[-1][:-18])

    plt.ylim(0,1)

    plt.title('%s'%C)
    plt.xlabel('Voltage [V]', labelpad=10)
    plt.ylabel('Capacitance [$\mu F/cm^2$]', labelpad=10)
    plt.legend()
    plt.grid(True)

    if A == 'T':
        plt.savefig('.\\res\\%s.png' % ('standard '+ C +' 500 ALL'))

    if B == 'T':
        plt.show(block=False)
        plt.pause(1)
        plt.close()



''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''




def standardplot(A,B,C):                                            # pad 사이즈 규격화 (주파수 고정하고 패드사이즈 규격화 시킨 후 비교) - pad 사이즈 개수 만큼 데이터 추가 
    file_path = '.\dat\CSV\*%s*.csv'%C                                      
    csv = []
    for filename in glob.glob(file_path, recursive=True):
        csv.append(filename)
    
    csv_tqdm = tqdm(csv)
    
    voltage = []
    capacitance = []
    conductunce = []
    for i in csv_tqdm:
        # filename = i.split('\\')[3]
        filename_1 = i.split('\\')[-1][:-4]
        csv_tqdm.set_description(f'Processing {filename_1}')
    

        Data = pd.read_csv(".\dat\CSV\%s" % filename_1 + '.csv')

        # Value = "DataValue"

        # find_row = Data.loc[(Data['Value'] == Value)]
        # find_row = find_row.iloc[:,1:3]
        # print(find_row)

        # vol = csv['Voltage (V)']
        # cap = csv['Cp']
        # con = csv['G']


        find_columns_v = Data.loc[:,'Voltage (V)']
        find_columns_c = Data.loc[:,'Cp(F)']
        find_columns_g = Data.loc[:,' G(s)']
        # print(find_columns_v)
        # print(cap.values.tolist())
        Voltage = list(map(float, find_columns_v.tolist()))
        voltage.append(Voltage)
        Capacitance = list(map(float, find_columns_c.tolist()))
        capacitance.append(Capacitance)
        Conductunce = list(map(float, find_columns_g.tolist()))                # 배열을 리스트로 바꾸기
        conductunce.append(Conductunce)
        


    plt.rcParams['font.size'] = 27
    # plt.rc('font', size=15)
    plt.figure(figsize=(12, 9))

    # line1 = ax1.plot(VBias[0:round((len(VBias)/2))+1], CMeas[0:round((len(VBias)/2))+1], color='blue', linestyle='-',marker = matplotlib.markers.CARETRIGHT, markersize = 10,label = 'Capacitance')
    # ax1.plot(VBias[round((len(VBias)/2)):-1], CMeas[round((len(VBias)/2)):-1], color='blue', linestyle='-',marker = matplotlib.markers.CARETLEFT, markersize = 10)
    # ax1.set_ylabel('Capacitance ', color='blue', fontsize=20, labelpad = 10)
    # ax1.tick_params('y', colors='blue')

    Volt_1 = voltage[0]
    Volt_2 = voltage[1]
    Volt_3 = voltage[2]
    # Volt_4 = voltage[3]

    Cap_1 = capacitance[0]
    Cap_2 = capacitance[1]
    Cap_3 = capacitance[2]
    # Cap_4 = capacitance[3]

    # r1 = float(r)
    # r2 = float(r)
    # r3 = float(r)



    # r1= float(csv[0].split('\\')[-1][:-18].split('_')[-1].split(' ')[1])
    # r2= float(csv[1].split('\\')[-1][:-18].split('_')[-1].split(' ')[1])
    # r3= float(csv[2].split('\\')[-1][:-18].split('_')[-1].split(' ')[1])
    # r4= float(csv[3].split('\\')[-1][:-18].split('_')[3])

    r1= float(csv[0].split('\\')[-1][:-18].split('_')[1])                        # 반지름 바꾸기 이거 쓰기
    r2= float(csv[1].split('\\')[-1][:-18].split('_')[1])
    r3= float(csv[2].split('\\')[-1][:-18].split('_')[1])

    cap_standard_1 = []
    for i in Cap_1:
        x= i*1e6/(np.pi * (pow(r1,2)) * 1e-8)
        cap_standard_1.append(round(x,3))
    cap_standard_2 = []
    for z in Cap_2:
        x= z*1e6/(np.pi * (pow(r2,2)) * 1e-8)
        cap_standard_2.append(round(x,3))
    cap_standard_3 = []
    for k in Cap_3:
        x= k*1e6/(np.pi * (pow(r3,2)) * 1e-8)
        cap_standard_3.append(round(x,3))
    # cap_standard_4 = []
    # for k in Cap_4:
    #     x= k*1e6/(np.pi * (pow(r4,2)) * 1e-8)
    #     cap_standard_4.append(round(x,3))    
    # print(Cap_1)
    # print(cap_standard_3, cap_standard_2, cap_standard_1)
    
    plt.plot(Volt_3[0:round((len(Volt_1)/2))+1], cap_standard_3[0:round((len(Volt_1)/2))+1], color='green', linestyle='-',marker = matplotlib.markers.CARETRIGHT, markersize = 10,label = '%s um' % csv[2].split('\\')[-1][:-18].split('_')[1])
    plt.plot(Volt_3[round((len(Volt_1)/2)):-1], cap_standard_3[round((len(Volt_1)/2)):-1], color='green', linestyle='-',marker = matplotlib.markers.CARETLEFT, markersize = 10)

    plt.plot(Volt_1[0:round((len(Volt_1)/2))+1], cap_standard_1[0:round((len(Volt_1)/2))+1], color='blue', linestyle='-',marker = matplotlib.markers.CARETRIGHT, markersize = 10,label ='%s um' % csv[0].split('\\')[-1][:-18].split('_')[1])
    plt.plot(Volt_1[round((len(Volt_1)/2)):-1], cap_standard_1[round((len(Volt_1)/2)):-1], color='blue', linestyle='-',marker = matplotlib.markers.CARETLEFT, markersize = 10)
    
    plt.plot(Volt_2[0:round((len(Volt_1)/2))+1], cap_standard_2[0:round((len(Volt_1)/2))+1], color='orange', linestyle='-',marker = matplotlib.markers.CARETRIGHT, markersize = 10,label = '%s um' % csv[1].split('\\')[-1][:-18].split('_')[1])
    plt.plot(Volt_2[round((len(Volt_1)/2)):-1], cap_standard_2[round((len(Volt_1)/2)):-1], color='orange', linestyle='-',marker = matplotlib.markers.CARETLEFT, markersize = 10)

    

    # plt.plot(Volt_4[0:round((len(Volt_1)/2))+1], cap_standard_4[0:round((len(Volt_1)/2))+1], color='red', linestyle='-',marker = matplotlib.markers.CARETRIGHT, markersize = 10,label = '%s um' % csv[3].split('\\')[-1][:-18].split('_')[1])
    # plt.plot(Volt_4[round((len(Volt_1)/2)):-1], cap_standard_4[round((len(Volt_1)/2)):-1], color='red', linestyle='-',marker = matplotlib.markers.CARETLEFT, markersize = 10)
        # ax1.set_xlabel("Voltage [V]", fontsize=20)
        # lines = line1 + line2

        # labels = [l.get_label() for l in lines]
        # ax1.legend(lines, labels, loc='upper right')
        # plt.tight_layout()
            

    # plt.plot(voltage[0], capacitance[0], label='%s' % csv[0].split('\\')[-1][:-18].split('_')[-1])
    # plt.plot(voltage[1], capacitance[1], label='%s' % csv[1].split('\\')[-1][:-18].split('_')[-1])
    # plt.plot(voltage[2], capacitance[2], label='%s' % csv[2].split('\\')[-1][:-18].split('_')[-1])
    # # plt.plot(voltage[3], capacitance[3], label='%s' % csv[3].split('\\')[-1][:-18])
    # # plt.plot(voltage[4], capacitance[4], label='%s' % csv[4].split('\\')[-1][:-18])

    plt.ylim()

    plt.title('%s Hz'%C)
    plt.xlabel('Voltage [V]', labelpad=10)
    plt.ylabel('Capacitance [$\mu F/cm^2$]', labelpad=10)
    plt.legend()
    plt.grid(True)

    if A == 'T':
        plt.savefig('.\\res\\standard %s.png' % ((csv[0].split('\\')[-1][:-18].split('_')[0]) + ' ' + C +' ALL'))

    if B == 'T':
        plt.show(block=False)
        plt.pause(1)
        plt.close()


#standardplot('T','T','100 k')
# standardplot('T','T','AL2O3_%d_1-1_1 M'%(2*50),50)

# print('AL2O3_%d_1_1 M'%(2*50))

# title = ['HfO2_100_1','HfO2_150_1','HfO2_200_1','HfO2_RTA_200_1','HfO2_RTA_150_1','HfO2_RTA_100_1']
# for i in title:
#     allplot('T','T',i)
    
def Rstandardplot(A,B,C):
    file_path = '.\dat\CSV\*%s*.csv'%C                                      
    csv = []
    for filename in glob.glob(file_path, recursive=True):
        csv.append(filename)
    
    csv_tqdm = tqdm(csv)
    
    voltage = []
    capacitance = []
    conductunce = []
    for i in csv_tqdm:
        # filename = i.split('\\')[3]
        filename_1 = i.split('\\')[-1][:-4]
        csv_tqdm.set_description(f'Processing {filename_1}')
    

        Data = pd.read_csv(".\dat\CSV\%s" % filename_1 + '.csv')

        # Value = "DataValue"

        # find_row = Data.loc[(Data['Value'] == Value)]
        # find_row = find_row.iloc[:,1:3]
        # print(find_row)

        # vol = csv['Voltage (V)']
        # cap = csv['Cp']
        # con = csv['G']


        find_columns_v = Data.loc[:,'Voltage (V)']
        find_columns_c = Data.loc[:,'Cp(F)']
        find_columns_g = Data.loc[:,' G(s)']
        # print(find_columns_v)
        # print(cap.values.tolist())
        Voltage = list(map(float, find_columns_v.tolist()))
        voltage.append(Voltage)
        Capacitance = list(map(float, find_columns_c.tolist()))
        capacitance.append(Capacitance)
        Conductunce = list(map(float, find_columns_g.tolist()))                # 배열을 리스트로 바꾸기
        conductunce.append(Conductunce)
        


    plt.rcParams['font.size'] = 20
    # plt.rc('font', size=15)
    plt.figure(figsize=(12, 9))

    # line1 = ax1.plot(VBias[0:round((len(VBias)/2))+1], CMeas[0:round((len(VBias)/2))+1], color='blue', linestyle='-',marker = matplotlib.markers.CARETRIGHT, markersize = 10,label = 'Capacitance')
    # ax1.plot(VBias[round((len(VBias)/2)):-1], CMeas[round((len(VBias)/2)):-1], color='blue', linestyle='-',marker = matplotlib.markers.CARETLEFT, markersize = 10)
    # ax1.set_ylabel('Capacitance ', color='blue', fontsize=20, labelpad = 10)
    # ax1.tick_params('y', colors='blue')

    Volt_1 = voltage[0]
    Volt_2 = voltage[1]
    Volt_3 = voltage[2]
    Cap_1 = capacitance[0]
    Cap_2 = capacitance[1]
    Cap_3 = capacitance[2]

    r1= float(csv[0].split('\\')[-1][:-18].split('_')[3])
    r2= float(csv[1].split('\\')[-1][:-18].split('_')[3])
    r3= float(csv[2].split('\\')[-1][:-18].split('_')[3])


    cap_standard_1 = []
    for i in Cap_1:
        x= i*1e6/(np.pi * (pow(r1/2,2)) * 1e-8)
        cap_standard_1.append(round(x,3))
    cap_standard_2 = []
    for z in Cap_2:
        x= z*1e6/(np.pi * (pow(r2/2,2)) * 1e-8)
        cap_standard_2.append(round(x,3))
    cap_standard_3 = []
    for k in Cap_3:
        x= k*1e6/(np.pi * (pow(r3/2,2)) * 1e-8)
        cap_standard_3.append(round(x,3))
    # print(Cap_1)
    # print(cap_standard_3, cap_standard_2, cap_standard_1)
    

    plt.plot(Volt_1[0:round((len(Volt_1)/2))+1], cap_standard_1[0:round((len(Volt_1)/2))+1], color='blue', linestyle='-',marker = matplotlib.markers.CARETRIGHT, markersize = 10,label ='%s um' % csv[0].split('\\')[-1][:-18].split('_')[1])
    plt.plot(Volt_1[round((len(Volt_1)/2)):-1], cap_standard_1[round((len(Volt_1)/2)):-1], color='blue', linestyle='-',marker = matplotlib.markers.CARETLEFT, markersize = 10)
    
    plt.plot(Volt_2[0:round((len(Volt_1)/2))+1], cap_standard_2[0:round((len(Volt_1)/2))+1], color='orange', linestyle='-',marker = matplotlib.markers.CARETRIGHT, markersize = 10,label = '%s um' % csv[1].split('\\')[-1][:-18].split('_')[1])
    plt.plot(Volt_2[round((len(Volt_1)/2)):-1], cap_standard_2[round((len(Volt_1)/2)):-1], color='orange', linestyle='-',marker = matplotlib.markers.CARETLEFT, markersize = 10)

    plt.plot(Volt_3[0:round((len(Volt_1)/2))+1], cap_standard_3[0:round((len(Volt_1)/2))+1], color='green', linestyle='-',marker = matplotlib.markers.CARETRIGHT, markersize = 10,label = '%s um' % csv[2].split('\\')[-1][:-18].split('_')[1])
    plt.plot(Volt_3[round((len(Volt_1)/2)):-1], cap_standard_3[round((len(Volt_1)/2)):-1], color='green', linestyle='-',marker = matplotlib.markers.CARETLEFT, markersize = 10)
        # ax1.set_xlabel("Voltage [V]", fontsize=20)
        # lines = line1 + line2

        # labels = [l.get_label() for l in lines]
        # ax1.legend(lines, labels, loc='upper right')
        # plt.tight_layout()
            

    # plt.plot(voltage[0], capacitance[0], label='%s' % csv[0].split('\\')[-1][:-18].split('_')[-1])
    # plt.plot(voltage[1], capacitance[1], label='%s' % csv[1].split('\\')[-1][:-18].split('_')[-1])
    # plt.plot(voltage[2], capacitance[2], label='%s' % csv[2].split('\\')[-1][:-18].split('_')[-1])
    # # plt.plot(voltage[3], capacitance[3], label='%s' % csv[3].split('\\')[-1][:-18])
    # # plt.plot(voltage[4], capacitance[4], label='%s' % csv[4].split('\\')[-1][:-18])

    plt.ylim(0,1.2)

    plt.title('RTA %s Hz'%C)
    plt.xlabel('Voltage [V]', labelpad=10)
    plt.ylabel('Capacitance [uF/cm^2]', labelpad=10)
    plt.legend()
    plt.grid(True)

    if A == 'T':
        plt.savefig('.\\res\\standard RTA %s.png' % ((csv[0].split('\\')[-1][:-18].split('_')[0]) + ' ' + C +' ALL'))

    if B == 'T':
        plt.show(block=False)
        plt.pause(1)
        plt.close()

# title = ['Al_HfO2_Si_30 75','Al_HfO2_Si_30 100','Al_HfO2_Si_30 200'] 
# for i in title:
#     standardplot_2('T','T',i)

title = ['Al_ZrO2_O2_2_Si','Al_ZrO2_O2_4_Si','Al_ZrO2_O2_6_Si' ] 
for i in title:
    # allplot('T','T',i)
    standardplot_2('T','T',i)

# title_stan = [ '1 k', '10 k','100 k','1 M']
# for i in title_stan:
#     standardplot('T','T',i )

# title_stan = [ '10 k','100 k','1 M']
# for i in title_stan:
#     standardplot('T','T',i )
