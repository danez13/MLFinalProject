import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Processed_Data/train_FD001.csv")
sensor_cols = df.columns[5:26]
max_engines = df['ID'].max()
file = ""

# get sensor line graph
def get_engine_sensor_linegraph(engine_ID:int|None|list=None,sensor_num:int|None|list = None):
    # do all sensors for all engines
    if engine_ID is None and sensor_num is None:
        for e_num in range(1,max_engines+1):
            try_create_dir(f"Data_Analysis/{file}/engine_{e_num}")
            
            df[df.ID==e_num][sensor_cols].plot(figsize=(20, 8),legend=True)
            plt.savefig(f"Data_Analysis/{file}/engine_{e_num}/engine_{e_num}_graph.png")
            plt.close()

    # generate graph for all sensors for specified engines
    elif (type(engine_ID) is int or type(engine_ID) is list)and sensor_num is None:
        # single engine
        if type(engine_ID) is int:
            validate_engine_ID(engine_ID)

            try_create_dir(f"Data_Analysis/{file}/engine_{engine_ID}")

            # generate and save graphs
            df[df.ID==engine_ID][sensor_cols].plot(figsize=(20, 8),legend=True)
            plt.savefig(f"Data_Analysis/{file}/engine_{engine_ID}/engine_{engine_ID}_graph.png")
            plt.close

        # multiple engines
        elif type(engine_ID) is list:
            for i in engine_ID:
                validate_engine_ID(i)
                
                try_create_dir(f"Data_Analysis/{file}/engine_{i}")

                # generate and save graphs
                df[df.ID==i][sensor_cols].plot(figsize=(20, 8),legend=True)
                plt.savefig(f"Data_Analysis/{file}/engine_{i}/engine_{i}_graph.png")
                plt.close()
    # generate graphs for specific sensors for all engines
    elif engine_ID is None and (type(sensor_num) is int or type(sensor_num) is list):
        # single sensor
        if type(sensor_num) is int:
            sensor_num = validate_sensor(sensor_num)

            for e_num in range(1,max_engines+1):
                try_create_dir(f"Data_Analysis/{file}/engine_{e_num}")

                # generate and save graphs
                df[df.ID==e_num][sensor_cols[sensor_num]].plot(figsize=(20, 8),legend=True)
                plt.savefig(f"Data_Analysis/{file}/engine_{e_num}/sensor_{sensor_num+1}_graph.png")
                plt.close()    
        # multiple sensors
        elif type(sensor_num) is list:
            for e_num in range(1,max_engines+1):
                try_create_dir(f"Data_Analysis/{file}/engine_{e_num}")
                
                for i in sensor_num:

                    i = validate_sensor(i)

                    # generate and save graphs
                    df[df.ID==e_num][sensor_cols[i]].plot(figsize=(20, 8),legend=True)
                    plt.savefig(f"Data_Analysis/{file}/engine_{e_num}/sensor_{i+1}_graph.png")
                plt.close()

    # specifc engine and specific sensor
    elif (type(engine_ID) is int or type(engine_ID) is list) and (type(sensor_num) is int or type(sensor_num) is list):
        # single engine and single sensor
        if type(engine_ID) is int and type(sensor_num) is int:
            
            sensor_num = validate_sensor(sensor_num)
            
            validate_engine_ID(engine_ID)

            try_create_dir(f"Data_Analysis/{file}/engine_{engine_ID}")
            
            # generate and save graphs
            df[df.ID==engine_ID][sensor_cols[sensor_num]].plot(figsize=(20, 8),legend=True)
            plt.savefig(f"Data_Analysis/{file}/engine_{engine_ID}/sensor_{sensor_num+1}_graph.png")
            plt.close()

        # single engine and multiple sensors
        elif type(engine_ID) is int and type(sensor_num) is list:
            validate_engine_ID(engine_ID)

            try_create_dir(f"Data_Analysis/{file}/engine_{engine_ID}")
            
            s = ""
            for i in sensor_num:
                i = validate_sensor(i)
                
                # generate
                df[df.ID==engine_ID][sensor_cols[i]].plot(figsize=(20, 8),legend=True)
                s+=f"{i+1}_"

            # save graphs
            plt.savefig(f"Data_Analysis/{file}/engine_{engine_ID}/sensor_{s}graph.png")
            plt.close()

        # multiple engines and single sensor
        elif type(engine_ID) is list and type(sensor_num) is int:
            sensor_num = validate_sensor(sensor_num)

            for i in engine_ID:
                try_create_dir(f"Data_Analysis/{file}/engine_{i}")
                
                validate_engine_ID(i)
                
                # generate and save graphs
                df[df.ID==i][sensor_cols[sensor_num]].plot(figsize=(20, 8),legend=True)
                plt.savefig(f"Data_Analysis/{file}/engine_{i}/sensor_{sensor_num+1}_graph.png")
                plt.close()

        # multiple engines and multiple sensors
        elif type(engine_ID) is list and type(sensor_num) is list:
            for i in engine_ID:
                validate_engine_ID(i)
                
                try_create_dir(f"Data_Analysis/{file}/engine_{i}")

                s = ""
                for j in sensor_num:
                    j = validate_sensor(j)
                    
                    df[df.ID==i][sensor_cols[j]].plot(figsize=(20, 8),legend=True)
                    s += f"{j}_"
                plt.savefig(f"Data_Analysis/{file}/engine_{i}/sensor_{s}graph.png")
                plt.close()

def validate_sensor(num):
    if type(num) is not int:
        print(f" {num} not a valid engine ID. please input a valid numerical sensor ID between 1-21")
        exit()
    if num < 1 and num > 21:
        print("not a valid sensor ID. please input a valid sensor ID between 1-21")
        exit()
    return num-1

def validate_engine_ID(ID):
    # check i is an int
    if type(ID) is not int:
        print(f" {ID} not a valid engine ID. please input a valid numerical engine ID between 1-100")
        exit()
    # validate engine_ID is a valid engine ID
    if ID < 1 or ID > max_engines+1:
        print("not a valid engine ID. please input a valid engine ID between 1-100")
        exit()

def try_create_dir(path:str):
    try:
        os.mkdir(path)
    except:
        pass

def validate_dataset(data):
    data_lst = ["test_FD001","test_FD002", "test_FD003", "test_FD004", "train_FD001", "train_FD002", "train_FD003", "train_FD004"]
    if data in data_lst:
        return True
    return False
if __name__=="__main__":
    file = input("please input dataset i.e, train_FD001, train_FD002, etc... ")
    if validate_dataset(file):
        df = pd.read_csv(f"Processed_Data/{file}.csv")
        sensor_cols = df.columns[5:26]
        max_engines = df['ID'].max()
        
        try_create_dir(f"Data_Analysis/{file}")

        sensors = []        
        # number of elements as input
        n = int(input("Enter number of sensors: "))
        
        if n == 21:
            sensors = None
        elif n == 1:
            sensors = int(input("enter sensor number: "))
        else:
            # iterating till the range
            for i in range(0, n):
                ele = int(input(f"sensor {i}: "))
                # adding the element
                sensors.append(ele)
            sensors.sort()

        engines = []        
        # number of elements as input
        n = int(input("Enter number of engines: "))
        
        if n == 100:
            engines = None
        elif n == 1:
            engines = int(input("engine ID: "))
        else:
            # iterating till the range
            for i in range(0, n):
                ele = int(input(f"engine {i}: "))
                # adding the element)
                engines.append(ele)
            engines.sort()

        if sensors is None:
            print("all")
        else:
            print(sensors)
        
        if engines is None:
            print("all")
        else:
            print(engines)   

        get_engine_sensor_linegraph(engines,sensors)