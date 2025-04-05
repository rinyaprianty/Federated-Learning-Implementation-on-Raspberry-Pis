import socket
import pickle
import select
import copy

from time import sleep
from sys import getsizeof

from models.Net import Net
from utils.options import args_parser

args = args_parser()
net = Net(args=args)

soc = socket.socket()
host = '127.0.0.1'
port = 10000
print("Socket is created.")

soc.connect((host, port))
print("Connected to the server.")
i = 0
# while True:
while True:
    received_data = []
    while True:
        part = soc.recv(4092)
        # print(len(part))
        received_data.append(part)
        #print(len(part))
        try:
            msg = b''.join(received_data)
            data = pickle.loads(msg)
            # print(getsizeof(msg))
            break
        except Exception as e:
            # print(e)
            pass
        #if len(part) < 4092:
        #    break
    #print(received_data)
    #print(received_data.decode('utf-8'))

    #data = pickle.loads(b''.join(received_data))
    desc = data['desc']
    payload = data['payload']
    # print(desc)
    # print(payload)
    if(desc == 'model'):
        print("\nModel recieved")
        print("Training model, please wait...")
        m = payload



        interpreter, all_tensor_details = net.loadTFLiteModel(m)
        accuracy = net.train_tflite_model(interpreter)
        print("Training Accuracy : {:.3f}%\n".format(accuracy * 100))
        
        # msg = pickle.dumps(net.tflite_file)
        msg = pickle.dumps(net.tflite_model)
        soc.sendall(msg)
        print("Sent updated model to the server")

    elif(desc == 'fed_avg'):
        print("\nReceived average weight from server ")
        # print(data['payload'])
        print("Process Completed")
        break

    else:
        print("Message unrecognized")
        break

soc.close()
print("Socket is closed.")