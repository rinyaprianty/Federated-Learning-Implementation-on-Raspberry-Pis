import socket
import pickle
import copy

from time import sleep
from _thread import *
from sys import getsizeof

from models.Net import Net
from utils.options import args_parser

global payloads

args = args_parser()
soc = socket.socket()

host = '127.0.0.1'
port = 10000
clients = list()
print("Socket is created.")
try:
    soc.bind((host, port))
    print("Socket is bound to an address & port number.")
except socket.error as e:
    print(str(e))

print("Listening for incoming connection ...")
soc.listen()
    
def sendFedAvg(connection, weight):
    send_this = {
        "desc" : "fed_avg",
        "payload" : weight
    }
    msg = pickle.dumps(send_this)
    # print(getsizeof(msg))
    # round_cost += getsizeof(msg)
    connection.sendall(msg)
    print("Sent fed_avg to client")
    return None

def updateLocalWeight(connection, tflite_model):
    send_this = {
        "desc" : "model",
        "payload" : tflite_model
    }
    msg = pickle.dumps(send_this)

    connection.sendall(msg)
    print("Sent model to client")
    received_data = []
    while True:
        part = connection.recv(4092)
        received_data.append(part)
        try:
            msg = b''.join(received_data)
            data = pickle.loads(b''.join(received_data))
            # print(getsizeof(msg))
            break
        except:
            pass
    #data = pickle.loads(received_data)
    print("Received updated model from client")
    payloads.append(data)
    return None

net = Net(args=args)
send_this = {
    "desc" : "model",
    "payload" : net.tflite_model
} 
msg = pickle.dumps(send_this)
print('Waiting for {} clients...'.format(args.num_users))
while True:
    connection, address = soc.accept()
    clients.append({
        "connection": connection,
        "address": address
    })
    print('Connected to: ' + address[0] + ':' + str(address[1]))
    print('Thread Number ' + str(len(clients)))
    if(len(clients) >= args.num_users):
        print("Clients ready")
    else:
        continue

    sleep(3)
    costs = list()
    w_i = list()
    for i in range(args.epochs):
        payloads = list()
        print("")
        print("Global epoch : {}/{}".format((i+1), args.epochs))
        for client in clients:
            try:
                start_new_thread(updateLocalWeight, (client["connection"], net.tflite_model))
            except Exception as e:
                print("Error : {}".format(e))
                print("disconnected from " + str(client["address"][0]) + ':' + str(client["address"][1]))
                #clients.remove(client)

        client_weights = list()
        while True:
            if len(payloads) == (args.num_users):
                round_cost = 0
                for n in range(len(payloads)):
                    m = payloads[n]
                    # print(m)
                    # exit()
                    interpreter, all_client_details = net.loadTFLiteModel(m)
                    weights = net.getTFLiteWeight(interpreter, all_client_details)
                    client_weights.append(weights)
                    round_cost += getsizeof(payloads[n])

                costs.append(round_cost / 1000000)
                print("Calculating average weight...")
                fed_avg, interpreter = net.Fed_Avg(client_weights, payloads[n])
                print("Calculating accuracy...")
                accuracy = net.test_tflite_model(interpreter)
                print("Testing Accuracy : {:.3f}%".format(accuracy * 100))
                break

    print("")
    print("Accuracy on {} epochs : {:.3f}%".format(args.epochs, accuracy * 100))
    print("Costs each round in MB : ", costs)
    avg_cost = sum(costs) / len(costs)
    print("Average cost : {:.3f} MB".format(avg_cost))
    print("Total cost : {:.3f} MB".format(sum(costs)))
    print("")
    for client in clients:
        sendFedAvg(client["connection"], fed_avg)
    print("Process Completed")
    break

soc.close()
print("Socket is closed.")
