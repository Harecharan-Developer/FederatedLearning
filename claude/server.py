import socket
import pickle
import threading
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, LSTM, Bidirectional
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.spinner import Spinner
import json  # Add this at the top of your file

class ServerApp(App):
    def __init__(self):
        super().__init__()
        self.clients_data = []
        self.model = None
        self.architecture = None
        self.soc = None
        self.listen_thread = None
    
    def create_model(self, architecture, input_shape):
        model = Sequential()
        if architecture == 'GRU/BiLSTM':
            model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(None, input_shape)))
            model.add(GRU(32))
            model.add(Dense(1))
        elif architecture == 'MLP':
            model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    def create_socket(self, instance):
        try:
            self.soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            self.status_label.text = "Socket Created"
            self.create_socket_btn.disabled = True
            self.bind_btn.disabled = False
            self.close_socket_btn.disabled = False
        except Exception as e:
            self.status_label.text = f"Socket Creation Error: {str(e)}"

    def bind_socket(self, instance):
        try:
            self.soc.bind((self.server_ip.text, int(self.server_port.text)))
            self.status_label.text = "Socket Bound"
            self.bind_btn.disabled = True
            self.listen_btn.disabled = False
            self.architecture_spinner.disabled = False
        except Exception as e:
            self.status_label.text = f"Binding Error: {str(e)}"

    def listen_accept(self, instance):
        if not self.architecture:
            self.status_label.text = "Please select model architecture first"
            return
        
        try:
            self.soc.listen(5)
            self.status_label.text = "Listening for connections"
            self.listen_btn.disabled = True
            
            self.listen_thread = ListenThread(self)
            self.listen_thread.start()
        except Exception as e:
            self.status_label.text = f"Listen Error: {str(e)}"

    def select_architecture(self, instance, text):
        self.architecture = text
        self.status_label.text = f"Selected architecture: {text}"

    def perform_aggregation(self, method='normal'):
        if len(self.clients_data) < 2:
            return None
        
        weights_list = [data['weights'] for data in self.clients_data]
        
        if method == 'normal':
            avg_weights = [np.mean(weights, axis=0) for weights in zip(*weights_list)]
        elif method == 'weighted':
            total_samples = sum(data['num_samples'] for data in self.clients_data)
            weights = [data['num_samples']/total_samples for data in self.clients_data]
            avg_weights = [sum(w * layer for w, layer in zip(weights, layer_weights)) 
                         for layer_weights in zip(*weights_list)]
        else:  # performance based
            total_acc = sum(data['accuracy'] for data in self.clients_data)
            weights = [data['accuracy']/total_acc for data in self.clients_data]
            avg_weights = [sum(w * layer for w, layer in zip(weights, layer_weights)) 
                         for layer_weights in zip(*weights_list)]
        
        return avg_weights

    def close_socket(self, instance):
        if self.soc:
            self.soc.close()
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join()
        self.status_label.text = 'Socket Closed'
        self.create_socket_btn.disabled = False
        self.close_socket_btn.disabled = True
        self.bind_btn.disabled = True
        self.listen_btn.disabled = True

    def build(self):
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Status label
        self.status_label = Label(text="Server Status")
        
        # Socket controls
        self.create_socket_btn = Button(text="Create Socket")
        self.create_socket_btn.bind(on_press=self.create_socket)
        
        # IP and Port inputs
        socket_info = BoxLayout(orientation='horizontal', spacing=5)
        self.server_ip = TextInput(text='localhost', hint_text='Server IP')
        self.server_port = TextInput(text='10000', hint_text='Port')
        socket_info.add_widget(self.server_ip)
        socket_info.add_widget(self.server_port)
        
        # Bind button
        self.bind_btn = Button(text="Bind Socket", disabled=True)
        self.bind_btn.bind(on_press=self.bind_socket)
        
        # Architecture selection
        self.architecture_spinner = Spinner(
            text='Select Architecture',
            values=('GRU/BiLSTM', 'MLP'),
            disabled=True
        )
        self.architecture_spinner.bind(text=self.select_architecture)
        
        # Listen button
        self.listen_btn = Button(text="Listen", disabled=True)
        self.listen_btn.bind(on_press=self.listen_accept)
        
        # Close socket button
        self.close_socket_btn = Button(text="Close Socket", disabled=True)
        self.close_socket_btn.bind(on_press=self.close_socket)
        
        # Add all widgets to layout
        main_layout.add_widget(self.status_label)
        main_layout.add_widget(self.create_socket_btn)
        main_layout.add_widget(socket_info)
        main_layout.add_widget(self.bind_btn)
        main_layout.add_widget(self.architecture_spinner)
        main_layout.add_widget(self.listen_btn)
        main_layout.add_widget(self.close_socket_btn)
        
        return main_layout

class ListenThread(threading.Thread):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.daemon = True
        
    def run(self):
        while True:
            try:
                conn, addr = self.app.soc.accept()
                client_thread = ClientThread(conn, addr, self.app)
                client_thread.start()
                self.app.status_label.text = f"New client connected: {addr}"
            except Exception as e:
                self.app.status_label.text = f"Connection error: {str(e)}"
                break

class ClientThread(threading.Thread):
    def __init__(self, conn, addr, app):
        super().__init__()
        self.conn = conn
        self.addr = addr
        self.app = app
        self.daemon = True
        
    def run(self):
        try:
            # Send model architecture, defaulting to None if no weights are ready
            data = {
                'architecture': self.app.architecture,
                'aggregated_weights': self.app.perform_aggregation() or []
            }
            self.conn.send(json.dumps(data).encode())
            
            # Attempt to receive data from the client and add it to the clients_data list
            received_data = self.conn.recv(4096).decode()  # Decode the bytes to string
            client_data = json.loads(received_data) if received_data else {}

            if isinstance(client_data, dict):
                self.app.clients_data.append(client_data)
            
            # Perform aggregation if there are enough clients
            if len(self.app.clients_data) >= 2:
                aggregated_weights = self.app.perform_aggregation('normal')
                self.conn.send(json.dumps({'weights': aggregated_weights}).encode())
            
        except Exception as e:
            self.app.status_label.text = f"Error with client {self.addr}: {str(e)}"
        finally:
            self.conn.close()


if __name__ == '__main__':
    ServerApp().run()