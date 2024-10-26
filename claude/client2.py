
# Client.py
import socket
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
import kivy
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout

class ClientApp(App):
    def __init__(self):
        super().__init__()
        self.model = None
        
    def preprocess_data(self, data_path):
        df = pd.read_csv(data_path)
        df = df.dropna()
        
        # Calculate AQI
        df['AQI'] = (0.4 * df['PM2.5']) + (0.3 * df['PM10']) +                     (0.2 * df['NO2']) + (0.1 * df['SO2'])
        
        # Split into features and target
        X = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'SR', 'RH', 'WS', 'WD']].values
        y = df['AQI'].values
        
        # Split into training and federated averaging sets (80:20)
        X_train, X_fed_avg, y_train, y_fed_avg = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_fed_avg, y_train, y_fed_avg

    def create_socket(self, *args):
        self.soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.label.text = "Socket Created"
        self.create_socket_btn.disabled = True
        self.connect_btn.disabled = False
        
    def connect_to_server(self, *args):
        try:
            self.soc.connect((self.server_ip.text, int(self.server_port.text)))
            self.label.text = "Connected to server"
            self.connect_btn.disabled = True
            self.train_btn.disabled = False
            
            # Receive model architecture from server
            server_data = pickle.loads(self.soc.recv(4096))
            self.architecture = server_data['architecture']
            
            if server_data['aggregated_weights']:
                self.model.set_weights(server_data['aggregated_weights'])
                
        except Exception as e:
            self.label.text = f"Connection error: {str(e)}"
            
    def train_model(self, *args):
        try:
            # Preprocess data
            X_train, X_fed_avg, y_train, y_fed_avg = self.preprocess_data(
                r'C:\Users\DELL\Downloads\FederatedLearning\claude\processed_aqm_cbedata.csv'
            )
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=10,
                validation_split=0.2,
                verbose=1
            )
            
            # Evaluate model
            loss, mae = self.model.evaluate(X_train, y_train)
            accuracy = 1 - mae  # Converting MAE to accuracy-like metric
            
            # Send results to server
            client_data = {
                'weights': self.model.get_weights(),
                'num_samples': len(X_train),
                'accuracy': accuracy
            }
            self.soc.send(pickle.dumps(client_data))
            
            # Receive aggregated weights
            server_response = pickle.loads(self.soc.recv(4096))
            if 'weights' in server_response:
                self.model.set_weights(server_response['weights'])
                
                # Evaluate new model on federated averaging set
                new_loss, new_mae = self.model.evaluate(X_fed_avg, y_fed_avg)
                new_accuracy = 1 - new_mae
                
                self.label.text = f"Original accuracy: {accuracy:.4f}\n" + \
                                f"New accuracy: {new_accuracy:.4f}"
                
        except Exception as e:
            self.label.text = f"Training error: {str(e)}"
            
    def build(self):
        main_layout = BoxLayout(orientation='vertical')
        
        self.create_socket_btn = Button(text="Create Socket")
        self.create_socket_btn.bind(on_press=self.create_socket)
        
        socket_info = BoxLayout(orientation='horizontal')
        self.server_ip = TextInput(text='localhost', hint_text='Server IP')
        self.server_port = TextInput(text='10000', hint_text='Port')
        socket_info.add_widget(self.server_ip)
        socket_info.add_widget(self.server_port)
        
        self.connect_btn = Button(text="Connect to Server", disabled=True)
        self.connect_btn.bind(on_press=self.connect_to_server)
        
        self.train_btn = Button(text="Train Model", disabled=True)
        self.train_btn.bind(on_press=self.train_model)
        
        self.label = Label(text="Client Status")
        
        main_layout.add_widget(self.create_socket_btn)
        main_layout.add_widget(socket_info)
        main_layout.add_widget(self.connect_btn)
        main_layout.add_widget(self.train_btn)
        main_layout.add_widget(self.label)
        
        return main_layout

if __name__ == '__main__':
    ClientApp().run()
