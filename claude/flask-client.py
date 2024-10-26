import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, LSTM, Bidirectional, Reshape
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
        df['AQI'] = (0.4 * df['PM2.5']) + (0.3 * df['PM10']) + \
                    (0.2 * df['NO2']) + (0.1 * df['SO2'])
        
        # Split into features and target
        X = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'SR', 'RH', 'WS', 'WD']].values
        y = df['AQI'].values
        
        # Split into training and federated averaging sets (80:20)
        X_train, X_fed_avg, y_train, y_fed_avg = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_fed_avg, y_train, y_fed_avg

    def create_model(self, architecture, input_shape):
        model = Sequential()
        if architecture == 'GRU/BiLSTM':
            # Add reshape layer to convert 2D input to 3D
            model.add(Reshape((1, input_shape), input_shape=(input_shape,)))
            model.add(Bidirectional(LSTM(64, return_sequences=True)))
            model.add(GRU(32))
            model.add(Dense(1))
        elif architecture == 'MLP':
            model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    def connect_to_server(self, *args):
        try:
            server_url = f"http://{self.server_ip.text}:{self.server_port.text}"
            response = requests.get(f"{server_url}/get_model")
            server_data = response.json()
            
            if isinstance(server_data, dict):
                self.architecture = server_data.get('architecture', 'Unknown')
                input_shape = server_data.get('input_shape', 10)
                self.model = self.create_model(self.architecture, input_shape)
                
                if server_data.get('aggregated_weights'):
                    # Convert lists back to numpy arrays
                    weights = [np.array(w) for w in server_data['aggregated_weights']]
                    self.model.set_weights(weights)
                
                self.label.text = "Connected to server"
                self.connect_btn.disabled = True
                self.train_btn.disabled = False
            else:
                self.label.text = "Unexpected data format received"
                
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
            server_url = f"http://{self.server_ip.text}:{self.server_port.text}"
            client_data = {
                'weights': [w.tolist() for w in self.model.get_weights()],  # Convert numpy arrays to lists
                'num_samples': len(X_train),
                'accuracy': float(accuracy)  # Convert numpy float to Python float
            }
            
            response = requests.post(f"{server_url}/update_model", json=client_data)
            server_response = response.json()
            
            if 'weights' in server_response:
                # Convert lists back to numpy arrays
                weights = [np.array(w) for w in server_response['weights']]
                self.model.set_weights(weights)
                
                # Evaluate new model on federated averaging set
                new_loss, new_mae = self.model.evaluate(X_fed_avg, y_fed_avg)
                new_accuracy = 1 - new_mae
                
                self.label.text = f"Original accuracy: {accuracy:.4f}\n" + \
                                f"New accuracy: {new_accuracy:.4f}"
            else:
                self.label.text = server_response.get('message', 'Unknown response from server')
                
        except Exception as e:
            self.label.text = f"Training error: {str(e)}"
            
    def build(self):
        main_layout = BoxLayout(orientation='vertical')
        
        socket_info = BoxLayout(orientation='horizontal')
        self.server_ip = TextInput(text='localhost', hint_text='Server IP')
        self.server_port = TextInput(text='5000', hint_text='Port')
        socket_info.add_widget(self.server_ip)
        socket_info.add_widget(self.server_port)
        
        self.connect_btn = Button(text="Connect to Server")
        self.connect_btn.bind(on_press=self.connect_to_server)
        
        self.train_btn = Button(text="Train Model", disabled=True)
        self.train_btn.bind(on_press=self.train_model)
        
        self.label = Label(text="Client Status")
        
        main_layout.add_widget(socket_info)
        main_layout.add_widget(self.connect_btn)
        main_layout.add_widget(self.train_btn)
        main_layout.add_widget(self.label)
        
        return main_layout

if __name__ == '__main__':
    ClientApp().run()
