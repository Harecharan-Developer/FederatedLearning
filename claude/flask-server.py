from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, LSTM, Bidirectional, Reshape
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.spinner import Spinner
import threading

app = Flask(__name__)

class ServerApp(App):
    def __init__(self):
        super().__init__()
        self.clients_data = []
        self.model = None
        self.architecture = None
        self.flask_thread = None
        self.input_shape = 10  # Number of features
        
    def create_model(self, architecture):
        model = Sequential()
        if architecture == 'GRU/BiLSTM':
            # Add reshape layer to convert 2D input to 3D
            model.add(Reshape((1, self.input_shape), input_shape=(self.input_shape,)))
            model.add(Bidirectional(LSTM(64, return_sequences=True)))
            model.add(GRU(32))
            model.add(Dense(1))
        elif architecture == 'MLP':
            model.add(Dense(64, activation='relu', input_shape=(self.input_shape,)))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    def start_server(self, instance):
        try:
            if not self.architecture:
                self.status_label.text = "Please select model architecture first"
                return
            
            # Create initial model
            self.model = self.create_model(self.architecture)
            
            self.flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=int(self.server_port.text)))
            self.flask_thread.daemon = True
            self.flask_thread.start()
            
            self.status_label.text = f"Server running on port {self.server_port.text}"
            self.start_server_btn.disabled = True
            
        except Exception as e:
            self.status_label.text = f"Server Error: {str(e)}"

    def select_architecture(self, instance, text):
        self.architecture = text
        self.status_label.text = f"Selected architecture: {text}"

    def perform_aggregation(self, method='normal'):
        if len(self.clients_data) < 2:
            return None
        
        # Convert received lists back to numpy arrays
        weights_list = [[np.array(w) for w in data['weights']] for data in self.clients_data]
        
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
        
        # Convert numpy arrays to lists for JSON serialization
        return [w.tolist() for w in avg_weights]

    def build(self):
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Status label
        self.status_label = Label(text="Server Status")
        
        # Port input
        server_info = BoxLayout(orientation='horizontal', spacing=5)
        self.server_port = TextInput(text='5000', hint_text='Port')
        server_info.add_widget(self.server_port)
        
        # Architecture selection
        self.architecture_spinner = Spinner(
            text='Select Architecture',
            values=('GRU/BiLSTM', 'MLP')
        )
        self.architecture_spinner.bind(text=self.select_architecture)
        
        # Start server button
        self.start_server_btn = Button(text="Start Server")
        self.start_server_btn.bind(on_press=self.start_server)
        
        # Add all widgets to layout
        main_layout.add_widget(self.status_label)
        main_layout.add_widget(server_info)
        main_layout.add_widget(self.architecture_spinner)
        main_layout.add_widget(self.start_server_btn)
        
        return main_layout

# Create global server instance
server_app = ServerApp()

@app.route('/get_model', methods=['GET'])
def get_model():
    model_info = {
        'architecture': server_app.architecture,
        'input_shape': server_app.input_shape
    }
    
    aggregated_weights = server_app.perform_aggregation()
    if aggregated_weights is not None:
        model_info['aggregated_weights'] = aggregated_weights
        
    return jsonify(model_info)

@app.route('/update_model', methods=['POST'])
def update_model():
    client_data = request.json
    if isinstance(client_data, dict):
        server_app.clients_data.append(client_data)
        
    if len(server_app.clients_data) >= 2:
        aggregated_weights = server_app.perform_aggregation('normal')
        return jsonify({'weights': aggregated_weights})
    
    return jsonify({'message': 'Waiting for more clients'})

if __name__ == '__main__':
    server_app.run()
