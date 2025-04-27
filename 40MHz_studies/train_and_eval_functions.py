import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc


def load_and_preprocess(data_path, standard_scaler=True):

    print('Booting up...\nStarting to load data...\n')

    # Initialize empty dictionary to store the datasets. This will be a dictionary of dictionaries, one for each dataset
    datasets = {}

    # Load the data from the hdf5 files
    for file_name in os.listdir(data_path):
        if file_name.startswith('.'):
            continue

        dataset_name = file_name.split('_')[0]
        file_path = os.path.join(data_path, file_name)
        print(f'Loading {file_name}...')
        
        with h5py.File(file_path, 'r') as file:
            # by default, the shape is (n_events, 19, 4).
            # 19 = MET, 4 electrons, 4 muons, 10 jets in that order.
            # Each object has 4 features: pt, eta, phi, and 'obj_type' (1 = MET, 2 = electron, 3 = muon, 4 = jet).
            
            datasets[dataset_name] = {'data': file['Particles'][:, :, :-1]} # remove the 'obj_type' feature


    print('Beginning preprocessing...\n')

    # Split the data into training, validation, and test sets
    idxs = np.arange(len(datasets['background']['data']))
    train_idxs, _idxs = train_test_split(idxs, test_size=0.5, random_state=42)
    val_idxs, test_idxs = train_test_split(_idxs, test_size=0.5, random_state=42)

    datasets['train'] = {key: value[train_idxs] for key, value in datasets['background'].items()}
    datasets['val'] = {key: value[val_idxs] for key, value in datasets['background'].items()}
    datasets['test'] = {key: value[test_idxs] for key, value in datasets['background'].items()}
    del datasets['background']


    # flatten the data, and apply standard scaler if specified
    for tag, data_dict in datasets.items():
        num_features = np.prod(data_dict['data'].shape[-2:])
        data_dict['data'] = data_dict['data'].reshape(-1, num_features)


    # Fit and apply the standard scaler to the training data
    if standard_scaler:
        scaler = StandardScaler()
        scaler.fit(datasets['train']['data'])
        for tag, data_dict in datasets.items():
            data_dict['data'] = scaler.transform(data_dict['data'])

    print('Load and preprocessing complete!\n')

    return datasets

# Sampling layer for the VAE
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the input."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        
        # Instead of clipping, we'll use softer constraints through tf.math.softplus
        # This ensures log_var doesn't get too extreme while still allowing a wide range
        z_log_var_stable = tf.clip_by_value(z_log_var, -20.0, 20.0)
        std = tf.math.sqrt(tf.math.softplus(tf.math.exp(z_log_var_stable)) + 1e-8)
        
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + std * epsilon

        
# Create a small VAE.
def create_small_VAE(input_dim, h_dim_1, h_dim_2, latent_dim, l2_reg=0.01, dropout_rate=0):
    # Use He initialization with a smaller scale
    initializer = tf.keras.initializers.HeNormal(seed=42)
    
    # Encoder
    encoder_inputs = layers.Input(shape=(input_dim,))
    
    # Normalize inputs to help with training stability
    x = layers.BatchNormalization()(encoder_inputs)
    
    x = layers.Dense(h_dim_1, 
                    activation='relu', 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(h_dim_2, 
                    activation='relu', 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Let mean and log_var be unconstrained for better AD performance
    z_mean = layers.Dense(latent_dim, 
                         kernel_regularizer=regularizers.l2(l2_reg),
                         kernel_initializer=initializer)(x)
    
    z_log_var = layers.Dense(latent_dim,
                            kernel_regularizer=regularizers.l2(l2_reg),
                            kernel_initializer=initializer)(x)
    
    z = Sampling()([z_mean, z_log_var])
    
    encoder = Model(inputs=encoder_inputs, outputs=[z_mean, z_log_var, z])

    # Decoder
    decoder_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(h_dim_2, 
                    activation='relu', 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer=initializer)(decoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(h_dim_1, 
                    activation='relu', 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(input_dim, 
                         kernel_regularizer=regularizers.l2(l2_reg),
                         kernel_initializer=initializer)(x)

    decoder = Model(inputs=decoder_inputs, outputs=outputs)

    mean, log_var, latent = encoder(encoder_inputs)
    ae_outputs = [decoder(latent), mean, log_var]
    ae = Model(encoder_inputs, outputs=ae_outputs)

    return ae, encoder, decoder



class VAETrainer:
    def __init__(self, vae, encoder, decoder, beta=0.4, n_cycles=4, ratio=0.5):
        self.vae = vae
        self.encoder = encoder
        self.decoder = decoder
        self.max_beta = beta  # Store the maximum beta value
        self.n_cycles = n_cycles  # Number of cycles for beta annealing
        self.ratio = ratio  # Ratio of increasing beta phase in each cycle
        self.current_beta = 0.0  # Initialize current_beta to 0
        
        # Use a smaller initial learning rate for better stability
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        # Initialize history tracking
        self.history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'val_total_loss': [],
            'val_reconstruction_loss': [],
            'val_kl_loss': [],
            'beta': [],  # Track beta values
            'learning_rate': []  # Track learning rates
        }
        
        # Track best model weights
        self.best_loss = float('inf')
        self.best_weights = None

    def get_beta(self, epoch, total_epochs):
        """Compute beta value for the current epoch using cyclical annealing schedule."""
        # Calculate the period of one cycle
        cycle_length = total_epochs // self.n_cycles
        
        # Calculate current position in the cycle
        cycle_position = (epoch % cycle_length) / cycle_length
        
        # If we're in the increasing phase (determined by ratio)
        if cycle_position <= self.ratio:
            # Linear increase from 0 to max_beta
            beta = self.max_beta * (cycle_position / self.ratio)
        else:
            # Constant max_beta in the rest of the cycle
            beta = self.max_beta
            
        return beta

    def get_learning_rate(self, epoch, total_epochs):
        """Get learning rate based on current cycle."""
        cycle_length = total_epochs // self.n_cycles
        current_cycle = epoch // cycle_length
        
        # Learning rate schedule:
        # Cycle 0: 0.01
        # Cycles 1-2: 0.001
        # Cycle 3: 0.0001
        if current_cycle == 0:
            return 0.01
        elif current_cycle <= 2:
            return 0.001
        else:
            return 0.0001

    def compute_losses(self, x):
        """Compute all components of the loss"""
        # Ensure input is float32
        x = tf.cast(x, tf.float32)
        
        # Get VAE outputs
        mean, log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        
        # Debug: Check components
        if tf.math.reduce_any(tf.math.is_nan(mean)):
            print("NaN detected in mean!")
            print("mean stats:", tf.reduce_min(mean), tf.reduce_max(mean))
        if tf.math.reduce_any(tf.math.is_nan(log_var)):
            print("NaN detected in log_var!")
            print("log_var stats:", tf.reduce_min(log_var), tf.reduce_max(log_var))
        if tf.math.reduce_any(tf.math.is_nan(z)):
            print("NaN detected in sampled z!")
            print("z stats:", tf.reduce_min(z), tf.reduce_max(z))
        
        # Compute reconstruction loss (MSE)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(x - reconstruction), axis=1)
        )
        
        # Compute KL divergence with numerical stability
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + tf.clip_by_value(log_var, -20.0, 20.0) - 
                tf.clip_by_value(tf.square(mean), 0.0, 100.0) - 
                tf.clip_by_value(tf.exp(log_var), 1e-20, 1e20), 
                axis=1
            )
        )
        
        # Get current beta value from the trainer's state
        current_beta = self.current_beta
        
        # Total loss with current beta
        total_loss = reconstruction_loss + current_beta * kl_loss
        
        return total_loss, reconstruction_loss, kl_loss
    
    @tf.function
    def train_step(self, x):
        """Single training step"""
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss = self.compute_losses(x)
            
        # Compute gradients
        trainable_vars = (
            self.encoder.trainable_variables + 
            self.decoder.trainable_variables
        )
        gradients = tape.gradient(total_loss, trainable_vars)
        
        # Clip gradients by global norm instead of individual norm for better stability
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return total_loss, reconstruction_loss, kl_loss
    
    def train(self, train_dataset, val_dataset, epochs=100, batch_size=128):
        """Full training loop with cyclical beta annealing and fixed learning rate schedule"""
        # Convert datasets to float32
        train_dataset = tf.cast(train_dataset, tf.float32)
        val_dataset = tf.cast(val_dataset, tf.float32)
        
        train_ds = tf.data.Dataset.from_tensor_slices(train_dataset).batch(batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices(val_dataset).batch(batch_size)
        
        for epoch in range(epochs):
            # Update beta and learning rate for this epoch
            self.current_beta = self.get_beta(epoch, epochs)
            current_lr = self.get_learning_rate(epoch, epochs)
            self.optimizer.learning_rate.assign(current_lr)
            
            # Training
            epoch_losses = {
                'total_loss': [],
                'reconstruction_loss': [],
                'kl_loss': []
            }
            
            for batch_idx, batch in enumerate(train_ds):
                total_loss, reconstruction_loss, kl_loss = self.train_step(batch)
                
                # Check for NaN losses during training
                if tf.math.is_nan(total_loss) or tf.math.is_nan(reconstruction_loss) or tf.math.is_nan(kl_loss):
                    print(f"\nNaN detected during training at epoch {epoch + 1}, batch {batch_idx + 1}")
                    print(f"Beta: {self.current_beta:.4f}, Learning Rate: {current_lr:.6f}")
                    print(f"Losses - Total: {total_loss:.4f}, Reconstruction: {reconstruction_loss:.4f}, KL: {kl_loss:.4f}")
                    raise ValueError("NaN losses detected during training")
                
                epoch_losses['total_loss'].append(total_loss)
                epoch_losses['reconstruction_loss'].append(reconstruction_loss)
                epoch_losses['kl_loss'].append(kl_loss)
            
            # Compute epoch metrics
            train_total = tf.reduce_mean(epoch_losses['total_loss'])
            train_reconstruction = tf.reduce_mean(epoch_losses['reconstruction_loss'])
            train_kl = tf.reduce_mean(epoch_losses['kl_loss'])
            
            # Validation
            val_losses = {
                'total_loss': [],
                'reconstruction_loss': [],
                'kl_loss': []
            }
            
            for batch in val_ds:
                val_total, val_reconstruction, val_kl = self.compute_losses(batch)
                val_losses['total_loss'].append(val_total)
                val_losses['reconstruction_loss'].append(val_reconstruction)
                val_losses['kl_loss'].append(val_kl)
            
            val_total = tf.reduce_mean(val_losses['total_loss'])
            val_reconstruction = tf.reduce_mean(val_losses['reconstruction_loss'])
            val_kl = tf.reduce_mean(val_losses['kl_loss'])
            
            # Update history
            self.history['total_loss'].append(float(train_total))
            self.history['reconstruction_loss'].append(float(train_reconstruction))
            self.history['kl_loss'].append(float(train_kl))
            self.history['val_total_loss'].append(float(val_total))
            self.history['val_reconstruction_loss'].append(float(val_reconstruction))
            self.history['val_kl_loss'].append(float(val_kl))
            self.history['beta'].append(float(self.current_beta))
            self.history['learning_rate'].append(float(current_lr))
            
            # Print progress
            print(f'Epoch {epoch + 1}/{epochs}')
            print(f'Beta: {self.current_beta:.4f}, Learning Rate: {current_lr:.6f}')
            print(f'Total Loss: {train_total:.4f} - Reconstruction: {train_reconstruction:.4f} - KL: {train_kl:.4f}')
            print(f'Val Total Loss: {val_total:.4f} - Val Reconstruction: {val_reconstruction:.4f} - Val KL: {val_kl:.4f}')
            
            # Track best weights (for final model)
            if val_total < self.best_loss:
                self.best_loss = val_total
                self.best_weights = [tf.identity(w) for w in self.vae.get_weights()]
        
        # At the end of training, use the best weights
        self.vae.set_weights(self.best_weights)

    def plot_history(self, save_path):
        """Plot and save training history including losses, beta schedule, and learning rate."""
        # Create the plots directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Set style parameters
        #plt.style.use('seaborn')
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f', '#9b59b6']
        
        # Create figure with 5 subplots (added learning rate plot)
        plt.figure(figsize=(15, 12))
        
        # Total Loss
        plt.subplot(3, 2, 1)
        plt.plot(self.history['total_loss'], color=colors[0], label='Train')
        plt.plot(self.history['val_total_loss'], color=colors[0], linestyle='--', label='Val')
        plt.title('Total Loss', fontsize=14, pad=10)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Reconstruction Loss
        plt.subplot(3, 2, 2)
        plt.plot(self.history['reconstruction_loss'], color=colors[1], label='Train')
        plt.plot(self.history['val_reconstruction_loss'], color=colors[1], linestyle='--', label='Val')
        plt.title('Reconstruction Loss', fontsize=14, pad=10)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # KL Loss
        plt.subplot(3, 2, 3)
        plt.plot(self.history['kl_loss'], color=colors[2], label='Train')
        plt.plot(self.history['val_kl_loss'], color=colors[2], linestyle='--', label='Val')
        plt.title('KL Divergence Loss', fontsize=14, pad=10)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Beta Schedule
        plt.subplot(3, 2, 4)
        plt.plot(self.history['beta'], color=colors[3], label='Beta')
        plt.title('Beta Annealing Schedule', fontsize=14, pad=10)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Beta', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Learning Rate
        plt.subplot(3, 2, 5)
        plt.plot(self.history['learning_rate'], color=colors[4], label='Learning Rate')
        plt.title('Learning Rate Schedule', fontsize=14, pad=10)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Use log scale for learning rate
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plots saved to {os.path.join(save_path, 'training_history.png')}")

    def save_model(self, save_path):
        """Save the VAE components"""
        self.vae.save_weights(f'{save_path}/vae.weights.h5')
        self.encoder.save_weights(f'{save_path}/encoder.weights.h5')
        self.decoder.save_weights(f'{save_path}/decoder.weights.h5')


def train_VAE(datasets, h_dim_1, h_dim_2, latent_dim, model_path, l2_reg=0.01, 
              dropout_rate=0.1, batch_size=128, epochs=100, beta=0.4, 
              max_reinit_attempts=10):
    
    print('Initializing training procedure... booting up...')
    
    def initialize_and_check_network():
        """Initialize network and check if initial losses are valid"""
        # Create the VAE
        input_dim = datasets['train']['data'].shape[1]
        vae, encoder, decoder = create_small_VAE(input_dim, h_dim_1, h_dim_2, 
                                               latent_dim, l2_reg, dropout_rate)
        
        # Create trainer
        trainer = VAETrainer(vae, encoder, decoder, beta=beta, n_cycles=4, ratio=0.5)
        
        # Get initial losses on a batch of data
        initial_data = datasets['train']['data'][:batch_size]
        total_loss, reconstruction_loss, kl_loss = trainer.compute_losses(initial_data)
        
        # Check if any loss is NaN
        if tf.math.is_nan(total_loss) or tf.math.is_nan(reconstruction_loss) or tf.math.is_nan(kl_loss):
            return None, None, None, True
        
        return vae, encoder, decoder, False

    # Try initializing the network until we get valid initial losses
    attempt = 0
    while attempt < max_reinit_attempts:
        print(f'\nInitialization attempt {attempt + 1}/{max_reinit_attempts}')
        vae, encoder, decoder, has_nan = initialize_and_check_network()
        
        if not has_nan:
            print('Successfully initialized network with valid losses!')
            break
        
        print('Detected NaN losses, reinitializing network...')
        attempt += 1
        
        if attempt == max_reinit_attempts:
            raise ValueError(f'Failed to initialize network with valid losses after {max_reinit_attempts} attempts')
    
    # Create trainer with the successful initialization
    trainer = VAETrainer(vae, encoder, decoder, beta=beta, n_cycles=4, ratio=0.5)
    
    # Train
    print('\nStarting training.')
    trainer.train(
        datasets['train']['data'],
        datasets['val']['data'],
        epochs=epochs,
        batch_size=batch_size
    )
    print('Training complete!')
    
    # Save the model
    print('Saving model...')
    trainer.save_model(model_path)
    # Plot training history
    trainer.plot_history(model_path)
    print('Model saved! Powering down...')
    
    return trainer.history

# # Loss function for the VAE
# def loss_fn(y_true, model_outputs, beta=0.5):
#     y_pred = model_outputs[0]
#     z_mean = model_outputs[1]
#     z_log_var = model_outputs[2]

#     MSE = tf.reduce_mean(tf.square(y_true - y_pred))
#     KLD = 0.5 * tf.reduce_mean(-1 - z_log_var + tf.square(z_mean) + tf.exp(z_log_var))

#     return MSE + beta * KLD

# # Define reconstruction and KL loss to track during training
# def reconstruction_loss(y_true, model_outputs):
#     y_pred = model_outputs[0]  # Get just the reconstruction
#     return tf.reduce_mean(tf.square(y_true - y_pred))

# def kl_loss(y_true, model_outputs):
#     z_mean = model_outputs[1]
#     z_log_var = model_outputs[2]
#     return 0.5 * tf.reduce_mean(-1 - z_log_var + tf.square(z_mean) + tf.exp(z_log_var))

# # Saving
# def save_vae(vae, encoder, decoder, save_path):

#     vae.save_weights(f'{save_path}/vae.weights.h5')
#     encoder.save_weights(f'{save_path}/encoder.weights.h5')
#     decoder.save_weights(f'{save_path}/decoder.weights.h5')

# Loading
def load_vae(save_path, input_dim, h_dim_1, h_dim_2, latent_dim, l2_reg=0.01, dropout_rate=0.1):
    vae, encoder, decoder = create_small_VAE(input_dim, h_dim_1, h_dim_2, latent_dim, l2_reg, dropout_rate)

    vae.load_weights(f'{save_path}/vae.weights.h5')
    encoder.load_weights(f'{save_path}/encoder.weights.h5')
    decoder.load_weights(f'{save_path}/decoder.weights.h5')

    return vae, encoder, decoder

# # Training the VAE
# def train_VAE(datasets, h_dim_1, h_dim_2, latent_dim, model_path, l2_reg=0.01, dropout_rate=0.1, batch_size=128, epochs=100, beta=0.4, stop_patience=8, lr_patience=4):

#     print('Initializing training procedure... booting up...')
    
#     # Create the VAE
#     input_dim = datasets['train']['data'].shape[1]
#     model, encoder, decoder = create_small_VAE(input_dim, h_dim_1, h_dim_2, latent_dim, l2_reg, dropout_rate)

#     # Compile
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss_fn, metrics = [[reconstruction_loss, kl_loss], None, None])

#     # Define callbacks
#     early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=stop_patience, restore_best_weights=True)
#     lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=lr_patience, min_lr=0.00001)
#     callbacks = [early_stopping, lr_scheduler]

#     # Train
#     print('Starting training.')
#     history = model.fit(datasets['train']['data'], datasets['train']['data'], batch_size=batch_size, epochs=epochs, validation_data=(datasets['val']['data'], datasets['val']['data']), callbacks=callbacks)
#     print('Training complete! ')

#     # Save the model
#     print('Saving model...')
#     save_vae(model, encoder, decoder, model_path)
#     print('Model saved! Powering down...')


# Define different AD score metrics
def MSE_AD_score(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred), axis=-1)

def KL_AD_score(z_mean, z_log_var):
    return np.mean(0.5 * (np.exp(z_log_var) - 1 - z_log_var + np.square(z_mean)), axis=-1)

def clipped_KL_AD_score(z_mean):
    return np.sum(np.square(z_mean), axis=-1)

def MSE_KL_AD_score(y_true, y_pred, z_mean, z_log_var, beta=0.5):
    return MSE_AD_score(y_true, y_pred) + beta * KL_AD_score(z_mean, z_log_var)

# Function to create the student network; trained to predict the AD score directly
def create_student_network(input_dim, h_dim_1, h_dim_2, l2_reg=0.01, dropout_rate=0):
    
    student_inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(h_dim_1, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(student_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_2, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    z = layers.Dense(1, kernel_regularizer=regularizers.l2(l2_reg))(x)

    student_network = Model(student_inputs, z)

    return student_network

# Training the student network
def train_student_network(datasets, save_path, h_dim_1, h_dim_2, l2_reg=0.01, dropout_rate=0.1, batch_size=128, epochs=100, stop_patience=8, lr_patience=4):

    print('Initializing knowledge distillation procedure...')
    
    # Create the student network
    print('Creating student network...')
    input_dim = datasets['train']['data'].shape[1]
    student_network = create_student_network(input_dim, h_dim_1, h_dim_2, l2_reg=l2_reg, dropout_rate=dropout_rate)

    # Compile
    student_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=stop_patience, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=lr_patience, min_lr=0.00001)
    callbacks = [early_stopping, lr_scheduler]

    # Train
    print('Starting training of the student network.')
    history = student_network.fit(datasets['train']['data'], datasets['train']['MSE_KL_AD_scores'], batch_size=batch_size, epochs=epochs, validation_data=(datasets['val']['data'], datasets['val']['MSE_KL_AD_scores']), callbacks=callbacks)
    
    # Save the model
    print('Saving student network...')
    student_network.save_weights(f'{save_path}/student.weights.h5')
    print('Student network saved! Knowledge distillation complete.')

    return student_network

# Plotting a scatter plot of true AD scores vs student predicted AD scores
def plot_student_performance(datasets, plots_path):
    plt.figure(figsize=(15, 8))
    plt.rcParams['axes.linewidth'] = 2.4


    skip_tags = ['train', 'val']
    for tag, data_dict in datasets.items():
        if tag in skip_tags:
            continue

        plt.scatter(data_dict['MSE_KL_AD_scores'], data_dict['student_AD_scores'], label=f'{tag}')


    # Plot diagonal line
    min_val = min(plt.xlim()[0], plt.ylim()[0])
    max_val = max(plt.xlim()[1], plt.ylim()[1])
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='grey', label='Perfect Performance')
    
    # Aesthetics
    plt.xlabel('MSE + KL', fontsize=20)
    plt.ylabel('Student', fontsize=20)
    plt.legend(loc='lower right', fontsize=20)

    # Save
    plt.savefig(os.path.join(plots_path, 'student_performance.png'))
    plt.close()

# Plotting ROC curves. Different plots for each signal, each plot as ROC curves for each AD score metric.
def plot_ROC_curves(datasets, plots_path, use_student=True):

    # Get the test set scores for each AD score metric. These are treated as background scores.
    bkg_MSE_scores = datasets['test']['MSE_AD_scores']
    bkg_KL_scores = datasets['test']['KL_AD_scores']
    bkg_clipped_KL_scores = datasets['test']['clipped_KL_AD_scores']
    bkg_MSE_KL_scores = datasets['test']['MSE_KL_AD_scores']
    if use_student:
        bkg_student_scores = datasets['test']['student_AD_scores']

    # Loop over the signals
    skip_tags = ['train', 'val', 'test']
    for tag, data_dict in datasets.items():
        if tag in skip_tags:
            continue

        # Get the signal scores for each AD score metric
        sig_MSE_scores = data_dict['MSE_AD_scores']
        sig_KL_scores = data_dict['KL_AD_scores']
        sig_clipped_KL_scores = data_dict['clipped_KL_AD_scores']
        sig_MSE_KL_scores = data_dict['MSE_KL_AD_scores']
        if use_student:   
            sig_student_scores = data_dict['student_AD_scores']

        # Create lists of scores and score names
        sig_scores_list = [sig_MSE_scores, sig_KL_scores, sig_clipped_KL_scores, sig_MSE_KL_scores]
        bkg_scores_list = [bkg_MSE_scores, bkg_KL_scores, bkg_clipped_KL_scores, bkg_MSE_KL_scores]
        score_names_list = ['MSE', 'KL', 'clipped KL', 'MSE + KL']
        if use_student:
            sig_scores_list.append(sig_student_scores)
            bkg_scores_list.append(bkg_student_scores)
            score_names_list.append('student')

        # Initialize the plot
        plt.figure(figsize=(15, 8))
        plt.rcParams['axes.linewidth'] = 2.4

        # Loop over each set of scores and score names
        for sig_scores, bkg_scores, score_name in zip(sig_scores_list, bkg_scores_list, score_names_list):

            # Create combined scores and labels -- background scores are 0, signal scores are 1
            combined_scores = np.concatenate((bkg_scores, sig_scores), axis=0)
            combined_labels = np.concatenate((np.zeros(len(bkg_scores)), np.ones(len(sig_scores))), axis=0)

            # Calculate the ROC curve using sklearn
            FPRs, TPRs, thresholds = roc_curve(y_true=combined_labels, y_score=combined_scores)
            AUC = auc(FPRs, TPRs)

            # Plot the ROC curve
            plt.plot(FPRs, TPRs, label=f'{tag} {score_name} (AUC = {AUC:.2f})')

        # Plot the diagonal line for the random classifier
        plt.plot([0, 1], [0, 1], '--', color='grey', label='Random Classifier')

        # Aesthetics
        plt.legend(loc='lower right', fontsize=20)
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.yscale('log')
        plt.xscale('log')
        plt.grid()
        plt.title(f'ROC Curves for {tag}', fontsize=20)

        # Save
        plt.savefig(os.path.join(plots_path, f'{tag}_ROC_curves.png'))
        plt.close()

# Run inference on the VAE and calculate the AD scores for each dataset
def calculate_AD_scores(datasets, model_path, h_dim_1, h_dim_2, latent_dim, l2_reg=0.01, dropout_rate=0.1, beta=0.5):
    
    # Load the VAE
    input_dim = datasets['train']['data'].shape[1]
    loaded_vae, loaded_encoder, loaded_decoder = load_vae(model_path, input_dim, h_dim_1, h_dim_2, latent_dim, l2_reg, dropout_rate)

    # Loop over the datasets
    for tag, data_dict in datasets.items():

        print(f'Evaluating {tag} set...')

        # Run inference on the VAE
        y_pred, z_mean, z_log_var = loaded_vae.predict(data_dict['data'])

        # Store the results
        data_dict['y_pred'] = y_pred
        data_dict['z_mean'] = z_mean
        data_dict['z_log_var'] = z_log_var

        # Calculate the AD scores
        data_dict['MSE_AD_scores'] = MSE_AD_score(data_dict['data'], y_pred)
        data_dict['KL_AD_scores'] = KL_AD_score(z_mean, z_log_var)
        data_dict['clipped_KL_AD_scores'] = clipped_KL_AD_score(z_mean)
        data_dict['MSE_KL_AD_scores'] = MSE_KL_AD_score(data_dict['data'], y_pred, z_mean, z_log_var, beta=beta)

    return datasets

def evaluate_VAE(datasets, model_path, plots_path, h_dim_1, h_dim_2, latent_dim, l2_reg=0.01, dropout_rate=0.1, beta=0.5, train_student=False, load_student=True):

    print('Beginning evaluation procedure... booting up...')
    
    # Calculate the AD scores for each dataset
    print('Calculating AD scores for each dataset...')
    datasets = calculate_AD_scores(datasets, model_path, h_dim_1, h_dim_2, latent_dim, l2_reg, dropout_rate, beta)

    # Train the student network if specified
    if train_student:   
        print('Training student network...')
        student_network = train_student_network(datasets, model_path, h_dim_1, h_dim_2, l2_reg=0.01, dropout_rate=0.1, batch_size=128, epochs=100, stop_patience=8, lr_patience=4)

    # Load the student network if specified
    if load_student:
        print('Loading student network...')
        input_dim = datasets['train']['data'].shape[1]

        student_network = create_student_network(input_dim, h_dim_1, h_dim_2, l2_reg=l2_reg, dropout_rate=dropout_rate)
        student_network.load_weights(f'{model_path}/student.weights.h5')
        print('Student network loaded!')

    # Run inference on the student network and plot performance if specified
    if train_student or load_student:
        print('Running inference on the student network...')
        for tag, data_dict in datasets.items():
            data_dict['student_AD_scores'] = student_network.predict(data_dict['data'])
        print('Inference complete! Plotting student performance...')
        plot_student_performance(datasets, plots_path)

    # Plot the ROC curves
    print('Plotting ROC curves...')
    use_student = train_student or load_student
    plot_ROC_curves(datasets, plots_path, use_student=use_student)
    print('ROC curves plotted! Evaluation complete.')

    return datasets
        
    