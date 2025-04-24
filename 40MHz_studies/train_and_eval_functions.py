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


def load_and_preprocess(standard_scaler=True):

    print('Booting up...\nStarting to load data...\n')

    # Initialize empty dictionary to store the datasets. This will be a dictionary of dictionaries, one for each dataset
    datasets = {}

    # Load the data from the hdf5 files
    base_path = '/eos/home-m/mmcohen/chop_or_not_development/data/'
    for file_name in os.listdir(base_path):
        if file_name.startswith('.'):
            continue

        dataset_name = file_name.split('_')[0]
        file_path = os.path.join(base_path, file_name)
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
        
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        
# Create a small VAE.
def create_small_VAE(input_dim, h_dim_1, h_dim_2, latent_dim, l2_reg=0.01, dropout_rate=0):
    
    # Encoder
    encoder_inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(h_dim_1, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(h_dim_2, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    z_mean = layers.Dense(latent_dim, kernel_regularizer=regularizers.l2(l2_reg))(x)
    z_log_var = layers.Dense(latent_dim, kernel_regularizer=regularizers.l2(l2_reg))(x)
    z = Sampling()([z_mean, z_log_var])
    
    encoder = Model(inputs=encoder_inputs, outputs=[z_mean, z_log_var, z])

    # Decoder
    decoder_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(h_dim_2, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(decoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(h_dim_1, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(input_dim, kernel_regularizer=regularizers.l2(l2_reg))(x)

    decoder = Model(inputs=decoder_inputs, outputs=outputs)

    mean, log_var, latent = encoder(encoder_inputs)

    ae_outputs = [decoder(latent), mean, log_var]
    ae = Model(encoder_inputs, outputs=ae_outputs)

    return ae, encoder, decoder

# Loss function for the VAE
def loss_fn(y_true, model_outputs, beta=0.5):
    y_pred = model_outputs[0]
    z_mean = model_outputs[1]
    z_log_var = model_outputs[2]

    MSE = tf.reduce_mean(tf.square(y_true - y_pred))
    KLD = 0.5 * tf.reduce_mean(-1 - z_log_var + tf.square(z_mean) + tf.exp(z_log_var))

    return MSE + beta * KLD

# Define reconstruction and KL loss to track during training
def reconstruction_loss(y_true, model_outputs):
    y_pred = model_outputs[0]  # Get just the reconstruction
    return tf.reduce_mean(tf.square(y_true - y_pred))

def kl_loss(y_true, model_outputs):
    z_mean = model_outputs[1]
    z_log_var = model_outputs[2]
    return 0.5 * tf.reduce_mean(-1 - z_log_var + tf.square(z_mean) + tf.exp(z_log_var))

# Saving
def save_vae(vae, encoder, decoder, save_path):

    vae.save_weights(f'{save_path}/vae.weights.h5')
    encoder.save_weights(f'{save_path}/encoder.weights.h5')
    decoder.save_weights(f'{save_path}/decoder.weights.h5')

# Loading
def load_vae(save_path, input_dim, h_dim_1, h_dim_2, latent_dim, l2_reg=0.01, dropout_rate=0.1):
    vae, encoder, decoder = create_small_VAE(input_dim, h_dim_1, h_dim_2, latent_dim, l2_reg, dropout_rate)

    vae.load_weights(f'{save_path}/vae.weights.h5')
    encoder.load_weights(f'{save_path}/encoder.weights.h5')
    decoder.load_weights(f'{save_path}/decoder.weights.h5')

    return vae, encoder, decoder

# Training the VAE
def train_VAE(datasets, h_dim_1, h_dim_2, latent_dim, model_path, l2_reg=0.01, dropout_rate=0.1, batch_size=128, epochs=100, beta=0.4, stop_patience=8, lr_patience=4):

    print('Initializing training procedure... booting up...')
    
    # Create the VAE
    input_dim = datasets['train']['data'].shape[1]
    model, encoder, decoder = create_small_VAE(input_dim, h_dim_1, h_dim_2, latent_dim, l2_reg, dropout_rate)

    # Compile
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss_fn, metrics = [[reconstruction_loss, kl_loss], None, None])

    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=stop_patience, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=lr_patience, min_lr=0.00001)
    callbacks = [early_stopping, lr_scheduler]

    # Train
    print('Starting training.')
    history = model.fit(datasets['train']['data'], datasets['train']['data'], batch_size=batch_size, epochs=epochs, validation_data=(datasets['val']['data'], datasets['val']['data']), callbacks=callbacks)
    print('Training complete! ')

    # Save the model
    print('Saving model...')
    save_vae(model, encoder, decoder, model_path)
    print('Model saved! Powering down...')


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
        
    