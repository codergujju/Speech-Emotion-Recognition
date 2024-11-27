Speech Emotion Recognition using the TESS Dataset
This project focuses on building a deep learning model to recognize emotions from speech data using the TESS (Toronto Emotional Speech Set) dataset. It employs Librosa for audio feature extraction, Seaborn and Matplotlib for visualization, and an LSTM-based neural network for classification.

Dataset
The TESS Dataset contains audio samples from two speakers expressing seven emotions:

Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral.
Dataset Download
The dataset can be downloaded from Kaggle:

bash
Copy code
kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess
Project Structure
Data Loading and Preprocessing
Load audio files and labels.
Visualize data distribution.
Feature Extraction
Use MFCC (Mel Frequency Cepstral Coefficients) for feature extraction.
Normalize features to standardize inputs.
Data Visualization
Waveplots and spectrograms for each emotion.
Model Architecture
A Sequential LSTM-based neural network with:
One LSTM layer.
Dense layers for classification.
Dropout for regularization.
Output layer with softmax activation for multi-class emotion classification.
Training
Trained using categorical_crossentropy loss and Adam optimizer for 120 epochs.
kaggle dataset:- https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
Feature Extraction
The MFCC features were extracted using librosa.feature.mfcc, capturing 40 coefficients per audio sample. These features were standardized and used as input to the LSTM model.

Model Architecture
The LSTM model includes:

Input shape: (40, 1) (MFCC features for each audio sample).
Layers:
LSTM layer with 123 units.
Dense layers with 64 and 32 units, using ReLU activation.
Dropout layers with a rate of 0.2 for regularization.
Softmax output layer for multi-class classification.
Results
Training and validation accuracy were tracked over 120 epochs.
Accuracy plots demonstrate the model's performance.
Accuracy Plot
Visualized training and validation accuracy trends:

Python
Copy code
epochs = list(range(120))
plt.plot(epochs, acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
Visualization
Waveplot and Spectrogram
Waveforms and spectrograms provide insights into the structure of audio data. Examples:
Waveplot (Fear):
Spectrogram (Fear):
