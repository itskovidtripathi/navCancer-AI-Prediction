import torch

# Data paths
TRAIN_DATA_PATH = "data/lung_cancer_image_set"
TEST_DATA_PATH = "data/lung_cancer_image_set"

# Model parameters
MODEL_NAME = "resnet18"
NUM_CLASSES = 3
INPUT_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

# Training parameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model save path
MODEL_SAVE_PATH = "models/lung_cancer_detector.pth"

# Data augmentation parameters
MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
STD = [0.229, 0.224, 0.225]   # ImageNet std

# Class names mapping
CLASS_NAMES = {
    'lung_n': 'Normal',
    'lung_scc': 'Squamous Cell Carcinoma',
    'lung_aca': 'Adenocarcinoma'
} 


# Medical Information Dictionary
MEDICAL_INFO = {
    'Normal': {
        'description': 'The chest X-ray appears normal with no significant abnormalities detected. The lung fields are clear and well-expanded.',
        'characteristics': [
            'Clear lung fields',
            'Normal cardiac silhouette',
            'No visible masses or nodules'
        ],
        'risk_factors': [
            'Smoking history',
            'Exposure to secondhand smoke',
            'Family history of lung cancer',
            'Exposure to radon',
            'Exposure to asbestos'
        ],
        'recommendations': [
            'Continue routine health check-ups',
            'Maintain a healthy lifestyle',
            'Schedule next screening as per medical guidelines'
        ],
        'follow_up': 'Schedule next routine screening in 1 year or as recommended by your healthcare provider.'
    },
    'Squamous Cell Carcinoma': {
        'description': 'Squamous cell carcinoma is detected, which is a type of non-small cell lung cancer that usually starts in the larger airways.',
        'characteristics': [
            'Central mass or nodule',
            'Possible cavitation',
            'Associated hilar lymphadenopathy',
            'May show post-obstructive pneumonia'
        ],
        'risk_factors': [
            'Heavy smoking history',
            'Exposure to industrial carcinogens',
            'Chronic lung inflammation',
            'Previous radiation therapy',
            'Genetic predisposition'
        ],
        'recommendations': [
            'Immediate consultation with an oncologist',
            'Further diagnostic testing (PET scan, biopsy)',
            'Staging evaluation',
            'Discussion of treatment options (surgery, radiation, chemotherapy)',
            'Regular follow-up appointments'
        ],
        'follow_up': 'Immediate follow-up with oncology team for comprehensive evaluation and treatment planning. Regular imaging and clinical assessments will be scheduled based on treatment protocol.'
    },
    'Adenocarcinoma': {
        'description': 'Adenocarcinoma is detected, which is a type of non-small cell lung cancer that typically occurs in the outer parts of the lung.',
        'characteristics': [
            'Peripheral nodule or mass',
            'Ground-glass opacity possible',
            'May show pleural involvement',
            'Possible lymphangitic spread'
        ],
        'risk_factors': [
            'Smoking history',
            'Exposure to air pollution',
            'Genetic mutations (EGFR, ALK)',
            'Hormonal factors',
            'Previous lung diseases'
        ],
        'recommendations': [
            'Urgent oncology consultation',
            'Complete staging workup',
            'Molecular testing for targeted therapy',
            'Discussion of personalized treatment plan',
            'Regular monitoring and follow-up',
            'Consider genetic counseling'
        ],
        'follow_up': 'Urgent follow-up with oncology team for molecular testing and personalized treatment planning. Regular imaging and clinical assessments will be scheduled based on treatment protocol.'
    }
} 