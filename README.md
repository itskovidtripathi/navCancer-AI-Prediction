# LungScan-AI

Advanced Lung Cancer Detection Using Artificial Intelligence

![LungScan-AI Logo](https://img.shields.io/badge/LungScan-AI-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iI2ZmZiIgZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPjwvc3ZnPg==)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red?style=for-the-badge&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-orange?style=for-the-badge&logo=pytorch)

## Features

- Upload and analyze chest X-ray images
- Detect three types of conditions:
  - Normal
  - Squamous Cell Carcinoma
  - Adenocarcinoma
- Detailed medical report generation
- Interactive visualization of results
- User registration and history tracking

## Technical Stack

- Python 3.11+
- Streamlit for web interface
- PyTorch for deep learning (ResNet18)
- SQLite for database

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd LungScan-AI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.env` file in the project root
   - Add the following variables:
   ```
   ADMIN_USERNAME=your_admin_username
   ADMIN_PASSWORD=your_secure_password
   DATABASE_PATH=lung_cancer_app.db
   ```

4. Ensure the model file is in the correct location:
```
models/lung_cancer_detector.pth
```

## Running the Application

```bash
streamlit run src/app.py
```

## Deployment

### Local Deployment
1. Follow the installation steps above
2. Run the application using `streamlit run src/app.py`
3. Access the application at `http://localhost:8501`

### Streamlit Cloud Deployment
1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select src/app.py as the main file
5. Deploy

## Project Structure

```
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── config.py           # Configuration settings
│   ├── data_preprocessing.py # Data preprocessing utilities
│   ├── train.py            # Model training script
│   └── database.py         # Database operations
├── models/
│   └── lung_cancer_detector.pth  # Trained model
├── data/
│   └── train/              # Training data directory
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Model Performance

- Accuracy: 97.03%
- Supports detection of three classes
- Based on ResNet18 architecture

## Security and Privacy

- User data is stored securely in a local database
- X-ray images are processed locally
- No data is shared with external services

## License

[Your License] 