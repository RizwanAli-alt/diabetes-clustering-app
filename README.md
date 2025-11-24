# 🔬 Pakistani Diabetes K-Means Clustering - Streamlit App

A professional machine learning web application for diabetes patient clustering analysis, perfect for showcasing on your LinkedIn profile.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

- **Interactive Data Analysis**: Explore the Pakistani Diabetes Dataset with real-time visualizations
- **K-Means Clustering**: Advanced unsupervised learning for patient segmentation
- **Performance Metrics**: Comprehensive evaluation with Silhouette, Calinski-Harabasz, and Davies-Bouldin scores
- **Professional UI**: Clean, modern interface perfect for portfolio presentation
- **Downloadable Results**: Export clustering results for further analysis

## 🚀 Live Demo

[View Live Demo](http://localhost:8501/) 

## 📸 Screenshots

### Main Dashboard
- Dataset overview with key statistics
- Interactive feature distributions
- Correlation matrix visualization

### Clustering Analysis
- Elbow method for optimal K selection
- Real-time clustering with adjustable parameters
- PCA visualization of clusters

### Performance Metrics
- Silhouette score analysis
- Cross-dataset performance comparison
- Comprehensive evaluation metrics

### Cluster Insights
- Detailed cluster characteristics
- Diabetes prevalence by cluster
- Risk stratification analysis

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Deployment**: Streamlit Cloud

## 📋 Prerequisites

- Python 3.8 or higher
- Pakistani Diabetes Dataset (CSV format)

## 💻 Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/diabetes-clustering-app.git
   cd diabetes-clustering-app
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

4. **Run the app**
   ```bash
   streamlit run streamlit_diabetes_clustering.py
   ```

5. **Open in browser**
   - Navigate to `http://localhost:8501`

## 🌐 Deployment on Streamlit Cloud

1. **Push to GitHub**
   - Create a new repository on GitHub
   - Push your code including `requirements_streamlit.txt`

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and branch
   - Set main file path: `streamlit_diabetes_clustering.py`
   - Click "Deploy"

3. **Share your app**
   - Get your public URL: `https://your-app-name.streamlit.app`
   - Add to your LinkedIn profile

## 📊 Dataset Information

The app uses the Pakistani Diabetes Dataset with the following features:
- **Age**: Patient age in years
- **Weight (wt)**: Patient weight in kg
- **BMI**: Body Mass Index
- **A1c**: Glycated hemoglobin level (%)
- **B.S.R**: Blood Sugar Random (mg/dL)
- **Outcome**: Diabetes status (0: No, 1: Yes)

## 🎯 Key Insights

The clustering analysis reveals:
- Three distinct patient segments with varying diabetes risk
- High-risk clusters characterized by elevated A1c and BMI
- Clear patterns for targeted intervention strategies

## 📱 Usage Tips

1. **Upload Data**: Use the sidebar to upload your dataset
2. **Adjust Parameters**: Experiment with different K values
3. **Explore Tabs**: Navigate through all five analysis tabs
4. **Download Results**: Export clustering results as CSV

## 🔧 Customization

### Update Branding
- Replace placeholder logo in sidebar
- Update color scheme in CSS section
- Add your name and LinkedIn profile

### Extend Features
- Add more clustering algorithms
- Include additional visualizations
- Implement real-time predictions

## 📝 Code Structure

```
diabetes-clustering-app/
│
├── streamlit_diabetes_clustering.py  # Main application
├── requirements_streamlit.txt        # Dependencies
├── README_streamlit.md              # This file
└── Pakistani_Diabetes_Dataset.csv   # Dataset (not included)
```

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**[Your Name]**
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- GitHub: [github.com/yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Pakistani healthcare institutions for the dataset
- Streamlit team for the amazing framework
- scikit-learn community for ML tools

---

### 📌 LinkedIn Post Template

```
🚀 Excited to share my latest Machine Learning project!

I've developed an interactive web application that uses K-Means clustering to analyze diabetes patient data. The app identifies distinct patient segments, enabling targeted healthcare interventions.

🔬 Key Features:
• Real-time clustering analysis
• Interactive visualizations
• Performance metrics dashboard
• Risk stratification insights

💻 Tech Stack: Python, Streamlit, scikit-learn, Plotly

🔗 Live Demo: [your-app-url]
📊 GitHub: [your-github-url]

#MachineLearning #DataScience #Healthcare #Python #Streamlit #Portfolio
```

---

*Last updated: November 2024*