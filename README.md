# 📚 Biblious: AI-Powered Book Recommendation Engine

Welcome to my fork of the **Biblious** project! Biblious is a next-generation social book tracking and intelligent discovery platform. 

This repository specifically highlights my contributions to the project: the design, development, and optimization of the **Machine Learning Recommendation Engine**. 

## 🧑‍💻 My Role & Contribution

This project was developed by a team of 5 members. While my teammates successfully built the Full-Stack architecture using Next.js and FastAPI, **I took full ownership of the data analysis and machine learning pipeline.**

My primary focus and the core of this repository's machine learning logic is located in the `recommendation_engine.py` file. I was responsible for:

* **Data Processing & Filtering:** Processing millions of rows from the Goodreads dataset, filtering out noisy tags and sparse interactions to curate a highly reliable dataset of 5.4 million ratings.

![veri hazırlama](https://github.com/user-attachments/assets/e2f65074-15d2-4d75-9976-3cc76bdd639e)

* **Algorithm Development:** Building a 3-layered Hybrid Recommendation System combining Collaborative Filtering and Content-Based Filtering.
* **Model Evaluation:** Designing realistic hold-out validation methods to measure Precision@K, Catalog Coverage, and RMSE to prevent model overfitting.

## 🧠 Recommendation Engine Architecture (recommendation_engine.py)

The recommendation engine goes beyond simple popularity metrics by understanding complex user taste patterns through a hybrid approach:

* **1. Collaborative Filtering (SVD):** Uses Singular Value Decomposition (SVD) from the `Surprise` library. It learns latent factors from user-item interactions to predict how much a user will rate an unread book.

![kullanıcı bazlı](https://github.com/user-attachments/assets/adf1e1f7-95f9-4c9c-9df9-a744eb75d272)

* **2. Content-Based Filtering (TF-IDF Tag Profiles):** Analyzes book metadata. It builds a sparse TF-IDF matrix of user-generated tags to calculate cosine similarity between books, ensuring recommendations make thematic sense.
* **3. Hybrid Scoring Mechanism:** Blends SVD predictions (user taste) with Tag similarities (thematic overlap) and applies a slight, logarithmically scaled popularity boost to avoid recommending overly obscure titles.
* **4. Smart Discovery Filters:** * **Anti-Series Repetition:** Automatically detects series names from titles and prevents the engine from redundantly recommending books from a series the user has already read.
  * **Author Exploration:** Recommends popular unread books from authors the user already enjoys.

![Tek kitap Hibrit](https://github.com/user-attachments/assets/d2cd9acb-94e2-42b4-864a-b1221bb1377f)

## 📊 Model Performance & Metrics

![metrikler](https://github.com/user-attachments/assets/939201d8-ccd6-4c36-8421-b324ec28c22c)

To ensure the engine provides diverse and accurate recommendations, the model was evaluated using strict validation methods:

* **Train RMSE:** 0.7899 (Measuring model's memory)
* **Test RMSE:** 0.8370 (Measuring generalization power)
* **Precision@5:** 21.18% (Percentage of top 5 recommendations that were actually highly rated by the user in the hold-out set)
* **Catalog Coverage:** 52.15% (Ensuring the model doesn't just recommend top bestsellers, but explores over half the catalog)

## 🎯 Final Product

![öneri sayfası](https://github.com/user-attachments/assets/82e6e4b8-c9f3-48f9-bcab-6669f715d79c)

## 🛠️ Tech Stack & Libraries Used

* **Language:** Python 3.12+
* **Machine Learning:** Scikit-Surprise (SVD)
* **Data Manipulation:** Pandas, NumPy
* **Sparse Matrix Operations:** SciPy (csr_matrix)
* **Text Processing:** Regex (re)

## 🚀 How to Run the Recommendation Engine

You can test the recommendation logic locally without running the entire web application.

1. Install the required dependencies:
```bash
pip install pandas numpy scikit-surprise scipy
```

2. Place your dataset files (`books.csv`, `ratings.csv`, `tags.csv`, `book_tags.csv`) in the root directory.

3. Run the Python script:
```bash
python recommendation_engine.py
```

The script will train the SVD model, build the TF-IDF tag matrix, calculate evaluation metrics, and print out sample recommendations (Smart Recommendations, Same Author, Series Continuations, and Hybrid Similarities) directly to the terminal.

## 🌐 Full-Stack Application Repository

This repository focuses specifically on the Machine Learning pipeline and Recommendation Engine. If you want to see the full implementation of the web platform, including the Next.js frontend and FastAPI backend, please visit the main team repository:
👉 [Full-Stack Biblious Repo](https://github.com/metehanyurdunusvn/Biblious)
