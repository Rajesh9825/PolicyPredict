# PolicyPredict  - Insurance Policy Price Prediction

**PolicyPredict** is my product-based machine learning project aimed at predicting insurance charges based on a customer's profile. Though it's a regression problem at its core, I’ve given it a product-like structure with a complete machine learning pipeline, web interface, and deployment setup to simulate a real-world scenario.

##  Overview

Predicting medical insurance costs can help both customers and insurers understand premium estimates better. This project uses demographic and health-related features such as age, BMI, smoking habits, and region to predict insurance charges.

This project goes beyond just model training - it covers everything from data ingestion to model deployment with a full modular ML pipeline, a web interface, and CI/CD deployment to the cloud.

---

##  Research & EDA

Before diving into pipeline development, I explored the dataset using **Kaggle Notebooks**:

- Tested multiple models: `Linear Regression`, `Elastic Net`, `SVM`, `Decision Tree`, `Random Forest`, and `XGBoost`.
- Performed **feature importance analysis**, and observed that `smoker`, `bmi`, and `age` were the most influential features.
- Plotted RMSE for each model and found **Decision Tree Regressor** performed the best on the test set.
- Plotted **actual vs predicted charges** and **residual plots** - confirming the Decision Tree’s better fit.

Based on this exploration, I chose **Decision Tree** as the final model for production.

Kaggle Notebook : https://www.kaggle.com/code/rajeshgajengi/insurance-price-prediction

---

##  Project Workflow

###  1. Data Ingestion
- Previously, I used to download and unzip Kaggle datasets manually.
- In this project, I used the **Kaggle API** to automate dataset download and extraction.
- Saved the raw data directly in `.csv` format as part of the pipeline.

###  2. Data Validation
- Verified schema: column names, data types, and null checks.
- Generated a `JSON` report showing validation results (True/False).
- Only if validation passes, data moves to the transformation stage.

###  3. Data Transformation
- Performed an 80/20 train-test split.
- Applied **OneHotEncoding** using `sklearn ColumnTransformer`.
- Saved the transformed and split data for downstream use.

###  4. Model Training
- Loaded the transformed data.
- Trained the **Decision Tree Regressor**, based on prior EDA findings.

###  5. Model Evaluation
- Used **MLflow** to track experiments, parameters, and metrics.
- Evaluated using:
  - R² Score
  - RMSE
  - MAE
- Compared runs and selected the best-performing parameters via MLflow UI.

---

##  Web Application

- Developed a simple web interface using **Flask + HTML**.
- Features:
  - Take user input for predictions
  - Display predicted insurance cost
  - Optional: Retrain the model with new data.

---

##  Deployment

###  Docker
- Created a `Dockerfile` to containerize the entire application with all dependencies.

###  CI/CD with GitHub Actions
- Implemented a CI/CD pipeline to automate deployment.
- On every push to GitHub:
  - Docker image is built
  - Pushed to **AWS ECR**
  - Deployed to **AWS EC2**

###  Hosting
- For cost-effective public hosting, used **Render** .
- Render’s built-in CI/CD re-deploys the app automatically on every commit.

 **Live Demo**: https://policypredict.onrender.com  
 <!-- **Demo Video**: [Add Video Link Here] -->

---

##  Key Learnings & Takeaways

- Built a complete **end-to-end ML project with modular code structure**
- Discovered a better approach to data ingestion using **Kaggle API**
- Learned how to use **MLflow for experiment tracking**
- Gained hands-on experience in **Dockerization** and **AWS deployment**
- Implemented **CI/CD pipelines** using GitHub Actions
- Experienced how to structure ML workflows like a real-world product

---

##  Tech Stack

- Python, Pandas, NumPy, Scikit-learn, XGBoost
- MLflow, Matplotlib, Seaborn
- Flask, HTML, CSS
- Docker
- AWS (ECR + EC2)
- Render (free hosting)
- GitHub Actions (CI/CD)

---

<!-- ## 📈 Results

| Model           | RMSE     | R² Score |
|----------------|----------|----------|
| Linear Regression | ...    | ...      |
| Decision Tree   | ✅ Lowest | ✅ Best   |
| XGBoost         | ...      | ...      |

*(Full comparison and plots are available in the notebook.)*

--- -->

<!-- ## 🧭 Future Improvements

- Use a larger dataset with more policy features
- Add user authentication in the web app
- Integrate a database for storing user predictions
- Monitor model drift and retrain automatically

--- -->

<!-- ## 🙋‍♂️ Author

**Your Name**  
📧 [YourEmail@example.com]  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

--- -->

<!-- ## 📁 Folder Structure -->

