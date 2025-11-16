import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- Configuration & Setup ---

# Initialize session state for storing base prediction results
if 'base_prediction' not in st.session_state:
    st.session_state['base_prediction'] = None
if 'base_actual_salary_value' not in st.session_state:
    st.session_state['base_actual_salary_value'] = 0.0

try:
    model = joblib.load("final_mlr_model.joblib") 
except FileNotFoundError:
    st.error("Error: Model file 'final_mlr_model.joblib' not found. Please ensure it's in the correct directory.")
    st.stop()

# Load dataset for dropdowns 
try:
    df = pd.read_csv("cleaned_file.xlsx - Sheet1.csv")
except FileNotFoundError:
    try:
        df = pd.read_excel("cleaned_file.xlsx")
    except FileNotFoundError:
        st.error("Error: Data file 'cleaned_file.xlsx - Sheet1.csv' or 'cleaned_file.xlsx' not found.")
        st.stop()

# Extract model components for transparency (Calculation Display)
mlr_model = model['mlr']
INTERCEPT = mlr_model.intercept_
COEFFICIENTS = pd.Series(mlr_model.coef_, index=model['preprocess'].get_feature_names_out()).to_dict()
COEFFICIENTS = {k.split('__', 1)[-1]: v for k, v in COEFFICIENTS.items()}


# --- NEW CUSTOM SORTING FUNCTION ---
def sort_options_with_other(options):
    """Sorts options alphabetically but ensures 'Other' is always last."""
    options_list = list(options)
    
    # Check for both common spellings of 'Other'
    other_labels = ['Other', 'other', 'Other ', 'other ']
    
    other_items = [item for item in options_list if item in other_labels]
    
    # Filter out the other items from the main list
    main_items = [item for item in options_list if item not in other_labels]
    
    # Sort the main list
    main_items.sort()
    
    # Append the "Other" labels (if they exist) to the end
    return main_items + other_items
# --- END CUSTOM SORTING FUNCTION ---


# Function to perform prediction and return calculation breakdown
def calculate_feature_contribution(input_data_dict):
    """Calculates prediction and provides a detailed breakdown of feature contributions."""
    
    input_df = pd.DataFrame([input_data_dict])
    total_prediction = model.predict(input_df)[0]
    
    X_processed = model['preprocess'].transform(input_df)
    
    breakdown = []
    feature_names_processed = model['preprocess'].get_feature_names_out()
    
    for i, feature_name_full in enumerate(feature_names_processed):
        feature_name = feature_name_full.split('__', 1)[-1]
        coef = COEFFICIENTS.get(feature_name)
        value_processed = X_processed[0, i]
        contribution = coef * value_processed
        
        if np.abs(contribution) > 0.001:
            if '_' in feature_name:
                display_name = feature_name
                
                # Correction for Q12 display
                if display_name.startswith('Q12 - Highest Level of Education'):
                    display_name = display_name.replace('Q12 - Highest Level of Education_', 'Education_')
                
                # Correction for Programming Language display
                if display_name.startswith('Favorite Programming Language.1'):
                    display_name = display_name.replace('Favorite Programming Language.1', 'Favorite Programming Language')
                
            else:
                original_value = input_data_dict.get(feature_name)
                display_name = f"{feature_name} (Value: {original_value:.2f})"
            
            breakdown.append({
                'Feature': display_name,
                'Contribution': contribution
            })
            
    breakdown.append({'Feature': 'Intercept (Base Salary)', 'Contribution': INTERCEPT})
    
    return total_prediction, breakdown


st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="wide")
st.title("💼 HR Salary Prediction & What-If Analysis")

# --- COLLECT AND SORT ALL REQUIRED OPTIONS ---
roles = sort_options_with_other(df["Role"].dropna().unique())
countries = sort_options_with_other(df["Country"].dropna().unique())
education_levels = sort_options_with_other(df["Q12 - Highest Level of Education"].dropna().unique())
industry_options = sort_options_with_other(df["Industry"].dropna().unique())
language_options = sort_options_with_other(df["Favorite Programming Language.1"].dropna().unique())
gender_options = sort_options_with_other(df["Gender"].dropna().unique())


# --- 1. CURRENT PROFILE INPUTS (SIDEBAR) ---
st.sidebar.header("1. Your Current Profile")

# Primary Inputs - STACKED VERTICALLY
base_actual_salary = st.sidebar.number_input("Actual Current Salary (K USD)", min_value=1.0, value=70.0, step=1.0, format="%.2f", key="base_actual_salary_input")
base_age = st.sidebar.slider("Age (Base)", 18, 70, 30, key="base_age") 
base_role = st.sidebar.selectbox("Role (Base)", roles, key="base_role")
base_country = st.sidebar.selectbox("Country (Base)", countries, key="base_country")
base_education_clean = st.sidebar.selectbox("Highest Level of Education", education_levels, 
                                  index=education_levels.index('Bachelors') if 'Bachelors' in education_levels else 0, 
                                  key="base_education_clean")

# Secondary Inputs - STACKED VERTICALLY
base_gender = st.sidebar.selectbox("Gender", gender_options, index=gender_options.index('Male') if 'Male' in gender_options else 0, key="base_gender")
base_industry = st.sidebar.selectbox("Industry", industry_options, index=industry_options.index('Tech') if 'Tech' in industry_options else 0, key="base_industry")
base_language = st.sidebar.selectbox("Highest Proficient Language", language_options, index=language_options.index('Python') if 'Python' in language_options else 0, key="base_language")


# --- HAPPINESS COMPONENTS (MAIN PANEL - SEPARATE SECTION) ---
st.subheader("Happiness Component Scores (Scale 1-10)")

# Main panel columns for compact display
h_cols = st.columns(3)
h_cols_b = st.columns(3)

# COLLECTING 6 HAPPINESS COMPONENTS
with h_cols[0]:
    base_h1 = st.slider("Salary Happiness", 1, 10, 7, key="base_h1")
with h_cols[1]:
    base_h2 = st.slider("Work/Life Balance Happiness", 1, 10, 7, key="base_h2")
with h_cols[2]:
    base_h3 = st.slider("Coworkers Happiness", 1, 10, 7, key="base_h3")
with h_cols_b[0]:
    base_h4 = st.slider("Management Happiness", 1, 10, 7, key="base_h4")
with h_cols_b[1]:
    base_h5 = st.slider("Upward Mobility Happiness", 1, 10, 7, key="base_h5")
with h_cols_b[2]:
    base_h6 = st.slider("Learning New Things Happiness", 1, 10, 7, key="base_h6")

# CALCULATE TOTAL HAPPINESS SCORE (MEAN)
base_happiness_components = [base_h1, base_h2, base_h3, base_h4, base_h5, base_h6]
base_happiness = np.mean(base_happiness_components)

st.markdown("---")


# --- STAGE 1 PREDICTION BUTTON (MAIN PANEL) ---
if st.button("Predict Average Salary for Profile"):
    # 1. CALCULATE BASE PREDICTION
    base_data_dict = {
        "Age": base_age,
        "Total Happines": base_happiness, 
        "Role": base_role,
        "Country": base_country,
        "Q12 - Highest Level of Education": base_education_clean, 
        "Industry": base_industry,
        "Favorite Programming Language.1": base_language, 
        "Gender": base_gender,
    }
    
    base_prediction, _ = calculate_feature_contribution(base_data_dict)

    # STORE RESULTS IN SESSION STATE
    st.session_state['base_prediction'] = base_prediction
    st.session_state['base_actual_salary_value'] = base_actual_salary
    
    # --- CALCULATE PERCENTAGE DIFFERENCE FOR BASE ---
    prediction_vs_actual_diff = base_prediction - base_actual_salary
    if base_actual_salary != 0:
        prediction_vs_actual_perc = (prediction_vs_actual_diff / base_actual_salary) * 100
        delta_text = f"{prediction_vs_actual_perc:.2f}% ({'Overpredicted' if prediction_vs_actual_perc >= 0 else 'Underpredicted'})"
    else:
        delta_text = "N/A"
    
    # --- DISPLAY BASE RESULT ---
    st.markdown("### 🎯 Model's Current Prediction")
    
    p_col1, p_col2 = st.columns(2)
    
    with p_col1:
        st.metric(
            label="Model Predicted Salary (Your Profile)",
            value=f"{base_prediction:.2f} K USD"
        )
    with p_col2:
        st.metric(
            label="Difference: Predicted vs. Actual Salary", 
            value=f"{prediction_vs_actual_diff:.2f} K USD",
            delta=delta_text
        )
    
st.markdown("---")

# --- 2. WHAT-IF SCENARIO INPUTS (STAGE 2 - MAIN PANEL) ---
st.header("2. What-If Scenario Builder")

if st.session_state['base_prediction'] is None:
    st.warning("Please click 'Predict Average Salary for Profile' first to establish the base for comparison.")

enable_what_if = st.checkbox("Enable What-If Scenario (Check to compare against a hypothetical profile)")

# Conditional Input Definitions
if enable_what_if:
    st.info("Modify any of the fields below. Unchanged fields inherit values from your 'Current Profile'.")

    # Get current base values for defaults
    current_base_education = base_education_clean 
    current_base_gender = base_gender
    current_base_industry = base_industry
    current_base_language = base_language

    # MAIN PANEL LAYOUT FOR WHAT-IF INPUTS
    w_cols = st.columns(4)
    w_cols_b = st.columns(4)
    w_h_cols = st.columns(3)
    w_h_cols_b = st.columns(3)

    with w_cols[0]:
        what_if_age = st.slider("Age (What-If)", 18, 70, base_age, key="what_if_age")
        what_if_role = st.selectbox("Role (What-If)", roles, index=roles.index(base_role), key="what_if_role")

    with w_cols[1]:
        what_if_country = st.selectbox("Country (What-If)", countries, index=countries.index(base_country), key="what_if_country")
        what_if_education = st.selectbox("Highest Level of Education (What-If)", education_levels, 
                                       index=education_levels.index(current_base_education), key="what_if_education")
        
    with w_cols[2]:
        what_if_gender = st.selectbox("Gender (What-If)", gender_options, index=gender_options.index(current_base_gender), key="what_if_gender")
        what_if_industry = st.selectbox("Industry (What-If)", industry_options, index=industry_options.index(current_base_industry), key="what_if_industry")

    with w_cols[3]:
        what_if_language = st.selectbox("Highest Proficient Language (What-If)", language_options, index=language_options.index(current_base_language), key="what_if_language")

    st.subheader("What-If Happiness Component Scores (Scale 1-10)")

    # COLLECTING 6 WHAT-IF HAPPINESS COMPONENTS (Defaults to current base)
    with w_h_cols[0]:
        what_if_h1 = st.slider("Salary Happiness (W)", 1, 10, base_h1, key="what_if_h1")
    with w_h_cols[1]:
        what_if_h2 = st.slider("Work/Life Balance Happiness (W)", 1, 10, base_h2, key="what_if_h2")
    with w_h_cols[2]:
        what_if_h3 = st.slider("Coworkers Happiness (W)", 1, 10, base_h3, key="what_if_h3")
    with w_h_cols_b[0]:
        what_if_h4 = st.slider("Management Happiness (W)", 1, 10, base_h4, key="what_if_h4")
    with w_h_cols_b[1]:
        what_if_h5 = st.slider("Upward Mobility Happiness (W)", 1, 10, base_h5, key="what_if_h5")
    with w_h_cols_b[2]:
        what_if_h6 = st.slider("Learning New Things Happiness (W)", 1, 10, base_h6, key="what_if_h6")

    # CALCULATE WHAT-IF TOTAL HAPPINESS SCORE (MEAN)
    what_if_happiness_components = [what_if_h1, what_if_h2, what_if_h3, what_if_h4, what_if_h5, what_if_h6]
    what_if_happiness = np.mean(what_if_happiness_components)


    # --- STAGE 2 PREDICTION BUTTON (MAIN PANEL) ---
    if st.button("Predict New Salary for What-If"):
        
        if st.session_state['base_prediction'] is None:
            st.error("Please run the 'Predict Average Salary for Profile' first.")
            st.stop()
            
        # Prepare What-If data dictionary (8 features)
        what_if_data_dict = {
            "Age": what_if_age, 
            "Total Happines": what_if_happiness, 
            "Role": what_if_role,
            "Country": what_if_country,
            "Q12 - Highest Level of Education": what_if_education,
            "Industry": what_if_industry,
            "Favorite Programming Language.1": what_if_language,
            "Gender": what_if_gender,
        }

        # Calculate prediction and breakdown
        what_if_prediction, what_if_breakdown = calculate_feature_contribution(what_if_data_dict)

        # --- Display Results Comparison ---
        st.markdown("### 🚀 What-If Scenario: Potential Gain")
        
        actual_salary = st.session_state['base_actual_salary_value']
        potential_vs_actual_diff = what_if_prediction - actual_salary
        
        # Calculate percentage gain
        if actual_salary != 0:
            potential_vs_actual_perc = (potential_vs_actual_diff / actual_salary) * 100
            delta_text_what_if = f"{potential_vs_actual_perc:.2f}% ({'Potential Gain' if potential_vs_actual_diff >= 0 else 'Shortfall'})"
        else:
            delta_text_what_if = "N/A"
        
        w_res_col1, w_res_col2 = st.columns(2)
        
        with w_res_col1:
            st.metric(
                label="Hypothetical Predicted Salary",
                value=f"{what_if_prediction:.2f} K USD"
            )
        with w_res_col2:
             st.metric(
                label="Potential Gain vs. Your Actual Salary",
                value=f"{potential_vs_actual_diff:.2f} K USD", 
                delta=delta_text_what_if
            )
            
        # --- SHOW CALCULATIONS (REQUESTED) ---
        st.markdown("#### Detailed Calculation Breakdown")
        
        breakdown_df = pd.DataFrame(what_if_breakdown)
        total_sum = breakdown_df['Contribution'].sum()
        
        st.dataframe(
            breakdown_df.style.format({'Contribution': "{:.4f} K USD"}),
            hide_index=True,
            use_container_width=True
        )
        st.write(f"**Total Predicted Salary = Sum of Contributions $\\approx$ {total_sum:.2f} K USD**")

else:
    if st.session_state['base_prediction'] is not None:
        st.info("Check the 'Enable What-If Scenario' box and set new variables to see your potential salary gain.")