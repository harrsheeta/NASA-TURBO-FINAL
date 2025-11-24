import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Page configuration
st.set_page_config(page_title="RUL Prediction", layout="wide")
st.title("🔧 Turbofan Engine RUL Prediction")
st.markdown("Predict the Remaining Useful Life (RUL) of turbofan engines using sensor data")

# Sensor dictionary
Sensor_dictionary = {}
dict_list = [
    "(Fan inlet temperature) (◦R)",
    "(LPC outlet temperature) (◦R)",
    "(HPC outlet temperature) (◦R)",
    "(LPT outlet temperature) (◦R)",
    "(Fan inlet Pressure) (psia)",
    "(bypass-duct pressure) (psia)",
    "(HPC outlet pressure) (psia)",
    "(Physical fan speed) (rpm)",
    "(Physical core speed) (rpm)",
    "(Engine pressure ratio(P50/P2)",
    "(HPC outlet Static pressure) (psia)",
    "(Ratio of fuel flow to Ps30) (pps/psia)",
    "(Corrected fan speed) (rpm)",
    "(Corrected core speed) (rpm)",
    "(Bypass Ratio)",
    "(Burner fuel-air ratio)",
    "(Bleed Enthalpy)",
    "(Required fan speed)",
    "(Required fan conversion speed)",
    "(High-pressure turbines Cool air flow)",
    "(Low-pressure turbines Cool air flow)"
]
i = 1
for x in dict_list:
    Sensor_dictionary['s_'+str(i)] = x
    i += 1

# Load data and model
@st.cache_data
def load_data():
    index_names = ['unit_number', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i+1) for i in range(0, 21)]
    col_names = index_names + setting_names + sensor_names
    
    valid = pd.read_csv('nasa-cmaps/CMaps/test_FD001.txt', sep='\s+', 
                        header=None, index_col=False, names=col_names)
    
    # Load actual RUL values
    rul_actual = pd.read_csv('nasa-cmaps/CMaps/RUL_FD001.txt', 
                             sep='\s+', header=None, names=['RUL'])
    
    return valid, rul_actual

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

# Preprocessing function
def preprocess_data(data):
    index_names = ['unit_number', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    drop_labels2 = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    
    # Keep only sensor columns (drop index, settings, and constant sensors)
    X_processed = data.drop(columns=index_names + setting_names + drop_labels2, 
                           axis=1, errors='ignore')
    
    # Scale the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_processed)
    
    return X_scaled

try:
    valid_data, rul_actual = load_data()
    model = load_model()
    
    # Get sensor names that are actually used (excluding dropped ones)
    drop_labels2 = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    all_sensors = ['s_{}'.format(i+1) for i in range(0, 21)]
    used_sensors = [s for s in all_sensors if s not in drop_labels2]
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📝 Manual Input", "🎲 Random Example", "📊 Full Evaluation"])
    
    # Tab 1: Manual Input
    with tab1:
        st.subheader("Enter Sensor Values")
        st.markdown("Input the sensor readings for prediction:")
        
        # Create columns for sensor inputs
        cols = st.columns(3)
        sensor_values = {}
        
        for idx, sensor in enumerate(used_sensors):
            with cols[idx % 3]:
                # Get sensor description
                sensor_desc = Sensor_dictionary.get(sensor, sensor)
                label = f"{sensor.upper()}\n{sensor_desc}"
                
                sensor_values[sensor] = st.number_input(
                    label, 
                    value=0.0, 
                    format="%.4f",
                    key=f"manual_{sensor}",
                    help=sensor_desc
                )
        
        if st.button("🔮 Predict RUL", type="primary", use_container_width=True):
            # Create dataframe with user inputs
            input_data = pd.DataFrame([sensor_values])
            
            # Scale
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(input_data)
            
            # Predict
            prediction = model.predict(X_scaled)[0]
            
            # Display result
            st.success("### Prediction Complete!")
            st.metric("Predicted Remaining Useful Life (RUL)", f"{prediction:.2f} cycles")
    
    # Tab 2: Random Example
    with tab2:
        st.subheader("Test with Validation Data")
        st.markdown("Randomly select an example from the validation dataset")
        
        if st.button("🎲 Try Random Example", type="primary", use_container_width=True):
            # Get last cycle for each unit (these are the test points)
            last_cycles = valid_data.groupby('unit_number').tail(1).reset_index(drop=True)
            
            # Select random unit
            random_idx = np.random.randint(0, len(last_cycles))
            random_unit = last_cycles.iloc[random_idx]
            unit_num = int(random_unit['unit_number'])
            
            # Preprocess the single unit's last cycle
            unit_last_cycle = last_cycles.iloc[[random_idx]]
            X_processed = preprocess_data(unit_last_cycle)
            
            # Predict
            prediction = model.predict(X_processed)[0]
            
            # Get actual RUL (unit_number - 1 because RUL file is 0-indexed)
            actual_rul = rul_actual.iloc[unit_num - 1]['RUL']
            
            # Calculate error percentage for color coding
            error = abs(prediction - actual_rul)
            error_pct = (error / max(actual_rul, 1)) * 100  # Avoid division by zero
            
            # Determine color based on accuracy (green if close, red if far)
            if error_pct <= 5:
                pred_color = "#00ff00"  # Bright green
            elif error_pct <= 10:
                pred_color = "#7fff00"  # Yellow-green
            elif error_pct <= 20:
                pred_color = "#ffff00"  # Yellow
            elif error_pct <= 30:
                pred_color = "#ffa500"  # Orange
            else:
                pred_color = "#ff4500"  # Red-orange
            
            # Create two columns
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.info(f"### 📋 Unit #{unit_num}")
                st.write(f"**Total Cycles Run:** {int(random_unit['time_cycles'])}")
                st.write("")
                
                # Show sensor values in expandable section
                with st.expander("📈 View Sensor Readings", expanded=False):
                    sensor_df = pd.DataFrame({
                        'Sensor': used_sensors,
                        'Description': [Sensor_dictionary.get(s, s) for s in used_sensors],
                        'Value': [f"{random_unit[s]:.4f}" for s in used_sensors]
                    })
                    st.dataframe(sensor_df, use_container_width=True, hide_index=True)
            
            with col_right:
                st.markdown("### 🎯 Prediction Results")
                
                # Create results table
                results_data = {
                    'Metric': ['Actual RUL', 'Predicted RUL', 'Absolute Error', 'Error %'],
                    'Value': [
                        f"{actual_rul:.2f} cycles",
                        f"{prediction:.2f} cycles",
                        f"{error:.2f} cycles",
                        f"{error_pct:.2f}%"
                    ]
                }
                results_df = pd.DataFrame(results_data)
                
                # Display as styled table
                st.markdown(f"""
                <style>
                .results-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                .results-table td, .results-table th {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                .actual-row {{
                    background-color: #00ff0020;
                    font-weight: bold;
                }}
                .predicted-row {{
                    background-color: {pred_color}20;
                    font-weight: bold;
                }}
                </style>
                <table class="results-table">
                    <tr class="actual-row">
                        <td>✅ Actual RUL</td>
                        <td style="color: #00ff00; font-size: 1.2em;">{actual_rul:.2f} cycles</td>
                    </tr>
                    <tr class="predicted-row">
                        <td>🔮 Predicted RUL</td>
                        <td style="color: {pred_color}; font-size: 1.2em;">{prediction:.2f} cycles</td>
                    </tr>
                    <tr>
                        <td>📊 Absolute Error</td>
                        <td>{error:.2f} cycles</td>
                    </tr>
                    <tr>
                        <td>📈 Error Percentage</td>
                        <td>{error_pct:.2f}%</td>
                    </tr>
                </table>
                """, unsafe_allow_html=True)
                
                # Accuracy indicator
                if error_pct <= 10:
                    st.success("🎉 Excellent prediction!")
                elif error_pct <= 20:
                    st.info("👍 Good prediction")
                elif error_pct <= 30:
                    st.warning("⚠️ Moderate accuracy")
                else:
                    st.error("❌ Large deviation")
    
    # Tab 3: Full Evaluation
    with tab3:
        st.subheader("Evaluate Model on Entire Validation Dataset")
        st.markdown("Compute performance metrics across all validation units")
        
        if st.button("🚀 Run Full Evaluation", type="primary", use_container_width=True):
            with st.spinner("Processing all validation units..."):
                # Get last cycle for each unit
                last_cycles = valid_data.groupby('unit_number').tail(1).reset_index(drop=True)
                
                # Preprocess all last cycles at once
                X_processed = preprocess_data(last_cycles)
                
                # Get predictions for all units
                all_predictions = model.predict(X_processed)
                all_actuals = rul_actual['RUL'].values
                
                # Convert to numpy arrays
                y_true = np.array(all_actuals)
                y_pred = np.array(all_predictions)
                
                # Calculate metrics
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                # Display metrics in columns
                st.success("### 📊 Model Performance Metrics")
                
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric("RMSE", f"{rmse:.2f}", help="Root Mean Squared Error")
                
                with metric_cols[1]:
                    st.metric("MAE", f"{mae:.2f}", help="Mean Absolute Error")
                
                with metric_cols[2]:
                    st.metric("R² Score", f"{r2:.4f}", help="Coefficient of Determination")
                
                with metric_cols[3]:
                    st.metric("Total Units", len(y_true))
                
                # Create comparison dataframe
                st.markdown("---")
                st.markdown("### 📋 Detailed Results")
                
                comparison_df = pd.DataFrame({
                    'Unit': range(1, len(y_true) + 1),
                    'Actual RUL': y_true,
                    'Predicted RUL': y_pred,
                    'Error': np.abs(y_true - y_pred),
                    'Error %': (np.abs(y_true - y_pred) / np.maximum(y_true, 1)) * 100
                })
                
                # Add color coding column
                def get_accuracy_label(error_pct):
                    if error_pct <= 10:
                        return "🟢 Excellent"
                    elif error_pct <= 20:
                        return "🟡 Good"
                    elif error_pct <= 30:
                        return "🟠 Moderate"
                    else:
                        return "🔴 Poor"
                
                comparison_df['Accuracy'] = comparison_df['Error %'].apply(get_accuracy_label)
                
                # Format numeric columns
                comparison_df['Actual RUL'] = comparison_df['Actual RUL'].apply(lambda x: f"{x:.2f}")
                comparison_df['Predicted RUL'] = comparison_df['Predicted RUL'].apply(lambda x: f"{x:.2f}")
                comparison_df['Error'] = comparison_df['Error'].apply(lambda x: f"{x:.2f}")
                comparison_df['Error %'] = comparison_df['Error %'].apply(lambda x: f"{x:.2f}%")
                
                # Display table
                st.dataframe(comparison_df, use_container_width=True, hide_index=True, height=400)
                
                # Summary statistics
                st.markdown("---")
                st.markdown("### 📈 Summary Statistics")
                
                summary_cols = st.columns(3)
                
                with summary_cols[0]:
                    excellent = len([x for x in comparison_df['Accuracy'] if '🟢' in x])
                    st.metric("Excellent Predictions", f"{excellent} ({excellent/len(y_true)*100:.1f}%)")
                
                with summary_cols[1]:
                    good = len([x for x in comparison_df['Accuracy'] if '🟡' in x])
                    st.metric("Good Predictions", f"{good} ({good/len(y_true)*100:.1f}%)")
                
                with summary_cols[2]:
                    poor = len([x for x in comparison_df['Accuracy'] if '🔴' in x or '🟠' in x])
                    st.metric("Needs Improvement", f"{poor} ({poor/len(y_true)*100:.1f}%)")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application predicts the **Remaining Useful Life (RUL)** of turbofan engines 
        using NASA's C-MAPSS dataset.
        
        **Features:**
        - Manual sensor input for custom predictions
        - Random validation examples with actual vs predicted comparison
        
        **Model:** Support Vector Regression (SVR)
        """)
        
        st.markdown("---")
        st.header("📊 Sensors Used")
        
        # Create expandable sensor list with descriptions
        for sensor in used_sensors:
            with st.expander(f"**{sensor.upper()}**"):
                st.write(Sensor_dictionary.get(sensor, "No description available"))
        
        st.markdown("---")
        st.caption("💡 Hover over input fields for sensor descriptions")

except FileNotFoundError as e:
    st.error(f"⚠️ Error loading files: {e}")
    st.info("Please ensure the following files are in the correct location:\n"
            "- nasa-cmaps/CMaps/test_FD001.txt\n"
            "- nasa-cmaps/CMaps/RUL_FD001.txt\n"
            "- SVR_final.pkl")
except Exception as e:
    st.error(f"⚠️ An error occurred: {e}")
    st.exception(e)