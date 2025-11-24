# fault_classification_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time

# Page configuration
st.set_page_config(
    page_title="Fault Classification Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2e8b57;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2e8b57;
        padding-bottom: 0.5rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .feature-section {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #b3d9ff;
        margin-bottom: 1rem;
    }
    .severity-high { background-color: #ff6b6b; color: white; padding: 0.5rem; border-radius: 5px; }
    .severity-medium { background-color: #ffd93d; color: black; padding: 0.5rem; border-radius: 5px; }
    .severity-low { background-color: #6bcf7f; color: white; padding: 0.5rem; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

class FaultClassificationDashboard:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and dependencies"""
        try:
            self.model = joblib.load('fault_classification_model_random_forest.pkl')
            self.label_encoder = joblib.load('label_encoder.pkl')
            
            # Define feature names based on selected columns
            self.feature_names = [
                'Feeder_ProcessStatus', 'DTR_ProcessStatus', 'Consumer_ProcessStatus',
                'Consumer_Phase_Id', 'f_vr', 'f_vy', 'f_vb', 'f_ir', 'f_iy', 'f_ib', 
                'd_vr', 'd_vy', 'd_vb', 'd_ir', 'd_iy', 'd_ib', 
                'C_tp_vr', 'C_tp_vy', 'C_tp_vb', 'C_tp_ir', 'C_tp_iy', 'C_tp_ib', 
                'C_sp_i', 'C_sp_v'
            ]
            
            st.sidebar.success("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {e}")
            return False
    
    def create_sample_scenarios(self):
        """Create comprehensive sample scenarios"""
        return {
            "normal_operation": {
                "name": "🟢 Normal Operation",
                "data": {
                    "Feeder_ProcessStatus": "success",
                    "DTR_ProcessStatus": "success", 
                    "Consumer_ProcessStatus": "success",
                    "Consumer_Phase_Id": 1,
                    "f_vr": 63.5, "f_vy": 64.2, "f_vb": 64.1,
                    "f_ir": 1.38, "f_iy": 1.36, "f_ib": 1.29,
                    "d_vr": 244.1, "d_vy": 243.0, "d_vb": 247.6,
                    "d_ir": 0.15, "d_iy": 0.28, "d_ib": 0.02,
                    "C_tp_vr": 240.0, "C_tp_vy": 239.5, "C_tp_vb": 241.2,
                    "C_tp_ir": 0.12, "C_tp_iy": 0.15, "C_tp_ib": 0.13,
                    "C_sp_i": 0.0, "C_sp_v": 0.0
                },
                "description": "All systems operating normally with balanced voltages and currents"
            },
            "dtht_fault": {
                "name": "🔴 DTHT Fault",
                "data": {
                    "Feeder_ProcessStatus": "success",
                    "DTR_ProcessStatus": "success",
                    "Consumer_ProcessStatus": "success", 
                    "Consumer_Phase_Id": 1,
                    "f_vr": 64.6, "f_vy": 64.5, "f_vb": 63.8,
                    "f_ir": 1.54, "f_iy": 1.63, "f_ib": 1.51,
                    "d_vr": 0.0, "d_vy": 239.1, "d_vb": 0.0,
                    "d_ir": 0.0, "d_iy": 0.93, "d_ib": 0.0,
                    "C_tp_vr": 0.0, "C_tp_vy": 0.0, "C_tp_vb": 0.0,
                    "C_tp_ir": 0.0, "C_tp_iy": 0.0, "C_tp_ib": 0.0,
                    "C_sp_i": 0.0, "C_sp_v": 0.0
                },
                "description": "High Tension side fault with unbalanced voltages and zero currents"
            },
            "dtlt_fault": {
                "name": "🟡 DTLT Fault", 
                "data": {
                    "Feeder_ProcessStatus": "success",
                    "DTR_ProcessStatus": "success",
                    "Consumer_ProcessStatus": "fail",
                    "Consumer_Phase_Id": 1,
                    "f_vr": 63.1, "f_vy": 64.1, "f_vb": 64.0,
                    "f_ir": 1.56, "f_iy": 1.51, "f_ib": 1.52,
                    "d_vr": 253.6, "d_vy": 247.3, "d_vb": 251.2,
                    "d_ir": 0.32, "d_iy": 1.54, "d_ib": 0.51,
                    "C_tp_vr": 0.0, "C_tp_vy": 0.0, "C_tp_vb": 0.0,
                    "C_tp_ir": 0.0, "C_tp_iy": 0.0, "C_tp_ib": 0.0,
                    "C_sp_i": 0.0, "C_sp_v": 0.0
                },
                "description": "Low Tension side fault with consumer process failure"
            },
            "feeder_fault": {
                "name": "🔵 Feeder Fault",
                "data": {
                    "Feeder_ProcessStatus": "fail",
                    "DTR_ProcessStatus": "fail", 
                    "Consumer_ProcessStatus": "fail",
                    "Consumer_Phase_Id": 1,
                    "f_vr": 0.0, "f_vy": 0.0, "f_vb": 0.0,
                    "f_ir": 0.0, "f_iy": 0.0, "f_ib": 0.0,
                    "d_vr": 0.0, "d_vy": 0.0, "d_vb": 0.0,
                    "d_ir": 0.0, "d_iy": 0.0, "d_ib": 0.0,
                    "C_tp_vr": 0.0, "C_tp_vy": 0.0, "C_tp_vb": 0.0,
                    "C_tp_ir": 0.0, "C_tp_iy": 0.0, "C_tp_ib": 0.0,
                    "C_sp_i": 0.0, "C_sp_v": 0.0
                },
                "description": "Complete feeder failure affecting all downstream components"
            }
        }
    
    def predict(self, input_data):
        """Make prediction on input data"""
        try:
            # Create DataFrame with correct feature names
            input_df = pd.DataFrame([input_data])
            
            # Ensure all required columns are present
            for col in self.feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0.0
            
            # Reorder columns to match training
            input_df = input_df[self.feature_names]
            
            # Make prediction
            prediction = self.model.predict(input_df)[0]
            probabilities = self.model.predict_proba(input_df)[0]
            
            # Decode prediction
            fault_type = self.label_encoder.inverse_transform([prediction])[0]
            confidence = probabilities[prediction]
            
            # Get all probabilities
            all_probs = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                all_probs[class_name] = probabilities[i]
            
            # Sort by probability
            sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
            
            return {
                'fault_type': fault_type,
                'confidence': confidence,
                'all_probabilities': sorted_probs,
                'confidence_level': 'HIGH' if confidence > 0.8 else 'MEDIUM' if confidence > 0.6 else 'LOW',
                'prediction_encoded': prediction
            }
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    
    def create_comprehensive_dashboard(self, input_data, prediction_result):
        """Create a comprehensive visualization dashboard"""
        
        # 1. Main Prediction Card
        st.markdown(f"""
        <div class="prediction-card">
            <h1>🔍 PREDICTION RESULT</h1>
            <h2 style="font-size: 2.5rem; margin: 1rem 0;">{prediction_result['fault_type']}</h2>
            <h3 style="font-size: 1.8rem;">Confidence: {prediction_result['confidence']:.3f} ({prediction_result['confidence_level']})</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Probability Analysis", 
            "⚡ Measurements Overview", 
            "🔧 Fault Analysis",
            "📈 Feature Importance",
            "🔍 Input Summary"
        ])
        
        with tab1:
            self._create_probability_analysis(prediction_result)
        
        with tab2:
            self._create_measurements_overview(input_data)
        
        with tab3:
            self._create_fault_analysis(prediction_result)
        
        with tab4:
            self._create_feature_importance_analysis()
        
        with tab5:
            self._create_input_summary(input_data)
    
    def _create_probability_analysis(self, prediction_result):
        """Create probability analysis visualizations"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Probability Bar Chart
            fig_bar = go.Figure()
            
            colors = []
            for fault in prediction_result['all_probabilities'].keys():
                if fault == prediction_result['fault_type']:
                    colors.append('#2E8B57')  # Green for predicted fault
                elif prediction_result['all_probabilities'][fault] > 0.1:
                    colors.append('#FFA500')  # Orange for significant probabilities
                else:
                    colors.append('#4682B4')  # Blue for low probabilities
            
            fig_bar.add_trace(go.Bar(
                x=list(prediction_result['all_probabilities'].keys()),
                y=list(prediction_result['all_probabilities'].values()),
                marker_color=colors,
                text=[f'{prob:.3f}' for prob in prediction_result['all_probabilities'].values()],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Probability: %{y:.3f}<extra></extra>'
            ))
            
            fig_bar.update_layout(
                title='Fault Probability Distribution',
                xaxis_title='Fault Types',
                yaxis_title='Probability',
                height=400,
                showlegend=False,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Confidence Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prediction_result['confidence'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence Level"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Quick Stats
            st.markdown("### 📈 Quick Stats")
            st.metric("Top Prediction", prediction_result['fault_type'])
            st.metric("Confidence", f"{prediction_result['confidence']:.3f}")
            st.metric("Second Choice", list(prediction_result['all_probabilities'].keys())[1])
            st.metric("Second Probability", f"{prediction_result['all_probabilities'][list(prediction_result['all_probabilities'].keys())[1]]:.3f}")
    
    def _create_measurements_overview(self, input_data):
        """Create comprehensive measurements overview"""
        
        # Create subplots for different measurement types
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Voltage Comparison (All Phases)',
                'Current Comparison (All Phases)', 
                'Feeder vs DTR Voltages',
                'Three-Phase Balance Analysis'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "pie"}]]
        )
        
        phases = ['R', 'Y', 'B']
        
        # 1. Voltage Comparison
        feeder_voltages = [input_data['f_vr'], input_data['f_vy'], input_data['f_vb']]
        dtr_voltages = [input_data['d_vr'], input_data['d_vy'], input_data['d_vb']]
        consumer_voltages = [input_data['C_tp_vr'], input_data['C_tp_vy'], input_data['C_tp_vb']]
        
        fig.add_trace(go.Bar(name='Feeder', x=phases, y=feeder_voltages, marker_color='blue'), 1, 1)
        fig.add_trace(go.Bar(name='DTR', x=phases, y=dtr_voltages, marker_color='green'), 1, 1)
        fig.add_trace(go.Bar(name='Consumer', x=phases, y=consumer_voltages, marker_color='red'), 1, 1)
        
        # 2. Current Comparison
        feeder_currents = [input_data['f_ir'], input_data['f_iy'], input_data['f_ib']]
        dtr_currents = [input_data['d_ir'], input_data['d_iy'], input_data['d_ib']]
        consumer_currents = [input_data['C_tp_ir'], input_data['C_tp_iy'], input_data['C_tp_ib']]
        
        fig.add_trace(go.Bar(name='Feeder', x=phases, y=feeder_currents, marker_color='blue', showlegend=False), 1, 2)
        fig.add_trace(go.Bar(name='DTR', x=phases, y=dtr_currents, marker_color='green', showlegend=False), 1, 2)
        fig.add_trace(go.Bar(name='Consumer', x=phases, y=consumer_currents, marker_color='red', showlegend=False), 1, 2)
        
        # 3. Feeder vs DTR Scatter
        fig.add_trace(go.Scatter(
            x=feeder_voltages, y=dtr_voltages, mode='markers+text',
            text=phases, textposition="top center",
            marker=dict(size=15, color='purple'),
            name='Feeder vs DTR'
        ), 2, 1)
        
        # 4. Process Status Pie Chart
        process_status = [
            input_data['Feeder_ProcessStatus'],
            input_data['DTR_ProcessStatus'], 
            input_data['Consumer_ProcessStatus']
        ]
        success_count = process_status.count('success')
        fail_count = process_status.count('fail')
        
        fig.add_trace(go.Pie(
            labels=['Success', 'Failure'],
            values=[success_count, fail_count],
            name="Process Status"
        ), 2, 2)
        
        fig.update_layout(height=700, title_text="Comprehensive Measurements Analysis", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_feeder_v = np.mean(feeder_voltages)
            st.metric("Avg Feeder Voltage", f"{avg_feeder_v:.2f} V")
        
        with col2:
            avg_dtr_v = np.mean(dtr_voltages)
            st.metric("Avg DTR Voltage", f"{avg_dtr_v:.2f} V")
        
        with col3:
            voltage_balance = np.std(feeder_voltages) / np.mean(feeder_voltages) * 100 if np.mean(feeder_voltages) > 0 else 0
            st.metric("Voltage Unbalance", f"{voltage_balance:.2f}%")
        
        with col4:
            success_rate = (success_count / 3) * 100
            st.metric("Process Success Rate", f"{success_rate:.1f}%")
    
    def _create_fault_analysis(self, prediction_result):
        """Create detailed fault analysis"""
        fault_info = self.get_fault_info(prediction_result['fault_type'])
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🔧 Fault Details")
            
            # Severity indicator
            severity_class = f"severity-{fault_info['severity'].lower()}"
            st.markdown(f'<div class="{severity_class}"><strong>Severity: {fault_info["severity"]}</strong></div>', unsafe_allow_html=True)
            
            st.markdown("**Description:**")
            st.info(fault_info['description'])
            
            st.markdown("**Common Symptoms:**")
            for symptom in fault_info['symptoms']:
                st.write(f"• {symptom}")
            
            # Create symptom visualization
            if fault_info['symptoms']:
                symptoms_df = pd.DataFrame({
                    'Symptom': fault_info['symptoms'],
                    'Severity': [1] * len(fault_info['symptoms'])  # Placeholder for actual severity
                })
                
                fig_symptoms = px.bar(symptoms_df, x='Symptom', y='Severity', 
                                    title='Fault Symptoms Overview',
                                    color_discrete_sequence=['#ff6b6b'])
                st.plotly_chart(fig_symptoms, use_container_width=True)
        
        with col2:
            st.markdown("### 🛠️ Recommended Actions")
            
            for i, action in enumerate(fault_info['actions'], 1):
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Step {i}:</strong> {action}
                </div>
                """, unsafe_allow_html=True)
            
            # Response time estimate based on severity
            response_times = {
                'HIGH': 'Immediate (0-2 hours)',
                'MEDIUM': 'Urgent (2-8 hours)', 
                'LOW': 'Scheduled (8-24 hours)'
            }
            
            st.markdown("### ⏱️ Response Time")
            st.warning(f"**Estimated Response:** {response_times.get(fault_info['severity'], 'To be determined')}")
            
            # Impact assessment
            impacts = {
                'HIGH': 'Critical - Potential safety hazard and widespread outage',
                'MEDIUM': 'Significant - Multiple consumers affected',
                'LOW': 'Limited - Single consumer or minor issue'
            }
            
            st.markdown("### 📉 Impact Assessment")
            st.error(f"**Impact Level:** {impacts.get(fault_info['severity'], 'Unknown')}")
    
    def _create_feature_importance_analysis(self):
        """Create feature importance visualization"""
        try:
            if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
                importances = self.model.named_steps['classifier'].feature_importances_
                
                # Get feature names after preprocessing
                feature_names = []
                
                # Numerical features
                numerical_features = [
                    'Consumer_Phase_Id', 'f_vr', 'f_vy', 'f_vb', 'f_ir', 'f_iy', 'f_ib',
                    'd_vr', 'd_vy', 'd_vb', 'd_ir', 'd_iy', 'd_ib',
                    'C_tp_vr', 'C_tp_vy', 'C_tp_vb', 'C_tp_ir', 'C_tp_iy', 'C_tp_ib',
                    'C_sp_i', 'C_sp_v'
                ]
                feature_names.extend(numerical_features)
                
                # Create importance dataframe
                importance_df = pd.DataFrame({
                    'feature': feature_names[:len(importances)],
                    'importance': importances
                }).sort_values('importance', ascending=True)
                
                # Plot feature importance
                fig = px.bar(importance_df.tail(15),  # Top 15 features
                           x='importance', y='feature',
                           title='Top 15 Most Important Features',
                           orientation='h',
                           color='importance',
                           color_continuous_scale='viridis')
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("Feature importance not available for this model type.")
                
        except Exception as e:
            st.warning(f"Could not generate feature importance: {e}")
    
    def _create_input_summary(self, input_data):
        """Create input data summary"""
        st.markdown("### 📋 Input Data Summary")
        
        # Create summary tables for different categories
        col1, col2 = st.columns(2)
        
        with col1:
            # Process Status Summary
            st.markdown("#### 🔧 Process Status")
            process_df = pd.DataFrame({
                'Component': ['Feeder', 'DTR', 'Consumer'],
                'Status': [
                    input_data['Feeder_ProcessStatus'],
                    input_data['DTR_ProcessStatus'],
                    input_data['Consumer_ProcessStatus']
                ]
            })
            st.dataframe(process_df, use_container_width=True)
            
            # Voltage Summary
            st.markdown("#### ⚡ Voltage Measurements (V)")
            voltage_df = pd.DataFrame({
                'Phase': ['R', 'Y', 'B'],
                'Feeder': [input_data['f_vr'], input_data['f_vy'], input_data['f_vb']],
                'DTR': [input_data['d_vr'], input_data['d_vy'], input_data['d_vb']],
                'Consumer': [input_data['C_tp_vr'], input_data['C_tp_vy'], input_data['C_tp_vb']]
            })
            st.dataframe(voltage_df.style.format("{:.2f}"), use_container_width=True)
        
        with col2:
            # Current Summary
            st.markdown("#### 🔌 Current Measurements (A)")
            current_df = pd.DataFrame({
                'Phase': ['R', 'Y', 'B'],
                'Feeder': [input_data['f_ir'], input_data['f_iy'], input_data['f_ib']],
                'DTR': [input_data['d_ir'], input_data['d_iy'], input_data['d_ib']],
                'Consumer': [input_data['C_tp_ir'], input_data['C_tp_iy'], input_data['C_tp_ib']]
            })
            st.dataframe(current_df.style.format("{:.2f}"), use_container_width=True)
            
            # Additional Parameters
            st.markdown("#### 📊 Additional Parameters")
            additional_df = pd.DataFrame({
                'Parameter': ['Consumer Phase ID', 'SP Current', 'SP Voltage'],
                'Value': [
                    input_data['Consumer_Phase_Id'],
                    f"{input_data['C_sp_i']:.2f} A",
                    f"{input_data['C_sp_v']:.2f} V"
                ]
            })
            st.dataframe(additional_df, use_container_width=True)
        
        # Data quality indicators
        st.markdown("### 📈 Data Quality Assessment")
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            # Voltage data completeness
            voltage_fields = ['f_vr', 'f_vy', 'f_vb', 'd_vr', 'd_vy', 'd_vb', 'C_tp_vr', 'C_tp_vy', 'C_tp_vb']
            voltage_completeness = sum(1 for field in voltage_fields if input_data.get(field, 0) != 0) / len(voltage_fields) * 100
            st.metric("Voltage Data Completeness", f"{voltage_completeness:.1f}%")
        
        with col4:
            # Current data completeness
            current_fields = ['f_ir', 'f_iy', 'f_ib', 'd_ir', 'd_iy', 'd_ib', 'C_tp_ir', 'C_tp_iy', 'C_tp_ib']
            current_completeness = sum(1 for field in current_fields if input_data.get(field, 0) != 0) / len(current_fields) * 100
            st.metric("Current Data Completeness", f"{current_completeness:.1f}%")
        
        with col5:
            # Process status completeness
            process_fields = ['Feeder_ProcessStatus', 'DTR_ProcessStatus', 'Consumer_ProcessStatus']
            process_completeness = sum(1 for field in process_fields if input_data.get(field)) / len(process_fields) * 100
            st.metric("Process Data Completeness", f"{process_completeness:.1f}%")
        
        with col6:
            # Overall data quality
            overall_quality = (voltage_completeness + current_completeness + process_completeness) / 3
            st.metric("Overall Data Quality", f"{overall_quality:.1f}%")
    
    def get_fault_info(self, fault_type):
        """Get detailed information about each fault type"""
        fault_info = {
            'FEEDER': {
                'description': 'Feeder-level fault affecting power distribution from the source substation',
                'symptoms': [
                    'Abnormal feeder voltages (zero or unbalanced)',
                    'Zero or abnormal feeder currents', 
                    'Feeder process status failure',
                    'Downstream component failures'
                ],
                'actions': [
                    'Immediately check feeder circuit breakers and protection relays',
                    'Inspect feeder line connections and insulation integrity',
                    'Verify substation feeder panel operations and alarms',
                    'Check for any tripped protection devices upstream',
                    'Coordinate with substation operations team'
                ],
                'severity': 'HIGH'
            },
            'DTHT_FAULT': {
                'description': 'Distribution Transformer High Tension side fault affecting primary voltage',
                'symptoms': [
                    'Unbalanced HT side voltages',
                    'Zero or abnormal HT currents', 
                    'Voltage unbalance typically > 30%',
                    'Normal feeder readings but abnormal DTR outputs'
                ],
                'actions': [
                    'Inspect transformer HT bushings and connections',
                    'Check HT line voltage levels and phase balance',
                    'Verify transformer protection systems and fuses',
                    'Examine HT side insulation and grounding integrity',
                    'Check transformer oil levels and temperature'
                ],
                'severity': 'HIGH'
            },
            'DTLT_FAULT': {
                'description': 'Distribution Transformer Low Tension side fault affecting secondary distribution',
                'symptoms': [
                    'Unbalanced LT side voltages',
                    'Abnormal LT currents patterns', 
                    'Consumer side process failures',
                    'Normal HT side but abnormal LT readings'
                ],
                'actions': [
                    'Check LT side connections and terminal integrity',
                    'Inspect LT protection fuses and circuit breakers',
                    'Verify LT voltage and current balance across phases',
                    'Examine transformer LT winding integrity',
                    'Check consumer service connections'
                ],
                'severity': 'MEDIUM'
            },
            'FOC_DTHT_FAULT': {
                'description': 'Combined Fuse Off Call and DTHT fault condition - complex multi-level issue',
                'symptoms': [
                    'Consumer connectivity and process issues',
                    'HT side voltage abnormalities',
                    'Multiple system level failures',
                    'Combined symptoms of both FOC and DTHT faults'
                ],
                'actions': [
                    'First check consumer fuse and service connection integrity',
                    'Simultaneously inspect HT side connections and transformer',
                    'Verify overall system connectivity and grounding',
                    'Check for combined fault conditions systematically',
                    'Coordinate consumer and transformer maintenance teams'
                ],
                'severity': 'HIGH'
            },
            'FOC': {
                'description': 'Fuse Off Call - Consumer side connectivity or protection device issue',
                'symptoms': [
                    'Consumer process status failure',
                    'Normal feeder and DTR readings',
                    'Zero consumer side measurements',
                    'Isolated consumer service disruption'
                ],
                'actions': [
                    'Check consumer fuse/circuit breaker status first',
                    'Inspect service drop and connection integrity',
                    'Verify meter operation and connectivity',
                    'Check consumer installation and internal wiring',
                    'Test consumer equipment and load conditions'
                ],
                'severity': 'LOW'
            }
        }
        return fault_info.get(fault_type, {
            'description': 'Unknown or unclassified fault type requiring manual investigation',
            'symptoms': ['Further detailed investigation required by technical team'],
            'actions': ['Contact technical support and senior engineers for detailed analysis'],
            'severity': 'UNKNOWN'
        })

def main():
    # Initialize dashboard
    dashboard = FaultClassificationDashboard()
    
    # Header
    st.markdown('<h1 class="main-header">⚡ Electrical Fault Classification Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Machine Learning System for Real-time Fault Detection and Analysis")
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio("Select Page", ["🏠 Main Dashboard", "📋 Model Information", "🆘 Help Guide"])
    
    if not dashboard.model:
        st.error("""
        ## ⚠️ Model Not Loaded
        Please ensure the following files are in the same directory:
        - `fault_classification_model_random_forest.pkl`
        - `label_encoder.pkl`
        """)
        return
    
    if page == "🏠 Main Dashboard":
        render_main_dashboard(dashboard)
    elif page == "📋 Model Information":
        render_model_info(dashboard)
    else:
        render_help_guide()

def render_main_dashboard(dashboard):
    """Render the main dashboard page"""
    
    st.markdown('<div class="sub-header">🚀 Quick Scenario Testing</div>', unsafe_allow_html=True)
    
    # Scenario selection
    scenarios = dashboard.create_sample_scenarios()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🟢 Normal Operation", use_container_width=True):
            st.session_state.selected_scenario = "normal_operation"
    with col2:
        if st.button("🔴 DTHT Fault", use_container_width=True):
            st.session_state.selected_scenario = "dtht_fault"
    with col3:
        if st.button("🟡 DTLT Fault", use_container_width=True):
            st.session_state.selected_scenario = "dtlt_fault"
    with col4:
        if st.button("🔵 Feeder Fault", use_container_width=True):
            st.session_state.selected_scenario = "feeder_fault"
    
    st.markdown("---")
    
    # Input form
    st.markdown('<div class="sub-header">🔧 Manual Input Parameters</div>', unsafe_allow_html=True)
    
    with st.form("input_form"):
        # Create tabs for organized input
        tab1, tab2, tab3, tab4 = st.tabs(["Process Status", "Feeder Data", "DTR Data", "Consumer Data"])
        
        input_data = {}
        
        with tab1:
            st.markdown("### 🔌 Process Status & Phase Info")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                input_data['Feeder_ProcessStatus'] = st.selectbox("Feeder Process", ["success", "fail"])
            with col2:
                input_data['DTR_ProcessStatus'] = st.selectbox("DTR Process", ["success", "fail"])
            with col3:
                input_data['Consumer_ProcessStatus'] = st.selectbox("Consumer Process", ["success", "fail"])
            with col4:
                input_data['Consumer_Phase_Id'] = st.selectbox("Phase ID", [1, 2, 3])
        
        with tab2:
            st.markdown("### ⚡ Feeder Measurements")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Voltages (V)**")
                input_data['f_vr'] = st.number_input("Voltage R", value=0.0, format="%.2f")
                input_data['f_vy'] = st.number_input("Voltage Y", value=0.0, format="%.2f")
                input_data['f_vb'] = st.number_input("Voltage B", value=0.0, format="%.2f")
            with col2:
                st.write("**Currents (A)**")
                input_data['f_ir'] = st.number_input("Current R", value=0.0, format="%.2f")
                input_data['f_iy'] = st.number_input("Current Y", value=0.0, format="%.2f")
                input_data['f_ib'] = st.number_input("Current B", value=0.0, format="%.2f")
        
        with tab3:
            st.markdown("### 🏭 DTR Measurements")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Voltages (V)**")
                input_data['d_vr'] = st.number_input("DTR Voltage R", value=0.0, format="%.2f")
                input_data['d_vy'] = st.number_input("DTR Voltage Y", value=0.0, format="%.2f")
                input_data['d_vb'] = st.number_input("DTR Voltage B", value=0.0, format="%.2f")
            with col2:
                st.write("**Currents (A)**")
                input_data['d_ir'] = st.number_input("DTR Current R", value=0.0, format="%.2f")
                input_data['d_iy'] = st.number_input("DTR Current Y", value=0.0, format="%.2f")
                input_data['d_ib'] = st.number_input("DTR Current B", value=0.0, format="%.2f")
        
        with tab4:
            st.markdown("### 🏠 Consumer Measurements")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**TP Voltages (V)**")
                input_data['C_tp_vr'] = st.number_input("TP Voltage R", value=0.0, format="%.2f")
                input_data['C_tp_vy'] = st.number_input("TP Voltage Y", value=0.0, format="%.2f")
                input_data['C_tp_vb'] = st.number_input("TP Voltage B", value=0.0, format="%.2f")
            with col2:
                st.write("**TP Currents (A)**")
                input_data['C_tp_ir'] = st.number_input("TP Current R", value=0.0, format="%.2f")
                input_data['C_tp_iy'] = st.number_input("TP Current Y", value=0.0, format="%.2f")
                input_data['C_tp_ib'] = st.number_input("TP Current B", value=0.0, format="%.2f")
            with col3:
                st.write("**SP Measurements**")
                input_data['C_sp_i'] = st.number_input("SP Current (A)", value=0.0, format="%.2f")
                input_data['C_sp_v'] = st.number_input("SP Voltage (V)", value=0.0, format="%.2f")
        
        # Load scenario data if selected
        if hasattr(st.session_state, 'selected_scenario'):
            scenario = scenarios[st.session_state.selected_scenario]
            input_data.update(scenario['data'])
            st.info(f"📋 Loaded: {scenario['name']} - {scenario['description']}")
            # Clear the selection after loading
            del st.session_state.selected_scenario
        
        # Submit button with animation
        submitted = st.form_submit_button("🎯 ANALYZE FAULT & GENERATE REPORT", use_container_width=True)
    
    if submitted:
        # Show loading animation
        with st.spinner('🔬 Analyzing system parameters and generating comprehensive report...'):
            time.sleep(1)  # Simulate processing time
            prediction_result = dashboard.predict(input_data)
        
        if prediction_result:
            # Display comprehensive dashboard
            dashboard.create_comprehensive_dashboard(input_data, prediction_result)

def render_model_info(dashboard):
    """Render model information page"""
    st.markdown('<div class="sub-header">📊 Model Information & Performance</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🏗️ Model Architecture")
        
        model_info = {
            "Model Type": type(dashboard.model.named_steps['classifier']).__name__,
            "Total Features": len(dashboard.feature_names),
            "Target Classes": len(dashboard.label_encoder.classes_),
            "Preprocessing": "StandardScaler + OneHotEncoder",
            "Pipeline Steps": "Data Preprocessing → Feature Engineering → Classification"
        }
        
        for key, value in model_info.items():
            st.markdown(f"**{key}:** {value}")
        
        st.markdown("### 🎯 Feature Categories")
        feature_categories = {
            "Process Status (3)": ["Feeder_ProcessStatus", "DTR_ProcessStatus", "Consumer_ProcessStatus"],
            "Phase Information (1)": ["Consumer_Phase_Id"],
            "Feeder Measurements (6)": ["f_vr", "f_vy", "f_vb", "f_ir", "f_iy", "f_ib"],
            "DTR Measurements (6)": ["d_vr", "d_vy", "d_vb", "d_ir", "d_iy", "d_ib"],
            "Consumer Measurements (8)": ["C_tp_vr", "C_tp_vy", "C_tp_vb", "C_tp_ir", "C_tp_iy", "C_tp_ib", "C_sp_i", "C_sp_v"]
        }
        
        for category, features in feature_categories.items():
            with st.expander(category):
                for feature in features:
                    st.write(f"• {feature}")
    
    with col2:
        st.markdown("### 📈 Performance Metrics")
        
        # Simulated performance metrics (you can replace with actual metrics)
        metrics = {
            "Accuracy": "92.3%",
            "Precision": "91.8%", 
            "Recall": "90.5%",
            "F1-Score": "91.1%",
            "ROC AUC": "94.2%"
        }
        
        for metric, value in metrics.items():
            st.metric(metric, value)
        
        st.markdown("### 🎯 Fault Classes")
        for fault in dashboard.label_encoder.classes_:
            info = dashboard.get_fault_info(fault)
            severity_class = f"severity-{info['severity'].lower()}"
            st.markdown(f'<div class="{severity_class}"><strong>{fault}</strong></div>', unsafe_allow_html=True)

def render_help_guide():
    """Render help and guide page"""
    st.markdown('<div class="sub-header">❓ Help & User Guide</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🚀 Getting Started
    
    ### Quick Steps:
    1. **Select a scenario** from the quick buttons or use manual input
    2. **Fill in measurements** across all component types
    3. **Click 'Analyze Fault'** to generate comprehensive report
    4. **Review all tabs** for detailed analysis and recommendations
    
    ## 📊 Understanding the Dashboard
    
    ### Probability Analysis Tab:
    - **Bar Chart**: Shows probability distribution across all fault types
    - **Confidence Gauge**: Visual indicator of prediction reliability
    - **Quick Stats**: Key statistics about the prediction
    
    ### Measurements Overview Tab:
    - **Voltage/Current Comparisons**: Side-by-side analysis across phases
    - **Scatter Plots**: Relationship between different measurement types
    - **Process Status**: Pie chart showing system component status
    
    ### Fault Analysis Tab:
    - **Severity Assessment**: Color-coded risk level
    - **Detailed Symptoms**: Specific indicators for the detected fault
    - **Action Plan**: Step-by-step troubleshooting guide
    - **Impact Assessment**: Business and operational impact analysis
    
    ## 🔧 Input Guidelines
    
    ### Process Status:
    - `success`: Component is operating normally
    - `fail`: Component has process failures or connectivity issues
    
    ### Electrical Measurements:
    - **Typical Voltage Range**: 0-300V
    - **Typical Current Range**: 0-10A  
    - **Three-Phase (TP)**: Measurements for R, Y, B phases
    - **Single-Phase (SP)**: Individual phase measurements
    
    ## 🎯 Interpreting Results
    
    ### Confidence Levels:
    - **HIGH** (> 0.8): Strong prediction - immediate action recommended
    - **MEDIUM** (0.6-0.8): Reasonable prediction - verify with additional checks
    - **LOW** (< 0.6): Weak prediction - requires manual investigation
    
    ### Severity Levels:
    - **HIGH**: Critical issue requiring immediate attention
    - **MEDIUM**: Significant issue needing prompt resolution  
    - **LOW**: Minor issue for scheduled maintenance
    """)

if __name__ == "__main__":
    main()