import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime

# Page config with custom theme
st.set_page_config(
    page_title="Processor Lifecycle Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .stMetric {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    h1 {
        background: linear-gradient(135deg, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: bold;
    }
    .uploadedFile {
        border: 2px dashed #3b82f6;
        border-radius: 10px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load models
kmeans = joblib.load("backend/models/kmeans_model.joblib")
scaler = joblib.load("backend/models/scaler.joblib")
label_map = joblib.load("backend/models/label_mapping.joblib")

st.title(" Processor Lifecycle & Health Dashboard")
st.markdown("---")

st.markdown("""
### 📄 Required CSV Format  
Your CSV must contain the following columns for accurate prediction:

| Column Name | Description |
|------------|-------------|
| **overclock_proxy** | 1 if GPU is overclocked, otherwise 0 |
| **usage_hours** | Total runtime hours |
| **avg_power_watts** | Average power usage in watts |
| **peak_power_watts** | Peak observed watt usage |
| **avg_sm_pct** | Average Streaming Multiprocessor utilization (%) |
| **avg_mem_pct** | Average memory utilization (%) |
| **thermal_score** | Numeric temperature health score |

---

#### ✅ Optional (Will be used if provided):
| Column Name | Usage |
|------------|-------|
| **fan_speed_rpm** | Used to refine cooling/thermal recommendations |
| **voltage_mv** | Helps assess electrical stress |
| **memory_temp** | Used for memory aging estimation |

> 📌 *If any required columns are missing, the dashboard will auto-fill them with safe defaults, but predictions may be less accurate.*
""")

st.markdown("---")

# --- Upload CSV ---
uploaded = st.file_uploader("📁 Upload CSV Data File", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    # Feature processing
    FEATURE_COLS = scaler.feature_names_in_
    input_df = df.copy()
    for col in FEATURE_COLS:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Predictions
    scaled = scaler.transform(input_df)
    df["cluster"] = kmeans.predict(scaled)
    df["health_class"] = df["cluster"].map(label_map)
    df["life_score"] = (100 - (df["thermal_score"] + df["avg_sm_pct"] / 2)).clip(0, 100)
    df["GPU_ID"] = df.index + 1

    # Health Summary Cards
    st.markdown("### 📊 Health Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Life Score", f"{df['life_score'].mean():.1f} / 100", delta=None)
    with col2:
        st.metric("Most Common Condition", df["health_class"].mode()[0])
    with col3:
        high_risk = df[df['life_score'] < 30].shape[0]
        st.metric("High-Risk Units", high_risk, delta=f"-{high_risk}" if high_risk > 0 else "0", delta_color="inverse")

    st.markdown("---")

    # Charts in columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### GPU Stress Profile")
        numeric_cols = ["avg_power_watts", "avg_sm_pct", "avg_mem_pct", "usage_hours"]
        radar_sample = df[numeric_cols].mean()
        radar = go.Figure(data=go.Scatterpolar(
            r=radar_sample.values, theta=numeric_cols, fill='toself',
            fillcolor='rgba(59, 130, 246, 0.3)',
            line=dict(color='rgb(59, 130, 246)', width=2)
        ))
        radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(radar_sample.values)])),
            showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(radar, use_container_width=True)

    with col2:
        st.markdown("### Remaining Life Gauge")
        avg_life = df["life_score"].mean()
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_life,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "rgb(16, 185, 129)" if avg_life >= 70 else "rgb(251, 191, 36)" if avg_life >= 40 else "rgb(239, 68, 68)"},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(239, 68, 68, 0.2)"},
                    {'range': [40, 70], 'color': "rgba(251, 191, 36, 0.2)"},
                    {'range': [70, 100], 'color': "rgba(16, 185, 129, 0.2)"}
                ]
            },
            number={'suffix': " / 100", 'font': {'size': 40, 'color': 'white'}}
        ))
        gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=300)
        st.plotly_chart(gauge, use_container_width=True)

    st.markdown("---")

    # Cluster and Pie
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Cluster Positioning")
        fig = px.scatter(df, x="avg_power_watts", y="thermal_score", color="health_class",
                         color_discrete_map={"Healthy":"rgb(16,185,129)","Moderate":"rgb(251,191,36)","Critical":"rgb(239,68,68)"},
                         hover_data=["GPU_ID","life_score"])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,41,59,0.5)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.markdown("###  Health Distribution")
        pie = px.pie(df, names="health_class", color="health_class",
                     color_discrete_map={"Healthy":"rgb(16,185,129)","Moderate":"rgb(251,191,36)","Critical":"rgb(239,68,68)"})
        pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(pie, use_container_width=True)

    st.markdown("---")

    # Maintenance & Recycling
    st.markdown("###  Maintenance & Recycling Recommendations")
    def get_recommendations(row):
        recs = []
        if row["thermal_score"] > 18: recs.append("Clean fans, reapply thermal paste")
        if row["avg_power_watts"] > 180: recs.append("Lower power limit or increase cooling")
        if row["avg_sm_pct"] > 70 and row["overclock_proxy"] == 1: recs.append("Reduce overclock or enforce power cap")
        return ", ".join(recs) if recs else "None"
    def recycling_category(row):
        if row["life_score"] > 70: return "✅ Can continue use"
        elif row["life_score"] > 40: return "⚠️ Consider partial recycling"
        else: return "🔴 Recycle / Retire"
    df["Maintenance"] = df.apply(get_recommendations, axis=1)
    df["Recycling"] = df.apply(recycling_category, axis=1)

    display_cols = ["GPU_ID","cluster","health_class","life_score","Maintenance","Recycling",
                    "usage_hours","avg_power_watts","peak_power_watts","avg_sm_pct","avg_mem_pct","thermal_score"]
    st.dataframe(df[display_cols].style.background_gradient(subset=['life_score'], cmap='RdYlGn'), use_container_width=True, height=400)

    st.markdown("---")
    st.markdown("### GPU Comparison Mode")

    # GPU selection inputs
    gpu_list = df["GPU_ID"].tolist()
    colA, colB = st.columns(2)
    with colA: gpu1 = st.selectbox("Select GPU 1", gpu_list, key="gpu1_select")
    with colB: gpu2 = st.selectbox("Select GPU 2", gpu_list, key="gpu2_select")

    if gpu1 != gpu2:
        gpu_a = df[df["GPU_ID"] == gpu1].iloc[0]
        gpu_b = df[df["GPU_ID"] == gpu2].iloc[0]

        # Side-by-side numeric comparison
        col1, col2 = st.columns(2)
        with col1: st.markdown(f"#### 🟦 GPU {gpu1} Stats"); st.write(gpu_a[["health_class","life_score","usage_hours","avg_power_watts","avg_sm_pct","avg_mem_pct","thermal_score"]])
        with col2: st.markdown(f"#### 🟩 GPU {gpu2} Stats"); st.write(gpu_b[["health_class","life_score","usage_hours","avg_power_watts","avg_sm_pct","avg_mem_pct","thermal_score"]])

        # Performance vs Efficiency
        gpu_a_eff = gpu_a["avg_sm_pct"] / gpu_a["avg_power_watts"]
        gpu_b_eff = gpu_b["avg_sm_pct"] / gpu_b["avg_power_watts"]
        eff_df = pd.DataFrame({"GPU":[f"GPU {gpu1}",f"GPU {gpu2}"], "Efficiency Score":[gpu_a_eff,gpu_b_eff]})
        eff_chart = px.bar(eff_df, x="GPU", y="Efficiency Score", title=" Performance vs Power Efficiency", color="GPU", color_discrete_sequence=["rgb(59,130,246)","rgb(16,185,129)"])
        eff_chart.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,41,59,0.4)', font=dict(color='white'))
        st.plotly_chart(eff_chart, use_container_width=True)

        # Workload Recommendation
        def workload_recommendation(row):
            if row["avg_sm_pct"] > row["avg_mem_pct"]: return "Compute-Intensive Workloads (AI/ML, HPC)"
            elif row["avg_mem_pct"] > row["avg_sm_pct"]: return "Memory-Intensive Workloads (Rendering, Video Processing)"
            else: return "Balanced Workloads"
        rec1 = workload_recommendation(gpu_a)
        rec2 = workload_recommendation(gpu_b)
        st.markdown("### Recommended Optimal Usage Profiles")
        st.info(f"**GPU {gpu1} →** {rec1}\n\n**GPU {gpu2} →** {rec2}")

        # --- PDF Export ---
        st.markdown("### 📝 Export Comparison Report")
        def export_comparison_pdf(gpu1, gpu2, df):
            gpu_a = df[df["GPU_ID"] == gpu1].iloc[0]
            gpu_b = df[df["GPU_ID"] == gpu2].iloc[0]
            pdf_path = f"gpu_comparison_report_{gpu1}_vs_{gpu2}.pdf"
            c = canvas.Canvas(pdf_path, pagesize=letter)
            width, height = letter
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 50, f"GPU Comparison Report: GPU {gpu1} vs GPU {gpu2}")
            c.setFont("Helvetica", 12)
            c.drawString(50, height - 80, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            y_position = height - 120
            for col, label in [("health_class","Health"),("life_score","Life Score"),("usage_hours","Usage Hours"),
                               ("avg_power_watts","Avg Power (W)"),("peak_power_watts","Peak Power (W)"),
                               ("avg_sm_pct","Avg Compute (%)"),("avg_mem_pct","Avg Memory (%)"),
                               ("thermal_score","Thermal Score")]:
                c.drawString(50, y_position, f"{label} GPU {gpu1}: {gpu_a[col]}")
                c.drawString(300, y_position, f"{label} GPU {gpu2}: {gpu_b[col]}")
                y_position -= 20
            # Include Recommendations & Efficiency
            gpu_a_eff = gpu_a["avg_sm_pct"] / gpu_a["avg_power_watts"]
            gpu_b_eff = gpu_b["avg_sm_pct"] / gpu_b["avg_power_watts"]
            rec_a = workload_recommendation(gpu_a)
            rec_b = workload_recommendation(gpu_b)
            c.drawString(50, y_position-10, f"Efficiency Score GPU {gpu1}: {gpu_a_eff:.2f}")
            c.drawString(300, y_position-10, f"Efficiency Score GPU {gpu2}: {gpu_b_eff:.2f}")
            y_position -= 30
            c.drawString(50, y_position, f"Recommendation GPU {gpu1}: {rec_a}")
            c.drawString(300, y_position, f"Recommendation GPU {gpu2}: {rec_b}")
            c.save()
            return pdf_path

        if st.button("Generate PDF Report"):
            pdf_path = export_comparison_pdf(gpu1, gpu2, df)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(label="📥 Download PDF", data=pdf_bytes, file_name=f"GPU_Comparison_{gpu1}_vs_{gpu2}.pdf", mime="application/pdf")
            st.success("✅ PDF ready for download!")

else:
    st.info("👆 Please upload a CSV file to begin analysis")
