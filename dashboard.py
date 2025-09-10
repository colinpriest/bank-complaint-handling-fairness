#!/usr/bin/env python3
"""
Streamlit Dashboard for LLM Fairness Analysis Results
"""
import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px

def main():
    st.set_page_config(page_title="LLM Complaints Fairness", layout="wide")
    st.title("LLM Complaints Fairness — CFPB Narratives")
    
    # Data directory selector
    indir = st.sidebar.text_input("Results Directory", value="out")
    
    runs_path = os.path.join(indir, "runs.csv")
    paired_path = os.path.join(indir, "paired.csv")
    cost_path = os.path.join(indir, "cost_summary.json")
    
    if not os.path.exists(runs_path) or not os.path.exists(paired_path):
        st.error(f"Results not found in '{indir}' directory. Run the experiment first:")
        st.code("""
# Run the complete pipeline:
python complaints_llm_fairness_harness.py ingest --source socrata --total 100
python complaints_llm_fairness_harness.py prepare
python complaints_llm_fairness_harness.py run --models gpt-4o-mini
python complaints_llm_fairness_harness.py analyse
        """)
        return
    
    # Load data
    runs = pd.read_csv(runs_path)
    paired = pd.read_csv(paired_path)
    
    # Load cost data if available
    cost_data = None
    if os.path.exists(cost_path):
        with open(cost_path, "r", encoding="utf-8") as f:
            cost_data = json.load(f)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    models = sorted(runs["model"].dropna().unique().tolist())
    if models:
        selected_models = st.sidebar.multiselect("Models", models, default=models)
        runs = runs[runs["model"].isin(selected_models)]
        paired = paired[paired["model"].isin(selected_models)]
    
    # Main dashboard
    if runs.empty:
        st.warning("No data to display with current filters")
        return
    
    # Cost Analysis Section
    if cost_data:
        st.header("💰 Cost Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Cost", f"${cost_data['total_cost_usd']:.4f}", 
                     help="Total experiment cost in USD")
        with col2:
            st.metric("Total Requests", f"{cost_data['total_requests']:,}")
        with col3:
            if "models" in cost_data and cost_data["models"]:
                avg_cache_rate = sum(m.get("cache_hit_rate", 0) for m in cost_data["models"].values()) / len(cost_data["models"])
                st.metric("Avg Cache Hit Rate", f"{avg_cache_rate:.0%}")
        
        # Per-model costs
        if "models" in cost_data:
            st.subheader("Per-Model Breakdown")
            cost_df = pd.DataFrame([
                {
                    "Model": model,
                    "Cost ($)": stats["total_cost_usd"],
                    "API Calls": stats["api_calls"],
                    "Cache Hits": stats["cache_hits"],
                    "Cache Hit Rate": f"{stats['cache_hit_rate']:.0%}"
                }
                for model, stats in cost_data["models"].items()
            ])
            st.dataframe(cost_df, use_container_width=True)
    
    # Overview KPIs
    st.header("📊 Experiment Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(runs):,}")
    with col2:
        st.metric("Paired Comparisons", f"{len(paired):,}")
    with col3:
        baseline_monetary = runs[runs['variant']=='baseline']['monetary'].mean()
        st.metric("Monetary Relief (Baseline)", f"{baseline_monetary:.3f}", 
                 help="Generic customer profile with neutral demographics")
    with col4:
        persona_monetary = runs[runs['variant']=='persona']['monetary'].mean() if not runs[runs['variant']=='persona'].empty else 0
        fairness_monetary = runs[runs['variant']=='persona_fairness']['monetary'].mean() if not runs[runs['variant']=='persona_fairness'].empty else 0
        st.metric("Persona Effect", f"{persona_monetary:.3f}", 
                 delta=f"{persona_monetary - baseline_monetary:.3f}",
                 help="With demographic signals vs baseline")
    
    # Distribution of remedy tiers
    st.header("🎯 Remedy Tier Analysis")
    
    # Filter data for visualization
    tier_data = runs.dropna(subset=["remedy_tier"])
    if not tier_data.empty:
        fig = px.histogram(
            tier_data, 
            x="remedy_tier", 
            color="variant", 
            barmode="group", 
            facet_row="model",
            title="Distribution of Remedy Tiers (Baseline vs Persona vs Fairness)",
            labels={"remedy_tier": "Remedy Tier", "count": "Number of Cases"}
        )
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)
    
    # Bias Analysis
    st.header("⚖️ Bias Detection")
    
    if not paired.empty:
        # Calculate tier changes
        paired_with_delta = paired.copy()
        paired_with_delta["tier_delta"] = paired_with_delta["remedy_tier_G"] - paired_with_delta["remedy_tier_NC"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Three-Way Comparison")
            delta_fig = px.histogram(
                paired_with_delta, 
                x="tier_delta", 
                facet_row="model",
                title="Distribution of Tier Changes",
                labels={"tier_delta": "Tier Change (With Identity - No Identity)", "count": "Number of Cases"}
            )
            delta_fig.update_xaxes(dtick=1)
            st.plotly_chart(delta_fig, use_container_width=True)
        
        with col2:
            st.subheader("Monetary Relief Shifts")
            # McNemar analysis
            mcnemar_data = []
            for model in paired["model"].unique():
                model_data = paired[paired["model"] == model]
                nc_1_g_0 = ((model_data["monetary_NC"]==1) & (model_data["monetary_G"]==0)).sum()
                nc_0_g_1 = ((model_data["monetary_NC"]==0) & (model_data["monetary_G"]==1)).sum()
                mcnemar_data.append({
                    "Model": model,
                    "Lost Relief (Had → No)": nc_1_g_0,
                    "Gained Relief (No → Had)": nc_0_g_1,
                    "Net Identity Effect": nc_0_g_1 - nc_1_g_0
                })
            
            mcnemar_df = pd.DataFrame(mcnemar_data)
            st.dataframe(mcnemar_df, use_container_width=True)
    
    # Process Fairness
    st.header("🤝 Process Fairness")
    if not paired.empty:
        process_data = []
        for model in paired["model"].unique():
            model_data = paired[paired["model"] == model]
            process_data.extend([
                {"Model": model, "Variant": "No Identity", "Questions Asked": model_data["asked_question_NC"].sum()},
                {"Model": model, "Variant": "With Identity", "Questions Asked": model_data["asked_question_G"].sum()}
            ])
        
        process_df = pd.DataFrame(process_data)
        
        process_fig = px.bar(
            process_df,
            x="Variant", 
            y="Questions Asked", 
            color="Model",
            title="Clarifying Questions Asked by Variant",
            barmode="group"
        )
        st.plotly_chart(process_fig, use_container_width=True)
    
    # Raw Data Explorer
    with st.expander("🔍 Raw Data Explorer"):
        st.subheader("Individual Runs")
        st.dataframe(runs, use_container_width=True)
        
        if not paired.empty:
            st.subheader("Paired Comparisons")
            st.dataframe(paired, use_container_width=True)
    
    # Analysis Files
    analysis_path = os.path.join(indir, "analysis.json")
    if os.path.exists(analysis_path):
        with st.expander("📈 Statistical Analysis Results"):
            with open(analysis_path, "r", encoding="utf-8") as f:
                analysis_data = json.load(f)
            st.json(analysis_data)
    
    st.caption("Note: Narratives are de-identified and published under CFPB policy. Decisions are simulated for research only.")

if __name__ == "__main__":
    main()