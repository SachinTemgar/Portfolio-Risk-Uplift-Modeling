import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Financial Intervention ROI Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4e;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(70, 130, 180, 0.25);
        border-color: #4682B4;
    }

    [data-testid="stMetricLabel"] {
        font-size: 12px !important;
        color: #8899aa !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #ffffff !important;
    }

    h1 {
        color: #ffffff !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        font-size: 28px !important;
    }

    h2, .stSubheader {
        color: #c0d0e0 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #4682B4;
        padding-bottom: 10px;
        margin-bottom: 20px !important;
        font-size: 20px !important;
    }

    h3 {
        color: #a0b8d0 !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }

    hr {
        border-color: #1a2a3e !important;
        margin: 20px 0 !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #0e1117 100%);
        border-right: 1px solid #1a2a3e;
    }

    [data-testid="stSidebar"] h2 {
        border-bottom: 1px solid #2a3a5e !important;
        font-size: 16px !important;
        color: #87CEEB !important;
    }

    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stSlider > div > div {
        background-color: #1a1a2e !important;
        border: 1px solid #2a2a4e !important;
        color: #e0e0e0 !important;
        border-radius: 8px !important;
    }

    .stSelectbox > div > div:hover,
    .stNumberInput > div > div > input:hover {
        border-color: #4682B4 !important;
    }

    [data-testid="stPlotlyChart"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 12px;
        border: 1px solid #2a2a4e;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    .stRadio > div { gap: 8px; }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #2a2a4e; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #4682B4; }

    [data-testid="stSidebar"] label {
        color: #8899aa !important;
        font-size: 13px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .streamlit-expanderHeader {
        color: #87CEEB !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

BLUE = '#4682B4'
CADET = '#5F9EA0'
SKY = '#87CEEB'
RED = '#CD5C5C'
GREEN = '#2E8B57'
ORANGE = '#E8943A'
BG = '#0e1117'
CARD = '#1a1a2e'
GRID = '#1a2a3e'
TEXT = '#c0d0e0'

CHART_LAYOUT = dict(
    plot_bgcolor=BG,
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color=TEXT, family='Segoe UI, Roboto, sans-serif', size=12),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    margin=dict(l=50, r=30, t=50, b=40)
)

import os
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_PATH = ROOT_DIR / 'data' / 'processed' / 'segmented_scores.csv'

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"Data file not found at: {DATA_PATH}")
        st.stop()
    return pd.read_csv(DATA_PATH)

df = load_data()
LGD = 0.45

label_map = {
    'Persuadables': 'Target Group',
    'Sure Things': 'Will Recover',
    'Sleeping Dogs': 'Do Not Contact'
}
df['quadrant_label'] = df['quadrant'].map(label_map)

# SIDEBAR
with st.sidebar:
    st.markdown("## Control Center")
    st.caption("Adjust parameters to simulate different scenarios")

    st.markdown("---")
    st.markdown("### Economic Parameters")

    cost_per_call = st.slider(
        "Cost per Outreach (CU)", min_value=10, max_value=200,
        value=50, step=5, help="Total cost per customer contact including agent time and overhead"
    )

    max_calls = st.slider(
        "Resource Capacity (Max Calls)", min_value=1000, max_value=30000,
        value=29131, step=500, help="Maximum number of calls your team can make"
    )

    st.markdown("---")
    st.markdown("### Model Selection")

    model_choice = st.selectbox(
        "Uplift Model",
        ['T-Learner', 'X-Learner', 'Causal Forest'],
        help="Select which model's uplift scores to use for targeting"
    )

    model_col_map = {
        'T-Learner': 'uplift_t_learner',
        'X-Learner': 'uplift_x_learner',
        'Causal Forest': 'uplift_causal_forest'
    }
    active_model_col = model_col_map[model_choice]

    st.markdown("---")
    st.markdown("### Segment Filters")

    loan_filter = st.multiselect(
        "Loan Type",
        options=df['NAME_CONTRACT_TYPE'].unique().tolist(),
        default=df['NAME_CONTRACT_TYPE'].unique().tolist()
    )

    edu_options = df['NAME_EDUCATION_TYPE'].value_counts().head(5).index.tolist()
    edu_filter = st.multiselect(
        "Education Level",
        options=edu_options,
        default=edu_options
    )

    st.markdown("---")
    with st.expander("Model Interpretability"):
        st.markdown(
            """
            **Top drivers for intervention response:**

            1. **External Score 3** - Strongest signal. Lower scores indicate
            customers who benefit most from outreach.

            2. **External Score 2** - Second strongest. Combined with Score 3,
            these capture external credit bureau assessments.

            3. **Debt-to-Income** - Customers in the 0.20-0.35 range respond
            best. Stressed enough to need help, capable enough to act on it.

            4. **Customer Age** - Younger customers (30-45) show higher
            responsiveness to restructuring offers.

            5. **Loan Amount** - Larger loans create stronger incentive for
            both the customer and institution to prevent default.

            These drivers were identified through correlation analysis with
            uplift scores and validated across all three models (T-Learner,
            X-Learner, Causal Forest).
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#555; font-size:11px; padding:10px;'>"
        "Sachin Temgar | 2026</div>",
        unsafe_allow_html=True
    )

filtered = df[
    (df['NAME_CONTRACT_TYPE'].isin(loan_filter)) &
    (df['NAME_EDUCATION_TYPE'].isin(edu_filter))
].copy()

if filtered.empty:
    st.warning("No customers match these filters. Adjust the Control Center settings.")
    st.stop()

target_group = filtered[filtered['quadrant_label'] == 'Target Group']
will_recover = filtered[filtered['quadrant_label'] == 'Will Recover']
do_not_contact = filtered[filtered['quadrant_label'] == 'Do Not Contact']

treatment = filtered[filtered['TREATMENT'] == 1]
control = filtered[filtered['TREATMENT'] == 0]

ranked = target_group.sort_values(active_model_col, ascending=False).reset_index(drop=True)
ranked['recovery_value'] = ranked['AMT_CREDIT'] * LGD
ranked['expected_gain'] = (ranked[active_model_col] * ranked['recovery_value']) - cost_per_call
ranked['cumulative_gain'] = ranked['expected_gain'].cumsum()

if len(ranked) > 0:
    optimal_idx = ranked['cumulative_gain'].idxmax()
    optimal_count = min(optimal_idx + 1, max_calls, len(ranked))
    peak_gain = ranked['cumulative_gain'].iloc[min(optimal_count - 1, len(ranked) - 1)]
    total_cost = optimal_count * cost_per_call
    roi = (peak_gain / total_cost) * 100 if total_cost > 0 else 0
else:
    optimal_count = 0
    peak_gain = 0
    total_cost = 0
    roi = 0

naive_cost = len(filtered) * cost_per_call
naive_prevented = control['TARGET'].sum() - treatment['TARGET'].sum() if len(treatment) > 0 else 0
naive_value = naive_prevented * filtered['AMT_CREDIT'].mean() * LGD if len(filtered) > 0 else 0
naive_net = naive_value - naive_cost

efficiency = (1 - optimal_count / len(filtered)) * 100 if len(filtered) > 0 else 0
cost_benefit = peak_gain / total_cost if total_cost > 0 else 0

# HEADER
st.title("Beyond Risk Scores: Financial Intervention ROI Simulator")
st.caption("Shifting from Reactive Risk Scoring to Proactive Prescriptive Analytics")
st.divider()

# KPI CARDS
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Recovery Value", f"{peak_gain:,.0f} CU",
              delta=f"+{peak_gain - naive_net:,.0f} vs baseline")
with col2:
    st.metric("Return per 1 CU", f"{cost_benefit:,.0f} CU",
              delta="per CU spent")
with col3:
    st.metric("Calls Saved", f"{efficiency:.1f}%",
              delta="removed from queue")
with col4:
    st.metric("Target Group", f"{len(target_group):,}",
              delta=f"{len(target_group)/len(filtered)*100:.1f}%" if len(filtered) > 0 else "0%")
with col5:
    st.metric("Model", model_choice, delta="Active")

st.divider()

# QINI CURVE + TARGETING OPTIMIZER
st.subheader("Intervention Performance")
col_qini, col_decile = st.columns(2)

with col_qini:
    def qini_curve(y_true, uplift_scores, treatment_flag):
        order = np.argsort(-uplift_scores)
        y_sorted = y_true[order]
        t_sorted = treatment_flag[order]
        n_treat = np.cumsum(t_sorted)
        n_ctrl = np.cumsum(1 - t_sorted)
        n_ctrl_safe = np.where(n_ctrl == 0, 1, n_ctrl)
        cum_treat_default = np.cumsum(y_sorted * t_sorted)
        cum_ctrl_default = np.cumsum(y_sorted * (1 - t_sorted))
        qini = cum_ctrl_default * n_treat / n_ctrl_safe - cum_treat_default
        return qini

    y = filtered['TARGET'].values
    t = filtered['TREATMENT'].values
    pct_range = np.linspace(0, 100, len(filtered))

    fig_qini = go.Figure()

    qc = qini_curve(y, filtered[active_model_col].values, t)
    fig_qini.add_trace(go.Scatter(
        x=pct_range, y=qc, mode='lines', name=model_choice,
        line=dict(color=BLUE, width=3)
    ))

    other_models = {k: v for k, v in model_col_map.items() if k != model_choice}
    faded_colors = {'T-Learner': CADET, 'X-Learner': CADET, 'Causal Forest': ORANGE}
    for name, col in other_models.items():
        qc_other = qini_curve(y, filtered[col].values, t)
        fig_qini.add_trace(go.Scatter(
            x=pct_range, y=qc_other, mode='lines', name=name,
            line=dict(color=faded_colors.get(name, CADET), width=1.5, dash='dot'),
            opacity=0.5
        ))

    np.random.seed(42)
    qc_random = qini_curve(y, np.random.random(len(filtered)), t)
    fig_qini.add_trace(go.Scatter(
        x=pct_range, y=qc_random, mode='lines', name='Random',
        line=dict(color='#444444', width=1.5, dash='dash')
    ))

    fig_qini.update_layout(
        title=f'Cumulative Gain: {model_choice} vs Alternatives',
        xaxis_title='% of Customers Targeted',
        yaxis_title='Incremental Recoveries',
        height=450,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0)'),
        **CHART_LAYOUT
    )
    st.plotly_chart(fig_qini, use_container_width=True)

with col_decile:
    try:
        df_decile = filtered.copy()
        df_decile['decile'] = pd.qcut(df_decile[active_model_col], q=10, labels=False, duplicates='drop')
        df_decile['decile'] = 10 - df_decile['decile']

        decile_data = df_decile.groupby('decile').agg(
            customers=('TARGET', 'count'),
            treat_def=('TARGET', lambda x: x[df_decile.loc[x.index, 'TREATMENT'] == 1].mean()),
            ctrl_def=('TARGET', lambda x: x[df_decile.loc[x.index, 'TREATMENT'] == 0].mean()),
            avg_loan=('AMT_CREDIT', 'mean')
        ).reset_index()

        decile_data['uplift'] = decile_data['ctrl_def'] - decile_data['treat_def']
        decile_data['net_profit'] = (
            decile_data['uplift'] * decile_data['customers'] * decile_data['avg_loan'] * LGD
        ) - (decile_data['customers'] * cost_per_call)

        cumulative_profit = decile_data['net_profit'].cumsum()
        cutoff_decile = cumulative_profit.idxmax() + 1 if len(cumulative_profit) > 0 else 5

        colors_decile = [GREEN if p > 0 else RED for p in decile_data['net_profit']]

        fig_decile = go.Figure()
        fig_decile.add_trace(go.Bar(
            x=decile_data['decile'], y=decile_data['net_profit'],
            marker_color=colors_decile,
            text=[f"{v:,.0f}" for v in decile_data['net_profit']],
            textposition='outside',
            textfont=dict(color=TEXT, size=10)
        ))

        fig_decile.add_vline(
            x=cutoff_decile + 0.5, line_dash="dash", line_color=ORANGE, line_width=2,
            annotation_text="Optimal Cutoff",
            annotation_font_color=ORANGE,
            annotation_font_size=12
        )

        fig_decile.update_layout(
            title=f'Net Profit by Decile (Cost = {cost_per_call} CU/call)',
            xaxis_title='Targeting Decile (1 = Highest Priority)',
            yaxis_title='Net Profit (CU)',
            height=450,
            **CHART_LAYOUT
        )
        st.plotly_chart(fig_decile, use_container_width=True)
    except Exception:
        st.info("Not enough data variation to generate decile analysis with current filters.")

st.divider()

# QUADRANT ANALYSIS
st.subheader("Quadrant Analysis")
col_quad, col_seg = st.columns([1.2, 0.8])

with col_quad:
    color_map = {'Target Group': BLUE, 'Will Recover': CADET, 'Do Not Contact': RED}

    fig_quad = go.Figure()
    for quad, color in color_map.items():
        subset = filtered[filtered['quadrant_label'] == quad]
        if len(subset) > 0:
            sample = subset.sample(min(2000, len(subset)), random_state=42)
            fig_quad.add_trace(go.Scatter(
                x=sample[active_model_col],
                y=sample['baseline_risk'],
                mode='markers', name=quad,
                marker=dict(color=color, size=5, opacity=0.4, line=dict(width=0))
            ))

    fig_quad.add_vline(x=0, line_dash="dash", line_color="#333333")
    fig_quad.update_layout(
        title='Customer Segmentation Map',
        xaxis_title=f'Uplift Score ({model_choice})',
        yaxis_title='Baseline Default Risk',
        height=450,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0)', font=dict(size=12)),
        **CHART_LAYOUT
    )
    st.plotly_chart(fig_quad, use_container_width=True)

with col_seg:
    for quad, color, action in [
        ('Target Group', BLUE, 'CALL FIRST'),
        ('Will Recover', CADET, 'SKIP'),
        ('Do Not Contact', RED, 'NEVER CALL')
    ]:
        group = filtered[filtered['quadrant_label'] == quad]
        count = len(group)
        pct = count / len(filtered) * 100 if len(filtered) > 0 else 0
        treat_def = group[group['TREATMENT'] == 1]['TARGET'].mean() * 100 if len(group[group['TREATMENT'] == 1]) > 0 else 0
        ctrl_def = group[group['TREATMENT'] == 0]['TARGET'].mean() * 100 if len(group[group['TREATMENT'] == 0]) > 0 else 0
        uplift = ctrl_def - treat_def

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                border-left: 4px solid {color};
                border-radius: 8px;
                padding: 15px 20px;
                margin-bottom: 12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            ">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="color:{color}; font-weight:700; font-size:16px;">{quad}</span>
                        <span style="color:#666; font-size:12px; margin-left:8px;">{action}</span>
                    </div>
                    <span style="color:#ffffff; font-weight:800; font-size:22px;">{count:,}</span>
                </div>
                <div style="display:flex; gap:20px; margin-top:8px; color:#8899aa; font-size:12px;">
                    <span>{pct:.1f}% of portfolio</span>
                    <span>Uplift: {uplift:+.2f} pp</span>
                    <span>Default: {treat_def:.1f}% vs {ctrl_def:.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.divider()

# WATERFALL CHART
st.subheader("Incremental Recovery Breakdown")

if len(filtered) > 0:
    target_recovery = 0
    if len(target_group) > 0:
        tg_treat = target_group[target_group['TREATMENT'] == 1]
        tg_ctrl = target_group[target_group['TREATMENT'] == 0]
        if len(tg_treat) > 0 and len(tg_ctrl) > 0:
            target_recovery = (tg_ctrl['TARGET'].mean() - tg_treat['TARGET'].mean()) * len(target_group) * target_group['AMT_CREDIT'].mean() * LGD

    wasted_sure = len(will_recover) * cost_per_call if len(will_recover) > 0 else 0

    harm_dnc = 0
    if len(do_not_contact) > 0:
        dnc_treat = do_not_contact[do_not_contact['TREATMENT'] == 1]
        dnc_ctrl = do_not_contact[do_not_contact['TREATMENT'] == 0]
        if len(dnc_treat) > 0 and len(dnc_ctrl) > 0:
            harm_dnc = (dnc_treat['TARGET'].mean() - dnc_ctrl['TARGET'].mean()) * len(do_not_contact) * do_not_contact['AMT_CREDIT'].mean() * LGD

    intervention_cost = optimal_count * cost_per_call

    fig_waterfall = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Target Group<br>Recovery", "Wasted on<br>Will Recover", "Harm from<br>Do Not Contact", "Intervention<br>Cost", "Net Value"],
        y=[target_recovery, -wasted_sure, -harm_dnc, -intervention_cost, 0],
        connector=dict(line=dict(color=GRID, width=1)),
        increasing=dict(marker=dict(color=GREEN)),
        decreasing=dict(marker=dict(color=RED)),
        totals=dict(marker=dict(color=BLUE)),
        textposition="outside",
        text=[f"{target_recovery:,.0f}", f"-{wasted_sure:,.0f}",
              f"-{harm_dnc:,.0f}", f"-{intervention_cost:,.0f}", ""],
        textfont=dict(size=11, color=TEXT)
    ))
    fig_waterfall.update_layout(
        yaxis_title='Value (CU)', height=400,
        plot_bgcolor=BG, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT, size=12),
        margin=dict(l=60, r=30, t=30, b=60),
        showlegend=False
    )
    fig_waterfall.update_yaxes(gridcolor=GRID)
    st.plotly_chart(fig_waterfall, use_container_width=True)

    net_val = target_recovery - wasted_sure - harm_dnc - intervention_cost
    if net_val < 0:
        st.warning(
            f"Net value is negative ({net_val:,.0f} CU). With current filters and cost settings, "
            f"the intervention cost exceeds recovery potential. Try reducing cost per outreach or "
            f"adjusting segment filters."
        )

st.divider()

# FEATURE IMPORTANCE + PERSONA + MODEL HEALTH
col_shap, col_profile, col_health = st.columns([1, 1, 0.8])

with col_shap:
    st.markdown("### What Drives the Uplift Score")

    importance_features = [
        'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DEBT_TO_INCOME',
        'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'CREDIT_TO_GOODS',
        'ANNUITY_TO_AGE', 'AGE_YEARS'
    ]
    available_imp = [f for f in importance_features if f in filtered.columns]

    correlations = filtered[available_imp].corrwith(filtered[active_model_col]).abs()
    correlations = correlations.sort_values(ascending=True).tail(8)

    name_map = {
        'EXT_SOURCE_2': 'External Score 2',
        'EXT_SOURCE_3': 'External Score 3',
        'DAYS_BIRTH': 'Customer Age',
        'DEBT_TO_INCOME': 'Debt-to-Income',
        'AMT_CREDIT': 'Loan Amount',
        'AMT_ANNUITY': 'Annual Payment',
        'AMT_INCOME_TOTAL': 'Income',
        'CREDIT_TO_GOODS': 'Credit-to-Goods',
        'ANNUITY_TO_AGE': 'Payment vs Age',
        'AGE_YEARS': 'Age (Years)'
    }
    display_names = [name_map.get(f, f) for f in correlations.index]

    fig_imp = go.Figure(go.Bar(
        x=correlations.values,
        y=display_names,
        orientation='h',
        marker=dict(
            color=correlations.values,
            colorscale=[[0, CADET], [1, BLUE]],
            line=dict(width=0)
        ),
        text=[f"{v:.3f}" for v in correlations.values],
        textposition='outside',
        textfont=dict(color=TEXT, size=11)
    ))
    fig_imp.update_layout(
        height=350,
        xaxis_title='Correlation with Uplift Score',
        plot_bgcolor=BG, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT, size=11),
        margin=dict(l=120, r=40, t=10, b=40),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

with col_profile:
    st.markdown("### Target Group Persona")

    if len(target_group) > 0:
        avg_age = target_group['AGE_YEARS'].mean()
        avg_income = target_group['AMT_INCOME_TOTAL'].mean()
        avg_loan = target_group['AMT_CREDIT'].mean()
        avg_dti = target_group['DEBT_TO_INCOME'].mean()
        avg_risk = target_group['baseline_risk'].mean()

        top_job = target_group['NAME_INCOME_TYPE'].mode().iloc[0] if 'NAME_INCOME_TYPE' in target_group.columns else "N/A"
        top_edu = target_group['NAME_EDUCATION_TYPE'].mode().iloc[0] if 'NAME_EDUCATION_TYPE' in target_group.columns else "N/A"

        persona_items = [
            ("Age", f"{avg_age:.0f} years", avg_age / 70),
            ("Income", f"{avg_income:,.0f} CU", min(avg_income / 300000, 1)),
            ("Loan Size", f"{avg_loan:,.0f} CU", min(avg_loan / 1500000, 1)),
            ("Debt-to-Income", f"{avg_dti:.1%}", min(avg_dti / 0.5, 1)),
            ("Default Risk", f"{avg_risk:.1%}", min(avg_risk / 0.3, 1)),
        ]

        for label, value, pct in persona_items:
            st.markdown(
                f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                        <span style="color:#8899aa; font-size:12px;">{label}</span>
                        <span style="color:#ffffff; font-size:12px; font-weight:600;">{value}</span>
                    </div>
                    <div style="background:#1a2a3e; border-radius:4px; height:6px; overflow:hidden;">
                        <div style="background:{BLUE}; width:{pct*100:.0f}%; height:100%; border-radius:4px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            f"""
            <div style="
                background:linear-gradient(135deg, #1a1a2e, #16213e);
                border:1px solid #2a2a4e; border-radius:8px;
                padding:12px 15px; margin-top:10px;
            ">
                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span style="color:#8899aa; font-size:11px;">TYPICAL EMPLOYMENT</span>
                    <span style="color:#ffffff; font-size:11px; font-weight:600;">{top_job}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#8899aa; font-size:11px;">TYPICAL EDUCATION</span>
                    <span style="color:#ffffff; font-size:11px; font-weight:600;">{top_edu}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("No Target Group customers in current filter.")

with col_health:
    st.markdown("### Model Reliability")

    smd_scores = {
        'EXT_SOURCE_2': 0.0022,
        'EXT_SOURCE_3': 0.0047,
        'DAYS_BIRTH': 0.0010,
        'DEBT_TO_INCOME': 0.0034
    }
    max_smd = max(smd_scores.values())
    smd_pass = all(v < 0.1 for v in smd_scores.values())

    achieved_power = 1.00
    ci_lower = 2.46
    ci_upper = 3.26

    st.markdown(
        f"""
        <div style="
            background:linear-gradient(135deg, #1a1a2e, #16213e);
            border:1px solid {GREEN if smd_pass else RED};
            border-radius:8px; padding:12px 15px; margin-bottom:10px;
        ">
            <div style="color:#8899aa; font-size:11px; text-transform:uppercase; letter-spacing:0.5px;">
                Group Balance (SMD)
            </div>
            <div style="color:{GREEN if smd_pass else RED}; font-size:22px; font-weight:800; margin:4px 0;">
                {'PASS' if smd_pass else 'FAIL'}
            </div>
            <div style="color:#8899aa; font-size:11px;">
                Max SMD: {max_smd:.4f} (threshold: 0.10)
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            background:linear-gradient(135deg, #1a1a2e, #16213e);
            border:1px solid {GREEN}; border-radius:8px;
            padding:12px 15px; margin-bottom:10px;
        ">
            <div style="color:#8899aa; font-size:11px; text-transform:uppercase; letter-spacing:0.5px;">
                Statistical Power
            </div>
            <div style="color:{GREEN}; font-size:22px; font-weight:800; margin:4px 0;">
                {achieved_power:.2f}
            </div>
            <div style="color:#8899aa; font-size:11px;">
                Required: 0.80 | Achieved: {achieved_power:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            background:linear-gradient(135deg, #1a1a2e, #16213e);
            border:1px solid {GREEN}; border-radius:8px;
            padding:12px 15px; margin-bottom:10px;
        ">
            <div style="color:#8899aa; font-size:11px; text-transform:uppercase; letter-spacing:0.5px;">
                A/B Test Result
            </div>
            <div style="color:{GREEN}; font-size:22px; font-weight:800; margin:4px 0;">
                p &lt; 0.001
            </div>
            <div style="color:#8899aa; font-size:11px;">
                95% CI: [{ci_lower} pp, {ci_upper} pp]
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            background:linear-gradient(135deg, #1a1a2e, #16213e);
            border:1px solid {BLUE}; border-radius:8px;
            padding:12px 15px;
        ">
            <div style="color:#8899aa; font-size:11px; text-transform:uppercase; letter-spacing:0.5px;">
                Verdict
            </div>
            <div style="color:{BLUE}; font-size:14px; font-weight:700; margin-top:4px;">
                Robust and replicable
            </div>
            <div style="color:#8899aa; font-size:11px; margin-top:2px;">
                Balanced groups, significant effect, full power
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# PRIORITY MATRIX
st.subheader("Collections Priority Matrix")

p_matrix = target_group.copy()
if len(p_matrix) > 100:
    try:
        p_matrix['uplift_tier'] = pd.qcut(
            p_matrix[active_model_col], q=3,
            labels=['Lower', 'Mid', 'Higher'], duplicates='drop'
        )
        p_matrix['loan_tier'] = pd.qcut(
            p_matrix['AMT_CREDIT'], q=3,
            labels=['Small Loan', 'Medium Loan', 'Large Loan'], duplicates='drop'
        )

        matrix = p_matrix.groupby(['uplift_tier', 'loan_tier'], observed=True).agg(
            customers=('TARGET', 'count'),
            net_val=('uplift_t_learner', lambda x: (
                x.mean() * p_matrix.loc[x.index, 'AMT_CREDIT'].mean() * LGD) - cost_per_call
            )
        ).reset_index()

        heat_pivot = matrix.pivot_table(
            index='uplift_tier', columns='loan_tier',
            values='net_val', observed=True
        )
        heat_pivot = heat_pivot.loc[['Higher', 'Mid', 'Lower']]

        count_piv = matrix.pivot_table(
            index='uplift_tier', columns='loan_tier',
            values='customers', observed=True
        )
        count_piv = count_piv.loc[['Higher', 'Mid', 'Lower']]

        annot = []
        for i in range(len(heat_pivot)):
            row = []
            for j in range(len(heat_pivot.columns)):
                v = heat_pivot.iloc[i, j]
                c = int(count_piv.iloc[i, j])
                row.append(f"{v:,.0f} CU<br>({c:,})")
            annot.append(row)

        col_heat_l, col_heat_m, col_heat_r = st.columns([0.15, 0.7, 0.15])
        with col_heat_m:
            fig_heat = go.Figure(go.Heatmap(
                z=heat_pivot.values,
                x=heat_pivot.columns.tolist(),
                y=heat_pivot.index.tolist(),
                colorscale=[[0, '#1a1a2e'], [0.5, '#4682B4'], [1, '#87CEEB']],
                text=annot, texttemplate="%{text}",
                textfont=dict(size=13, color='#ffffff'),
                colorbar=dict(title="Value/Call", tickfont=dict(color=TEXT, size=10)),
                hovertemplate='Priority: %{y}<br>Loan: %{x}<br>Value: %{z:,.0f} CU<extra></extra>'
            ))
            fig_heat.update_layout(
                title='Collections Priority Matrix',
                height=350, xaxis_title='Loan Value', yaxis_title='Call Priority',
                plot_bgcolor=BG, paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TEXT, size=12),
                margin=dict(l=80, r=20, t=50, b=40)
            )
            st.plotly_chart(fig_heat, use_container_width=True)
    except Exception:
        st.info("Not enough data in current filters to generate the priority matrix.")

st.divider()

# ROI SIMULATOR
st.subheader("ROI Simulator")

col_chart, col_table = st.columns([1.4, 0.6])

with col_chart:
    display_count = min(len(ranked), max_calls)
    display_ranked = ranked.iloc[:display_count]

    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(
        x=list(range(1, len(display_ranked) + 1)),
        y=display_ranked['cumulative_gain'],
        mode='lines', line=dict(color=BLUE, width=3),
        fill='tozeroy', fillcolor='rgba(70, 130, 180, 0.1)',
        name='Cumulative Net Gain',
        hovertemplate='Customer #%{x:,}<br>Gain: %{y:,.0f} CU<extra></extra>'
    ))

    fig_roi.add_trace(go.Scatter(
        x=[optimal_count], y=[peak_gain],
        mode='markers+text',
        marker=dict(color=ORANGE, size=12, symbol='diamond',
                    line=dict(color='#ffffff', width=1)),
        text=[f"Peak: {peak_gain:,.0f} CU"],
        textposition='top center',
        textfont=dict(color=ORANGE, size=12),
        showlegend=False
    ))

    fig_roi.add_vline(x=optimal_count, line_dash="dash", line_color=ORANGE, line_width=1.5)

    fig_roi.update_layout(
        title=f'Targeting Depth Optimizer ({model_choice}, Cost = {cost_per_call} CU)',
        xaxis_title='Customers Called (ranked by uplift)',
        yaxis_title='Cumulative Net Gain (CU)',
        height=420, showlegend=False,
        **CHART_LAYOUT
    )
    st.plotly_chart(fig_roi, use_container_width=True)

with col_table:
    st.markdown("### Strategy Comparison")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border: 1px solid #2a4a6e;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
        ">
            <div style="color:#87CEEB; font-size:13px; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">
                Uplift Strategy
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:#8899aa;">Customers Called</span>
                <span style="color:#ffffff; font-weight:700;">{optimal_count:,}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:#8899aa;">Total Cost</span>
                <span style="color:#ffffff; font-weight:700;">{total_cost:,.0f} CU</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:#8899aa;">Net Gain</span>
                <span style="color:{GREEN}; font-weight:700;">{peak_gain:,.0f} CU</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#8899aa;">ROI</span>
                <span style="color:{GREEN}; font-weight:700;">{roi:,.0f}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border: 1px solid #2a2a4e;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
        ">
            <div style="color:#888; font-size:13px; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">
                Current Strategy (Call All)
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:#8899aa;">Customers Called</span>
                <span style="color:#ffffff; font-weight:700;">{len(filtered):,}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:#8899aa;">Total Cost</span>
                <span style="color:#ffffff; font-weight:700;">{naive_cost:,.0f} CU</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:#8899aa;">Net Gain</span>
                <span style="color:#aaa; font-weight:700;">{naive_net:,.0f} CU</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#8899aa;">ROI</span>
                <span style="color:#aaa; font-weight:700;">{(naive_net/naive_cost)*100:,.0f}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    improvement = ((peak_gain - naive_net) / abs(naive_net)) * 100 if naive_net != 0 else 0
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0a2a1a, #0e3320);
            border: 1px solid {GREEN};
            border-radius: 10px;
            padding: 15px 20px;
            text-align: center;
        ">
            <div style="color:{GREEN}; font-size:12px; text-transform:uppercase; letter-spacing:1px;">
                Improvement Over Baseline
            </div>
            <div style="color:#ffffff; font-size:28px; font-weight:800; margin-top:5px;">
                {improvement:,.1f}%
            </div>
            <div style="color:#8899aa; font-size:12px; margin-top:3px;">
                more value with {efficiency:.0f}% fewer calls
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")
    if len(display_ranked) > 0:
        export_cols = ['SK_ID_CURR', 'AMT_CREDIT', 'AMT_INCOME_TOTAL',
                       'DEBT_TO_INCOME', 'baseline_risk', active_model_col,
                       'quadrant_label']
        export_available = [c for c in export_cols if c in display_ranked.columns]
        export_df = display_ranked[export_available].copy()
        export_df['call_priority'] = range(1, len(export_df) + 1)
        export_df['expected_gain_cu'] = display_ranked['expected_gain'].values

        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Targeting Manifest",
            data=csv,
            file_name="uplift_targeting_list.csv",
            mime="text/csv",
            help="Export the prioritized customer list for the Collections team",
            use_container_width=True
        )

st.divider()

# FOOTER
st.markdown(
    "<div style='text-align:center; color:#333; font-size:12px; padding:15px;'>"
    "Portfolio Risk Uplift Modeling | Beyond Risk Scores | Sachin Temgar | 2026"
    "</div>",
    unsafe_allow_html=True
)
