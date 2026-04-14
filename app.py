import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import pandas_datareader as pdr
import plotly.graph_objects as go
import streamlit as st
import time

st.set_page_config(
    page_title="Factor Exposure Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Factor Exposure Analyzer")
st.markdown("Analyze your portfolio's exposure to Fama-French 5 risk factors")

st.subheader("⚙️ Portfolio Settings")

start_date = st.date_input(
    "Start Date",
    help="Select the start date for your analysis in this format: YYYY-MM-DD."
)

st.caption("📅 Ensure your selected stocks were all publicly listed from this date onwards.")

tickers_input = st.text_input(
    "Enter Tickers (comma separated)",
    placeholder="e.g. AAPL, MSFT, NVDA, JPM"
)

weights_input = st.text_input(
    "Enter Weights (must sum to 100)",
    placeholder="e.g. 40, 30, 20, 10  or  33.5, 33.5, 33",
    help="Enter one weight per ticker. Decimals are allowed. All weights must sum to 100."
)

run = st.button("Run Analysis", type="primary")
if run:
    from datetime import date
    
    today = date.today()
    months_of_data = (today.year - start_date.year) * 12 + (today.month - start_date.month)
    
    if start_date >= today:
        st.error("❌ Start date cannot be in the future. Please select an earlier date.")
        st.stop()
    
    elif months_of_data < 12:
        st.error(f"❌ Your selected start date only gives {months_of_data} months of data. Please select a start date at least 18 months ago for a reliable regression.")
        st.stop()
    
    elif months_of_data < 60:
        st.warning(f"⚠️ You have {months_of_data} months of data. 60+ months is recommended for stronger results, but we'll proceed.")

    # --- Data Collection ---
    progress = st.info("⏳ Fetching data... please wait.")
    time.sleep(1.8)

    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    weights = [float(w.strip())/100 for w in weights_input.split(",")]

    if len(tickers) != len(weights):
        st.error("❌ Number of tickers and weights must match. Please check your inputs.")
        st.stop()

    if round(sum(weights), 2) != 1.0:
        st.error(f"❌ Weights sum to {sum(weights)*100:.1f}%. They must sum to 100%. Please check your inputs.")
        st.stop()

    start_date_str = start_date.strftime("%Y-%m-%d")

    ff_factors = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3', start=start_date_str)[0]
    ff_factors.index = ff_factors.index.to_timestamp()
    ff_factors.index = ff_factors.index.to_period('M')
    ff_factors = ff_factors / 100

    prices = yf.download(tickers, start=start_date_str, auto_adjust=False)['Adj Close']
    monthly_prices = prices.resample('M').last()
    monthly_returns = monthly_prices.pct_change().dropna()

    portfolio_returns = monthly_returns.dot(weights)
    portfolio_returns.index = portfolio_returns.index.to_period('M')

    merged = pd.concat([portfolio_returns, ff_factors], axis=1).dropna()
    merged.columns = ['Portfolio', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    # --- Regression ---
    Y = merged['Portfolio'] - merged['RF']
    X = merged[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    X = sm.add_constant(X)
    if len(merged) < 24:
        st.error(f"❌ Not enough overlapping data after merging. Only {len(merged)} months found. Try an earlier start date or different tickers that have more historical data.")
        st.stop()
    model = sm.OLS(Y, X).fit()

    params = model.params
    pvalues = model.pvalues

    def significance(p):
        if p < 0.01:
            return "🟢 Highly Significant"
        elif p < 0.05:
            return "🟡 Significant"
        elif p < 0.10:
            return "🟠 Marginally Significant"
        else:
            return "🔴 Not Significant"

    monthly_alpha = params['const']
    annualized_alpha = (1 + monthly_alpha) ** 12 - 1

    # --- Factor Report ---
    progress.empty()
    st.success("✅ Analysis Complete!")
    st.subheader("📋 Factor Exposure Report")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("R-Squared", f"{model.rsquared:.4f}")
        st.metric("Adj. R-Squared", f"{model.rsquared_adj:.4f}")
        st.metric("Observations", f"{int(model.nobs)} months")

    with col2:
        st.metric("Monthly Alpha", f"{monthly_alpha*100:.2f}%")
        st.metric("Annualized Alpha", f"{annualized_alpha*100:.2f}%")
        st.metric("Alpha Significance", significance(pvalues['const']))

    st.divider()

    st.subheader("📊 Factor Loadings")
    factor_data = {
        "Factor": ["Market Beta", "Size (SMB)", "Value (HML)", "Profitability (RMW)", "Investment (CMA)"],
        "Coefficient": [f"{params[f]:.4f}" for f in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']],
        "P-Value": [f"{pvalues[f]:.3f}" for f in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']],
        "Significance": [significance(pvalues[f]) for f in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    }
    st.dataframe(pd.DataFrame(factor_data), use_container_width=True)
    st.divider()
    st.subheader("📈 Visualizations")

    # --- Chart 1 — Factor Loadings Bar Chart ---
    factors = ['Alpha', 'Market Beta', 'Size (SMB)', 'Value (HML)', 'Profitability (RMW)', 'Investment (CMA)']
    values = [params['const'], params['Mkt-RF'], params['SMB'], params['HML'], params['RMW'], params['CMA']]
    colors = ['green' if v > 0 else 'red' for v in values]

    fig1 = go.Figure(go.Bar(
        x=values,
        y=factors,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.4f}' for v in values],
        textposition='auto',
        textfont=dict(size=12)
    ))
    fig1.update_layout(
        title='Factor Loadings — Fama French 5 Factor Model',
        xaxis_title='Coefficient',
        template='plotly_dark',
        height=500,
        xaxis=dict(range=[min(values) * 1.6, max(values) * 1.6], automargin=True),
        margin=dict(l=160)
    )
    st.plotly_chart(fig1, use_container_width=True, config={
    'scrollZoom': False,
    'displayModeBar': False,
    'staticPlot': True
    })

    # --- Chart 2 — Actual vs Fitted ---
    fitted = model.fittedvalues
    actual = Y

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=actual.index.astype(str),
        y=actual.values,
        mode='lines',
        name='Actual Returns',
        line=dict(color='#00e5b4', width=2)
    ))
    fig2.add_trace(go.Scatter(
        x=fitted.index.astype(str),
        y=fitted.values,
        mode='lines',
        name='Fitted Returns',
        line=dict(color='#ff6b6b', width=2, dash='dash')
    ))
    fig2.update_layout(
        title='Actual vs Fitted Portfolio Returns',
        xaxis_title='Date',
        yaxis_title='Monthly Return',
        template='plotly_dark',
        height=450,
        hovermode='x unified'
    )
    st.plotly_chart(fig2, use_container_width=True, config={
    'scrollZoom': False,
    'displayModeBar': False,
    'staticPlot': True
    })

    # --- Chart 3 — Cumulative Returns ---
    cumulative_portfolio = (1 + merged['Portfolio']).cumprod() - 1
    cumulative_market = (1 + merged['Mkt-RF'] + merged['RF']).cumprod() - 1

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=merged.index.astype(str),
        y=cumulative_portfolio.values,
        mode='lines',
        name='Your Portfolio',
        line=dict(color='#00e5b4', width=2)
    ))
    fig3.add_trace(go.Scatter(
        x=merged.index.astype(str),
        y=cumulative_market.values,
        mode='lines',
        name='US Total Market (FF)',
        line=dict(color='#5b8cff', width=2, dash='dash'),
        st.caption("Market benchmark represents the total US stock market portfolio as defined by Kenneth French's data library.")
    ))
    fig3.update_layout(
        title='Cumulative Returns — Portfolio vs Market',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        yaxis_tickformat='.0%',
        template='plotly_dark',
        height=450,
        hovermode='x unified'
    )
    st.plotly_chart(fig3, use_container_width=True, config={
    'scrollZoom': False,
    'displayModeBar': False,
    'staticPlot': True
    })

    # --- Chart 4 — Rolling Beta ---
    rolling_beta = []
    rolling_dates = []

    for i in range(12, len(merged)):
        window = merged.iloc[i-12:i]
        Y_roll = window['Portfolio'] - window['RF']
        X_roll = window[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
        X_roll = sm.add_constant(X_roll)
        roll_model = sm.OLS(Y_roll, X_roll).fit()
        rolling_beta.append(roll_model.params['Mkt-RF'])
        rolling_dates.append(merged.index[i])

    rolling_dates_str = [str(d) for d in rolling_dates]

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=rolling_dates_str,
        y=rolling_beta,
        mode='lines',
        name='Rolling Beta',
        line=dict(color='#ffd166', width=2)
    ))
    fig4.add_hline(
        y=1.0,
        line_dash='dash',
        line_color='white',
        opacity=0.4,
        annotation_text='Beta = 1'
    )
    fig4.update_layout(
        title='Rolling 12-Month Market Beta',
        xaxis_title='Date',
        yaxis_title='Beta',
        template='plotly_dark',
        height=450,
        hovermode='x unified'
    )
    fig4.update_xaxes(
        tickmode='array',
        tickvals=[rolling_dates_str[i] for i in range(0, len(rolling_dates_str), 6)],
        ticktext=[rolling_dates_str[i] for i in range(0, len(rolling_dates_str), 6)]
    )
    st.plotly_chart(fig4, use_container_width=True, config={
    'scrollZoom': False,
    'displayModeBar': False,
    'staticPlot': True
    })

    st.divider()
    st.caption("Data sourced from Yahoo Finance and Kenneth French's Data Library at Dartmouth. For educational and research purposes only.")



