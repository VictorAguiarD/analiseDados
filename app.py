import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta

# ==========================================
# 1. PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="Dashboard Executivo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium, clean corporate look
st.markdown("""
    <style>
        /* General layout overrides */
        .reportview-container {
            background: #f8f9fa;
        }
        .main {
            background-color: #f8f9fa;
            font-family: 'Outfit', 'Inter', sans-serif;
        }
        
        /* Metric Card styling */
        .metric-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border-left: 5px solid #1E3A8A; /* Dark Blue */
            margin-bottom: 20px;
        }
        .metric-title {
            font-size: 14px;
            color: #6B7280;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-value {
            font-size: 28px;
            color: #111827;
            font-weight: 700;
            margin-top: 5px;
        }
        .metric-delta {
            font-size: 12px;
            color: #10B981; /* Green */
            font-weight: 500;
            margin-top: 5px;
        }
        
        /* Insight Card styling */
        .insight-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
            border-top: 4px solid #10B981; /* Green accent */
            margin-bottom: 20px;
        }
        .insight-header {
            font-size: 18px;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 12px;
        }
        .insight-body {
            font-size: 14px;
            color: #4B5563;
            line-height: 1.6;
        }
        .recommendation-badge {
            background-color: #EEF2F6;
            border-left: 4px solid #F59E0B; /* Amber */
            padding: 10px 15px;
            border-radius: 4px;
            font-size: 13.5px;
            color: #374151;
            margin-top: 15px;
            font-style: italic;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING & CACHING
# ==========================================
@st.cache_data
def load_data():
    file_path = "vendas_supermercado.csv"
    df = pd.read_csv(file_path)
    # Select only the first 6 columns to drop trailing empty columns
    df = df.iloc[:, :6]
    # Drop rows that are completely empty or have missing critical data (like empty line 2002)
    df = df.dropna(subset=['Data', 'Quantidade', 'Total_Venda'])
    # Parse dates
    df['Data'] = pd.to_datetime(df['Data'], format='%m/%d/%Y')
    
    # Cast numbers
    df['Quantidade'] = df['Quantidade'].astype(int)
    df['Preco_Unitario'] = df['Preco_Unitario'].astype(float)
    df['Total_Venda'] = df['Total_Venda'].astype(float)
    
    # Day of week mapping
    dias_semana_map = {
        'Monday': 'Segunda-feira',
        'Tuesday': 'Terça-feira',
        'Wednesday': 'Quarta-feira',
        'Thursday': 'Quinta-feira',
        'Friday': 'Sexta-feira',
        'Saturday': 'Sábado',
        'Sunday': 'Domingo'
    }
    df['Dia_Semana'] = df['Data'].dt.day_name().map(dias_semana_map)
    df['Mes_Ano'] = df['Data'].dt.to_period('M')
    df['Nome_Mes'] = df['Data'].dt.strftime('%B')
    
    meses_pt = {
        'January': 'Janeiro', 'February': 'Fevereiro', 'March': 'Março', 'April': 'Abril',
        'May': 'Maio', 'June': 'Junho', 'July': 'Julho', 'August': 'Agosto',
        'September': 'Setembro', 'October': 'Outubro', 'November': 'Novembro', 'December': 'Dezembro'
    }
    df['Nome_Mes'] = df['Nome_Mes'].map(meses_pt)
    
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Erro ao carregar o arquivo de vendas: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR FILTERS
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081986.png", width=80)
st.sidebar.title("Filtros Executivos")
st.sidebar.markdown("Refine os dados para análise detalhada:")

# Date Range Filter
min_date = df_raw['Data'].min().to_pydatetime()
max_date = df_raw['Data'].max().to_pydatetime()
start_date, end_date = st.sidebar.date_input(
    "Período de Análise",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Convert st date input back to datetime
start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)

# Category Filter
all_categories = sorted(df_raw['Categoria'].unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Categorias de Produto",
    options=all_categories,
    default=all_categories,
    help="Selecione as categorias que deseja visualizar no dashboard."
)

# Product Search/Filter
filtered_products = df_raw[df_raw['Categoria'].isin(selected_categories)]['Produto'].unique().tolist()
selected_products = st.sidebar.multiselect(
    "Produtos Específicos (Opcional)",
    options=sorted(filtered_products),
    default=[],
    help="Deixe em branco para incluir todos os produtos das categorias selecionadas."
)

# Apply filters to dataframe
df_filtered = df_raw[
    (df_raw['Data'] >= start_dt) & 
    (df_raw['Data'] <= end_dt) &
    (df_raw['Categoria'].isin(selected_categories))
]

if selected_products:
    df_filtered = df_filtered[df_filtered['Produto'].isin(selected_products)]

# ==========================================
# 4. DASHBOARD HEADER
# ==========================================
st.title("📊 Painel Analítico de Vendas - Supermercado")
st.markdown(
    f"Apresentação executiva baseada no desempenho de vendas de **{start_dt.strftime('%d/%m/%Y')}** a **{end_dt.strftime('%d/%m/%Y')}**."
)
st.markdown("---")

# ==========================================
# 5. TAB SYSTEM
# ==========================================
tab_exec, tab_prod, tab_temp, tab_forecast, tab_insights = st.tabs([
    "📈 Visão Geral Executiva",
    "🛍️ Produtos & Categorias",
    "📅 Sazonalidade & Calendário",
    "🔮 Previsão de Vendas (AI)",
    "💡 Insights & Recomendações"
])

# ------------------------------------------
# TAB 1: VISÃO GERAL EXECUTIVA
# ------------------------------------------
with tab_exec:
    if df_filtered.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
    else:
        # Calculate KPI values
        total_rev = df_filtered['Total_Venda'].sum()
        total_transactions = len(df_filtered)
        total_units = df_filtered['Quantidade'].sum()
        avg_tkt = total_rev / total_transactions if total_transactions > 0 else 0
        
        # Grid of KPI Cards
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Faturamento Total</div>
                    <div class="metric-value">R$ {total_rev:,.2f}</div>
                    <div class="metric-delta">▲ Performance Anual</div>
                </div>
            """, unsafe_allow_html=True)
            
        with kpi_col2:
            st.markdown(f"""
                <div class="metric-card" style="border-left-color: #10B981;">
                    <div class="metric-title">Volume de Transações</div>
                    <div class="metric-value">{total_transactions:,}</div>
                    <div class="metric-delta">▲ Cupons Emitidos</div>
                </div>
            """, unsafe_allow_html=True)
            
        with kpi_col3:
            st.markdown(f"""
                <div class="metric-card" style="border-left-color: #F59E0B;">
                    <div class="metric-title">Itens Vendidos</div>
                    <div class="metric-value">{total_units:,} u.</div>
                    <div class="metric-delta">▲ Giro de Gôndola</div>
                </div>
            """, unsafe_allow_html=True)
            
        with kpi_col4:
            st.markdown(f"""
                <div class="metric-card" style="border-left-color: #8B5CF6;">
                    <div class="metric-title">Ticket Médio</div>
                    <div class="metric-value">R$ {avg_tkt:,.2f}</div>
                    <div class="metric-delta">▲ Valor por Compra</div>
                </div>
            """, unsafe_allow_html=True)
            
        # Daily Revenue Plot
        st.subheader("Evolução Diária do Faturamento")
        df_daily_rev = df_filtered.groupby('Data')['Total_Venda'].sum().reset_index()
        
        # Calculate a 7-day rolling average to smooth daily noise
        df_daily_rev['Media_Movel_7D'] = df_daily_rev['Total_Venda'].rolling(window=7, min_periods=1).mean()
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=df_daily_rev['Data'], y=df_daily_rev['Total_Venda'],
            mode='lines', name='Faturamento Diário',
            line=dict(color='#93C5FD', width=1.5),
            hovertemplate='Data: %{x|%d/%m/%Y}<br>Faturamento: R$ %{y:,.2f}<extra></extra>'
        ))
        fig_line.add_trace(go.Scatter(
            x=df_daily_rev['Data'], y=df_daily_rev['Media_Movel_7D'],
            mode='lines', name='Tendência (Média Móvel 7 Dias)',
            line=dict(color='#1E3A8A', width=3),
            hovertemplate='Data: %{x|%d/%m/%Y}<br>Média Móvel: R$ %{y:,.2f}<extra></extra>'
        ))
        
        fig_line.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(showgrid=True, gridcolor='#E5E7EB', tickformat='%d/%m/%Y'),
            yaxis=dict(showgrid=True, gridcolor='#E5E7EB', title='Faturamento (R$)'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.01),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Summary Box
        st.info(
            "💡 **Resumo Analítico:** O faturamento apresenta flutuações cíclicas regulares (semanais). "
            "A linha de tendência escura (Média Móvel de 7 dias) nos ajuda a enxergar se a receita total está "
            "em ascensão, estabilizada ou declínio, eliminando o ruído das vendas de fim de semana."
        )

# ------------------------------------------
# TAB 2: PRODUTOS & CATEGORIAS
# ------------------------------------------
with tab_prod:
    if df_filtered.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
    else:
        col_cat, col_prod = st.columns(2)
        
        with col_cat:
            st.subheader("Participação por Categoria (Share)")
            df_cat = df_filtered.groupby('Categoria')['Total_Venda'].sum().reset_index()
            
            fig_donut = px.pie(
                df_cat, values='Total_Venda', names='Categoria',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_donut.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hovertemplate='Categoria: %{label}<br>Faturamento: R$ %{value:,.2f}<br>Share: %{percent}<extra></extra>'
            )
            fig_donut.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with col_prod:
            st.subheader("Top 10 Produtos por Faturamento")
            df_prod_rev = df_filtered.groupby(['Produto', 'Categoria'])['Total_Venda'].sum().reset_index()
            df_prod_top = df_prod_rev.sort_values(by='Total_Venda', ascending=True).tail(10)
            
            fig_bar_prod = px.bar(
                df_prod_top, x='Total_Venda', y='Produto',
                color='Categoria',
                orientation='h',
                color_discrete_sequence=px.colors.qualitative.Prism,
                labels={'Total_Venda': 'Faturamento (R$)', 'Produto': 'Produto'},
                hover_data={'Total_Venda': ':,.2f'}
            )
            fig_bar_prod.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(showgrid=True, gridcolor='#E5E7EB'),
                yaxis=dict(categoryorder='total ascending'),
                height=350,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig_bar_prod, use_container_width=True)
            
        # Treemap Portfolio Chart
        st.subheader("Mapa de Árvore (Treemap): Distribuição de Faturamento por Categoria e Produto")
        
        fig_treemap = px.treemap(
            df_filtered,
            path=['Categoria', 'Produto'],
            values='Total_Venda',
            color='Categoria',
            color_discrete_sequence=px.colors.qualitative.Prism,
            hover_data={'Total_Venda': ':,.2f'}
        )
        fig_treemap.update_layout(
            margin=dict(t=30, l=10, r=10, b=10),
            height=450
        )
        st.plotly_chart(fig_treemap, use_container_width=True)
        st.info(
            "💡 **Análise de Portfólio (Treemap):** O Mapa de Árvore permite visualizar de forma intuitiva "
            "a hierarquia das vendas. O tamanho de cada retângulo é proporcional ao faturamento. "
            "Você pode clicar em uma Categoria para dar zoom nos produtos específicos daquele setor."
        )

# ------------------------------------------
# TAB 3: SAZONALIDADE & CALENDÁRIO
# ------------------------------------------
with tab_temp:
    if df_filtered.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
    else:
        col_dow, col_month = st.columns(2)
        
        with col_dow:
            st.subheader("Faturamento por Dia da Semana")
            # Correct order of days
            ordem_dias = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
            
            df_dow = df_filtered.groupby('Dia_Semana')['Total_Venda'].sum().reindex(ordem_dias).reset_index()
            
            fig_bar_dow = px.bar(
                df_dow, x='Dia_Semana', y='Total_Venda',
                color='Dia_Semana',
                color_discrete_sequence=['#93C5FD', '#93C5FD', '#93C5FD', '#93C5FD', '#1E3A8A', '#1E3A8A', '#3B82F6'],
                labels={'Total_Venda': 'Faturamento Acumulado (R$)', 'Dia_Semana': 'Dia da Semana'}
            )
            fig_bar_dow.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#E5E7EB'),
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig_bar_dow, use_container_width=True)
            
        with col_month:
            st.subheader("Faturamento Mensal (Mês a Mês)")
            ordem_meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
            
            # Check monthly distribution
            df_monthly = df_filtered.groupby('Nome_Mes')['Total_Venda'].sum().reindex(ordem_meses).reset_index()
            
            fig_monthly = px.line(
                df_monthly, x='Nome_Mes', y='Total_Venda',
                markers=True,
                line_shape='spline',
                labels={'Total_Venda': 'Faturamento (R$)', 'Nome_Mes': 'Mês'}
            )
            fig_monthly.update_traces(line=dict(color='#10B981', width=3), marker=dict(size=8, color='#1E3A8A'))
            fig_monthly.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#E5E7EB'),
                height=350
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
            
        # Daily Average Ticket Analysis
        st.subheader("Análise do Ticket Médio por Dia da Semana")
        df_ticket_dow = df_filtered.groupby('Dia_Semana', observed=False).apply(
            lambda x: x['Total_Venda'].sum() / len(x) if len(x) > 0 else 0
        ).reindex(ordem_dias).reset_index()
        df_ticket_dow.columns = ['Dia_Semana', 'Ticket_Medio']
        
        fig_ticket_dow = px.bar(
            df_ticket_dow, x='Dia_Semana', y='Ticket_Medio',
            color='Ticket_Medio',
            color_continuous_scale='Blues',
            labels={'Ticket_Medio': 'Ticket Médio (R$)', 'Dia_Semana': 'Dia da Semana'}
        )
        fig_ticket_dow.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#E5E7EB'),
            coloraxis_showscale=False,
            height=300
        )
        st.plotly_chart(fig_ticket_dow, use_container_width=True)
        
        st.info(
            "💡 **Análise de Sazonalidade Executiva:** As vendas se concentram fortemente nas **Sextas-feiras** e **Sábados**, "
            "que representam os maiores picos de faturamento. Notavelmente, o ticket médio também tende a flutuar por dia, "
            "o que sugere padrões distintos de cestas de compras (ex: compras maiores/abastecimento no final de semana "
            "e compras de conveniência/emergência durante a semana)."
        )

# ------------------------------------------
# TAB 4: PREVISÃO DE VENDAS (FORECASTING)
# ------------------------------------------
with tab_forecast:
    st.subheader("🔮 Previsão de Demanda com Suavização Exponencial (Holt-Winters)")
    st.markdown(
        "Utilizamos o algoritmo Holt-Winters para modelar a série de faturamento diário. "
        "O modelo aprende automaticamente a tendência anual e o ciclo sazonal semanal (7 dias) "
        "para projetar o faturamento futuro."
    )
    
    # Selection of forecast horizon
    forecast_days = st.slider(
        "Selecione o Horizonte de Previsão (Dias)",
        min_value=15,
        max_value=90,
        value=30,
        step=5
    )
    
    # Process data for forecasting
    # We aggregate total sales per day from raw data to ensure a solid baseline
    df_fc_base = df_raw.groupby('Data')['Total_Venda'].sum().reset_index()
    df_fc_base = df_fc_base.set_index('Data').asfreq('D', fill_value=0)
    
    try:
        # Fit model
        model_hw = ExponentialSmoothing(
            df_fc_base['Total_Venda'],
            trend='add',
            seasonal='add',
            seasonal_periods=7
        )
        fit_hw = model_hw.fit()
        
        # Forecast
        forecast_index = pd.date_range(start=df_fc_base.index[-1] + timedelta(days=1), periods=forecast_days, freq='D')
        forecast_values = fit_hw.forecast(steps=forecast_days)
        
        # Uncertainty intervals based on training residuals
        residuals = fit_hw.resid
        std_error = np.std(residuals)
        steps = np.arange(1, forecast_days + 1)
        # Propagation of error: grows proportionally to square root of time step
        margin_of_error = 1.96 * std_error * np.sqrt(steps)
        
        upper_bound = forecast_values + margin_of_error
        lower_bound = np.clip(forecast_values - margin_of_error, 0, None)
        
        # Create DataFrames for visualization
        df_forecast = pd.DataFrame({
            'Previsao': forecast_values,
            'Limite_Superior': upper_bound,
            'Limite_Inferior': lower_bound
        }, index=forecast_index)
        
        # Plotly chart
        # Show last 90 days for better readability
        df_recent = df_fc_base.iloc[-90:]
        
        fig_fc = go.Figure()
        
        # Historical actual sales
        fig_fc.add_trace(go.Scatter(
            x=df_recent.index, y=df_recent['Total_Venda'],
            mode='lines', name='Histórico Real (Últimos 90 Dias)',
            line=dict(color='#1E3A8A', width=2),
            hovertemplate='Data: %{x|%d/%m/%Y}<br>Venda: R$ %{y:,.2f}<extra></extra>'
        ))
        
        # Predicted sales
        fig_fc.add_trace(go.Scatter(
            x=df_forecast.index, y=df_forecast['Previsao'],
            mode='lines', name='Previsão de Vendas',
            line=dict(color='#10B981', width=2.5, dash='dash'),
            hovertemplate='Data: %{x|%d/%m/%Y}<br>Previsão: R$ %{y:,.2f}<extra></extra>'
        ))
        
        # Confidence interval envelope
        fig_fc.add_trace(go.Scatter(
            x=df_forecast.index.tolist() + df_forecast.index.tolist()[::-1],
            y=df_forecast['Limite_Superior'].tolist() + df_forecast['Limite_Inferior'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Intervalo de Confiança (95%)',
            hoverinfo='skip'
        ))
        
        fig_fc.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(showgrid=True, gridcolor='#E5E7EB'),
            yaxis=dict(showgrid=True, gridcolor='#E5E7EB', title='Faturamento Diário (R$)'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.01),
            height=450
        )
        
        st.plotly_chart(fig_fc, use_container_width=True)
        
        # Projected totals
        projected_rev = forecast_values.sum()
        avg_projected_daily = forecast_values.mean()
        
        col_proj1, col_proj2 = st.columns(2)
        with col_proj1:
            st.metric(
                label="Faturamento Acumulado Projetado",
                value=f"R$ {projected_rev:,.2f}",
                delta=f"Próximos {forecast_days} dias"
            )
        with col_proj2:
            st.metric(
                label="Faturamento Médio Diário Projetado",
                value=f"R$ {avg_projected_daily:,.2f}",
                delta="Projeção Estabilizada"
            )
            
        st.markdown("""
            **Interpretando a Previsão para Tomada de Decisão:**
            1. **Repetição de Padrão Sazonal:** Observe como os dentes da engrenagem (picos e vales) continuam no futuro. O modelo compreendeu que o faturamento de final de semana sempre será maior.
            2. **Planejamento Financeiro:** O Faturamento Acumulado Projetado de **R$ {:,.2f}** serve como orçamento base para fluxo de caixa dos próximos {} dias.
            3. **Margem de Segurança:** O sombreamento verde-claro mostra a variação provável das vendas. Se a linha real sair dessa faixa, significa que um evento atípico (promoção agressiva, quebra de estoque ou feriado) ocorreu.
        """.format(projected_rev, forecast_days))
        
    except Exception as ex:
        st.error(f"Erro ao processar modelo preditivo: {ex}")

# ------------------------------------------
# TAB 5: INSIGHTS & RECOMENDAÇÕES
# ------------------------------------------
with tab_insights:
    st.subheader("💡 Relatório de Insights Analíticos & Plano de Ação")
    st.markdown(
        "Abaixo estão listadas as conclusões obtidas através da análise profunda dos dados de venda, "
        "com o respectivo impacto para o negócio e plano de ação recomendado."
    )
    
    # 4 Structured Insight Cards
    
    st.markdown("""
        <div class="insight-card">
            <div class="insight-header">1. Concentração de Faturamento (Curva A de Categorias)</div>
            <div class="insight-body">
                As categorias <b>Mercearia</b> e <b>Perecíveis</b> lideram de forma expressiva o share de faturamento do supermercado. 
                Essas duas categorias representam a base de sustento do negócio (produtos essenciais e de alto giro).
                <br><br>
                <b>Impacto Comercial:</b> A falta de produtos (ruptura) nestas categorias é fatal para a experiência do cliente, pois são produtos de primeira necessidade.
                <div class="recommendation-badge">
                    <b>Ação Recomendada:</b> Negociar acordos de fornecimento de longo prazo com distribuidores desses setores chaves, obtendo margens melhores e garantindo um estoque de segurança mínimo (safety stock) equivalente a 10 dias de venda média.
                </div>
            </div>
        </div>
        
        <div class="insight-card" style="border-top-color: #3B82F6;">
            <div class="insight-header">2. Janela de Ouro do Final de Semana</div>
            <div class="insight-body">
                Há um pico drástico de vendas e faturamento iniciando na <b>Sexta-feira</b> e estendendo-se até o <b>Sábado</b>. O faturamento de Domingo apresenta uma leve desaceleração mas continua alto comparado aos dias úteis (Segunda a Quarta).
                <br><br>
                <b>Impacto Operacional:</b> Gargalos nas filas do caixa, falta de carrinhos disponíveis e prateleiras vazias durante esses dias geram perda de receita e insatisfação.
                <div class="recommendation-badge">
                    <b>Ação Recomendada:</b> Redimensionar a escala de funcionários (caixas, repositores e limpeza) com foco nos horários de pico de sexta e sábado. Implementar ofertas do tipo "Combos de Final de Semana" para aumentar ainda mais o ticket médio.
                </div>
            </div>
        </div>
        
        <div class="insight-card" style="border-top-color: #F59E0B;">
            <div class="insight-header">3. Oportunidade Ociosa no Meio de Semana</div>
            <div class="insight-body">
                Terça e Quarta-feira representam os dias com menor faturamento acumulado e menor fluxo de transações. 
                Os custos fixos da operação (energia, ar condicionado, funcionários) continuam iguais, tornando estes dias os menos eficientes da operação.
                <br><br>
                <b>Impacto Operacional:</b> Capacidade instalada subutilizada.
                <div class="recommendation-badge">
                    <b>Ação Recomendada:</b> Lançar campanhas de guerrilha no meio de semana. A clássica "Feira do Hortifrúti" às terças e quartas-feiras ou "Dia de Limpeza" às quartas são estratégias comprovadas no varejo para atrair fluxo de clientes para o supermercado nesses dias mornos.
                </div>
            </div>
        </div>
        
        <div class="insight-card" style="border-top-color: #8B5CF6;">
            <div class="insight-header">4. Análise de Cesta e Elasticidade de Bebidas</div>
            <div class="insight-body">
                A categoria de <b>Bebidas</b> (Vinhos, Cervejas, etc.) apresenta um dos maiores valores de Preço Unitário e atua como uma alavanca de Ticket Médio. O comportamento mostra que o volume de bebidas por cupom é menor, mas o valor agregado é muito alto.
                <br><br>
                <b>Impacto de Cross-Selling:</b> Bebidas são altamente correlacionadas a momentos de lazer e acompanhamentos (carne, queijos, petiscos).
                <div class="recommendation-badge">
                    <b>Ação Recomendada:</b> Reposicionar gôndolas de bebidas finas (como Vinhos) próximas às seções de Perecíveis (Queijos finos e Frios) e Mercearia de alto valor para incentivar compras casadas por impulso (Cross-Merchandising).
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
