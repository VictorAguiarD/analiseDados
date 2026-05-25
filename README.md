# 📊 Dashboard Executivo de Vendas & Previsão de Demanda

Este repositório contém a solução completa de **Data Analytics** desenvolvida para a diretoria executiva de uma rede de supermercados. O projeto transforma dados brutos de transações de vendas em um painel interativo em tempo real, acompanhado por um modelo estatístico preditivo de demanda e análises de sazonalidade para tomada de decisões estratégicas.

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
Certifique-se de ter o Python 3.8+ instalado em sua máquina.

### Passo 1: Instalar dependências
Instale as bibliotecas necessárias utilizando o gerenciador de pacotes `pip`:
```bash
pip install streamlit pandas numpy plotly statsmodels
```

### Passo 2: Executar o Dashboard
Na pasta raiz do projeto, execute o comando do Streamlit:
```bash
streamlit run app.py
```
O dashboard será aberto automaticamente no seu navegador no endereço: `http://localhost:8501`.

---

## 📂 Estrutura do Projeto

*   **`app.py`**: O arquivo principal da aplicação. Desenvolvido com **Streamlit** e customizado com estilos CSS corporativos para oferecer uma experiência de usuário (UX) executiva. Utiliza **Plotly** para gráficos interativos.
*   **`Analise.ipynb`**: Notebook Jupyter contendo a documentação técnica da análise exploratória de dados (EDA), o pré-processamento de limpeza de dados e a prototipação do modelo de forecasting.
*   **`vendas_supermercado.csv`**: A base de dados histórica contendo informações sobre transações de 2023 (Categoria, Produto, Quantidade, Preço Unitário, Total e Data).
*   **`faturamento_categoria.png`** e **`sazonalidade_semana.png`**: Gráficos analíticos estáticos de referência exportados diretamente pelo notebook.

---

## 🎯 Principais Funcionalidades do Dashboard

O dashboard está estruturado em 5 abas de navegação focadas em valor de negócio:

1.  **📈 Visão Geral Executiva**
    *   **KPI Cards**: Exibição dos indicadores cruciais acumulados: Faturamento Total, Volume de Transações (cupons emitidos), Itens Vendidos e Ticket Médio.
    *   **Tendência Temporal**: Gráfico de linha interativo do faturamento diário, acompanhado de uma média móvel de 7 dias para atenuar as flutuações e revelar a real direção das vendas.
2.  **🛍️ Produtos & Categorias**
    *   **Donut Share**: Participação percentual de cada categoria no faturamento geral.
    *   **Top 10 Produtos**: Gráfico de barras horizontais destacando os itens campeões de venda.
    *   **Treemap Portfolio**: Mapa de árvore dinâmico e clicável que exibe a hierarquia completa de vendas do catálogo. Excelente para entender rapidamente a dominância de produtos dentro de cada setor (zoom in/out interativo).
3.  **📅 Sazonalidade & Calendário**
    *   **Sazonalidade Semanal**: Gráfico que aponta quais dias da semana registram picos de demanda.
    *   **Mês a Mês**: A curva contínua do faturamento mensal ao longo do ano para detecção de datas festivas.
    *   **Ticket Médio Semanal**: Mostra a variação do valor médio gasto por dia da semana.
4.  **🔮 Previsão de Vendas (AI)**
    *   **Modelo Holt-Winters**: Algoritmo estatístico de Suavização Exponencial Tripla, configurado para identificar tendências de longo prazo e sazonalidades cíclicas de 7 dias (semanais).
    *   **Horizonte Dinâmico**: Permite ao usuário escolher o período de previsão (de 15 a 90 dias futuros) por meio de um controle deslizante.
    *   **Bandas de Incerteza**: Exibe a margem de erro estimada do modelo com 95% de confiança.
    *   **Projeções de Orçamento**: Cálculo automático do faturamento acumulado e da média diária esperada para o período selecionado.
5.  **💡 Insights & Recomendações**
    *   Recomendações estruturadas e ações táticas detalhadas para o negócio com foco em:
        *   Planejamento de estoque de segurança e otimização de prateleiras.
        *   Dimensionamento de escalas de funcionários para horários de pico.
        *   Campanhas promocionais direcionadas para reverter dias ociosos da semana.
        *   Estratégias de *Cross-Merchandising* (vendas casadas) para a categoria de Bebidas.

---

## 🧹 Tratamento de Dados (Bug-Fixing)
A base original continha linhas residuais vazias ao final do arquivo e colunas não formatadas. O carregamento de dados foi blindado em ambos os arquivos (`app.py` e `Analise.ipynb`) garantindo que:
*   Apenas as 6 colunas de dados reais sejam analisadas.
*   Registros inteiramente nulos (como a linha 2002 do CSV) sejam descartados via `.dropna()`.
*   Casts numéricos (`int`, `float`) e datas sejam convertidos de forma limpa e otimizada em cache.
