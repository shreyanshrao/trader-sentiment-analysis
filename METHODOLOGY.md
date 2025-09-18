# Methodology: Bitcoin Market Sentiment vs Trader Performance Analysis

## Overview

This document outlines the comprehensive methodology used to analyze the relationship between Bitcoin market sentiment (Fear & Greed Index) and trading performance on Hyperliquid exchange.

## Data Sources

### 1. Fear & Greed Index Data
- **Source**: Alternative.me Bitcoin Fear & Greed Index
- **Frequency**: Daily readings
- **Scale**: 0-100 (0 = Extreme Fear, 100 = Extreme Greed)
- **Categories**: 
  - Extreme Fear (0-24)
  - Fear (25-49)
  - Neutral (50-74)
  - Greed (75-100)
  - Extreme Greed (75-100)

### 2. Hyperliquid Trading Data
- **Source**: Hyperliquid exchange historical trading data
- **Granularity**: Individual trade level
- **Key Fields**:
  - Account ID (anonymized)
  - Trading pair (Coin)
  - Execution price and size
  - Side (BUY/SELL)
  - Timestamp
  - Closed PnL
  - Fees

## Data Processing Pipeline

### Phase 1: Data Ingestion & Validation
1. **File Format Validation**
   - Verify CSV structure and column headers
   - Check for required fields and data types
   - Identify and log any format inconsistencies

2. **Data Quality Assessment**
   - Calculate missing value percentages
   - Identify outliers and anomalies
   - Validate timestamp ranges and formats

### Phase 2: Data Cleaning & Standardization
1. **Timestamp Standardization**
   - Convert all timestamps to consistent datetime format
   - Handle timezone conversions (IST to UTC)
   - Extract date components for temporal alignment

2. **Numeric Field Processing**
   - Convert string numbers to appropriate numeric types
   - Handle scientific notation (e.g., 8.95E+14)
   - Replace invalid values with NaN

3. **Categorical Field Standardization**
   - Normalize sentiment classifications
   - Standardize trading side indicators (BUY/SELL)
   - Clean coin symbols and trading pairs

### Phase 3: Feature Engineering
1. **Sentiment Features**
   - `sentiment_score`: Raw Fear & Greed Index value (0-100)
   - `is_fear`: Boolean flag for Fear/Extreme Fear periods
   - `is_greed`: Boolean flag for Greed/Extreme Greed periods
   - `is_extreme`: Boolean flag for extreme sentiment periods
   - `sentiment_category`: Categorical classification

2. **Trading Features**
   - `is_buy`/`is_sell`: Boolean trade direction flags
   - `pnl_per_token`: Normalized PnL per token traded
   - `is_profitable`: Boolean profitability flag
   - `trade_size_category`: Quantile-based size categorization
   - `hour`: Hour of day for intraday analysis

3. **Risk Metrics**
   - Trade size percentiles (25th, 50th, 75th, 95th)
   - Position size relative to account history
   - Volatility measures by sentiment period

### Phase 4: Temporal Alignment
1. **Date-based Merging**
   - Align trade data with daily sentiment readings
   - Handle missing sentiment data for certain dates
   - Maintain temporal integrity of analysis

2. **Lag Feature Creation**
   - Previous day sentiment (t-1)
   - Sentiment change indicators
   - Multi-day sentiment trends

## Analytical Framework

### 1. Descriptive Statistics
- **Central Tendency**: Mean, median PnL by sentiment
- **Dispersion**: Standard deviation, interquartile ranges
- **Distribution Analysis**: Histograms, box plots, density plots
- **Frequency Analysis**: Trade counts, win rates by condition

### 2. Comparative Analysis
- **Cross-Sentiment Performance**: PnL differences between sentiment states
- **Temporal Patterns**: Hourly, daily, weekly performance variations
- **Asset-Specific Analysis**: Performance by trading pair
- **Side Analysis**: Buy vs Sell performance differences

### 3. Pattern Recognition
1. **Risk-Taking Behavior Analysis**
   - Trade size distribution by sentiment
   - Position sizing patterns
   - Risk appetite indicators

2. **Momentum vs Contrarian Effects**
   - Performance during sentiment transitions
   - Autocorrelation in sentiment-performance relationships
   - Mean reversion vs trend-following behaviors

3. **Extreme Event Analysis**
   - Performance during extreme sentiment periods
   - Volatility clustering around extreme events
   - Recovery patterns post-extreme sentiment

### 4. Statistical Testing
- **Hypothesis Testing**: T-tests for mean differences
- **Effect Size Calculation**: Cohen's d for practical significance
- **Correlation Analysis**: Pearson/Spearman correlation coefficients
- **Distribution Testing**: Kolmogorov-Smirnov tests

## Visualization Strategy

### 1. Exploratory Visualizations
- Box plots for PnL distribution by sentiment
- Scatter plots for sentiment score vs performance
- Heat maps for multi-dimensional relationships
- Time series plots for temporal patterns

### 2. Summary Dashboard
- 9-panel comprehensive overview
- Performance rankings and comparisons
- Risk-return scatter plots
- Distribution overlays

### 3. Interactive Elements
- Hover information for data points
- Filtering by time periods or assets
- Zoom functionality for detailed views
- Export capabilities for further analysis

## Performance Metrics

### 1. Primary Metrics
- **Average PnL**: Mean profit/loss per trade
- **Win Rate**: Percentage of profitable trades
- **Total Volume**: Cumulative trading volume
- **Trade Count**: Number of trades per condition

### 2. Risk-Adjusted Metrics
- **Sharpe-like Ratio**: Return per unit of volatility
- **Maximum Drawdown**: Largest consecutive loss
- **Risk-Adjusted Return**: Return scaled by trade size volatility
- **Consistency Score**: Standard deviation of returns

### 3. Behavioral Metrics
- **Average Trade Size**: Position sizing patterns
- **Trade Frequency**: Activity levels by sentiment
- **Hold Duration**: Average position holding time
- **Leverage Usage**: Risk amplification patterns

## Validation & Robustness Checks

### 1. Data Integrity Validation
- Cross-reference timestamps across datasets
- Verify PnL calculations where possible
- Check for data leakage or forward-looking bias
- Validate sentiment category boundaries

### 2. Statistical Robustness
- Bootstrap confidence intervals
- Outlier sensitivity analysis
- Alternative metric calculations
- Subperiod stability testing

### 3. Practical Validation
- Transaction cost impact assessment
- Market impact considerations
- Liquidity constraint analysis
- Implementation feasibility review

## Limitations & Assumptions

### 1. Data Limitations
- Single exchange data (Hyperliquid only)
- Limited time period coverage
- Potential survivorship bias
- Missing macroeconomic context

### 2. Methodological Assumptions
- Daily sentiment alignment assumption
- Independence of trading decisions
- Stationarity of relationships over time
- Rational actor hypothesis

### 3. Market Structure Assumptions
- Consistent market microstructure
- Stable regulatory environment
- Unchanged fee structures
- Persistent liquidity conditions

## Future Enhancements

### 1. Data Expansion
- Multi-exchange integration
- Longer historical periods
- Higher frequency sentiment data
- Additional sentiment sources

### 2. Analytical Sophistication
- Machine learning model integration
- Regime detection algorithms
- Causal inference techniques
- Real-time analysis capabilities

### 3. Risk Management Integration
- Portfolio-level analysis
- Dynamic position sizing recommendations
- Real-time alert systems
- Stress testing frameworks

---

*This methodology represents a systematic approach to understanding sentiment-performance relationships in cryptocurrency trading. Regular updates and refinements ensure continued relevance and accuracy.*
