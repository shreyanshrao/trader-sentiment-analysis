# Bitcoin Market Sentiment vs Trader Performance Analysis

🚀 **A comprehensive analysis exploring the relationship between market sentiment and trading performance using the Fear & Greed Index and Hyperliquid trader data.**

## 📊 Project Overview

This project analyzes over **211,000 trades** from Hyperliquid exchange against daily Bitcoin Fear & Greed Index data to uncover hidden patterns in trader behavior and identify actionable trading insights.

### Key Findings

- **Best Performing Sentiment**: Extreme Greed (Avg PnL: $67.89, Win Rate: 46.49%)
- **Momentum Effect**: Greed periods outperform Fear periods ($53.88 vs $49.21)
- **Extreme Opportunities**: Extreme sentiment periods offer 24% better performance than normal conditions
- **Short Bias**: Sell trades outperform Buy trades significantly ($60.71 vs $35.69)

## 📁 Project Structure

```
trader-sentiment-analysis/
├── README.md                              # This file
├── trader_sentiment_analysis.py           # Main analysis script
├── requirements.txt                       # Python dependencies
├── data/                                  # Data directory
│   ├── sample_fear_greed_index.csv       # Sample sentiment data
│   └── sample_historical_data.csv        # Sample trader data
├── results/                               # Analysis outputs
│   ├── trader_sentiment_analysis.png     # Visualization dashboard
│   ├── merged_sentiment_trader_data.csv  # Combined dataset
│   └── trading_insights_summary.json     # Key metrics summary
└── docs/                                  # Documentation
    ├── METHODOLOGY.md                     # Detailed methodology
    └── INSIGHTS.md                        # Trading insights & recommendations
```

## 🔧 Installation & Setup

### Prerequisites

- Python 3.7+ 
- Git (for version control)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd trader-sentiment-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**
   - Place your Fear & Greed Index CSV in the `data/` directory
   - Place your Hyperliquid trading data CSV in the `data/` directory
   - Update file paths in the script if needed

## 🚀 Usage

### Quick Start

```bash
python trader_sentiment_analysis.py
```

This will:
1. Load and examine both datasets
2. Clean and preprocess the data
3. Merge datasets on temporal alignment
4. Perform comprehensive exploratory analysis
5. Discover hidden patterns
6. Generate visualizations
7. Export actionable insights

### Expected Data Formats

#### Fear & Greed Index Data
```csv
timestamp,value,classification,date
1517463000,30,Fear,2018-02-01
1517549400,15,Extreme Fear,2018-02-02
```

#### Hyperliquid Trading Data
```csv
Account,Coin,Execution Price,Size Tokens,Size USD,Side,Timestamp IST,Start Position,Direction,Closed PnL,Transaction Hash,Order ID,Crossed,Fee,Trade ID,Timestamp
0xae5e...,@107,7.9769,986.87,7872.16,BUY,02-12-2024 22:50,0,Buy,0,0xec09...,52017706630,TRUE,0.34540448,8.95E+14,1.73E+12
```

## 📈 Analysis Components

### 1. Data Preprocessing
- Timestamp alignment and standardization
- Missing value handling
- Feature engineering (sentiment categories, trade size buckets, etc.)

### 2. Exploratory Data Analysis
- PnL statistics by market sentiment
- Win rate analysis across different conditions
- Volume distribution patterns
- Risk-taking behavior analysis

### 3. Pattern Discovery
- **Risk-Taking Behavior**: How trade sizes vary with sentiment
- **Momentum Effects**: Performance during sentiment transitions
- **Extreme Sentiment Analysis**: Contrarian vs momentum opportunities
- **Time-Based Patterns**: Hourly performance variations
- **Asset-Specific Performance**: How different coins perform under various sentiments

### 4. Visualization Dashboard
A comprehensive 9-panel visualization including:
- PnL distribution by sentiment
- Win rates across market conditions
- Trading volume patterns
- Risk distribution analysis
- Performance heatmaps

## 🎯 Key Insights & Trading Recommendations

### Market Sentiment Performance Ranking
1. **Extreme Greed** - $67.89 avg PnL (46.49% win rate)
2. **Fear** - $54.29 avg PnL (42.08% win rate)  
3. **Greed** - $42.74 avg PnL (38.48% win rate)
4. **Extreme Fear** - $34.54 avg PnL (37.06% win rate)
5. **Neutral** - $34.31 avg PnL (39.70% win rate)

### Trading Strategy Insights

#### 🔄 Momentum Strategy
- **Signal**: Greed periods outperform Fear periods
- **Action**: Consider following the trend during sentiment shifts
- **Performance Edge**: +9.5% better average PnL

#### ⚡ Extreme Sentiment Strategy  
- **Signal**: Extreme conditions (Fear/Greed) outperform normal
- **Action**: Increase position sizes during extreme sentiment periods
- **Performance Edge**: +24% better average PnL

#### 📉 Short Bias Strategy
- **Signal**: Sell trades consistently outperform Buy trades
- **Action**: Favor short positions or selling into strength
- **Performance Edge**: +70% better average PnL

### Risk Management Guidelines
- Scale up position sizes during favorable sentiment conditions
- Extreme sentiment periods offer better risk-adjusted returns
- Monitor hourly patterns for optimal entry/exit timing

## 📊 Output Files

After running the analysis, you'll get:

1. **`trader_sentiment_analysis.png`** - Comprehensive visualization dashboard
2. **`merged_sentiment_trader_data.csv`** - Combined dataset for further analysis
3. **`trading_insights_summary.json`** - Key metrics in JSON format

## 🔬 Methodology

The analysis follows a systematic approach:

1. **Data Integration**: Temporal alignment of sentiment and trading data
2. **Feature Engineering**: Creation of sentiment categories and trading metrics
3. **Statistical Analysis**: Comprehensive performance metrics calculation
4. **Pattern Recognition**: Identification of hidden behavioral patterns
5. **Visualization**: Multi-dimensional data exploration
6. **Insight Generation**: Actionable trading recommendations

## ⚠️ Important Disclaimers

- **Past Performance Warning**: Historical results don't guarantee future performance
- **Market Dynamics**: Crypto markets are highly volatile and unpredictable
- **Risk Management**: Always use proper position sizing and stop-losses
- **Transaction Costs**: Consider fees and slippage in real trading
- **Data Limitations**: Analysis based on specific time period and exchange data

## 🛠️ Customization & Extension Ideas

### Data Sources
- Add multiple exchange data sources
- Incorporate social sentiment (Twitter, Reddit)
- Include macro-economic indicators
- Use real-time sentiment feeds

### Analysis Enhancements
- Add machine learning models for prediction
- Implement advanced statistical tests
- Create backtesting framework
- Add portfolio-level analysis

### Visualization Improvements
- Interactive dashboards with Plotly/Dash
- Real-time monitoring dashboards
- Mobile-responsive charts
- Animated time-series visualizations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

For questions, suggestions, or support:
- Create an issue in this repository
- Reach out via email: [your-email@example.com]

## 🙏 Acknowledgments

- **Fear & Greed Index**: Alternative.me for sentiment data
- **Hyperliquid**: For providing comprehensive trading data
- **Python Community**: For excellent data analysis libraries

## 📈 Future Roadmap

- [ ] Real-time analysis pipeline
- [ ] Machine learning prediction models  
- [ ] Multi-exchange support
- [ ] Portfolio optimization features
- [ ] Risk management tools
- [ ] Mobile app development
- [ ] API integration for live trading

---

**⭐ Star this repository if you found it useful!**

*Disclaimer: This analysis is for educational purposes only. Always consult with financial advisors and do your own research before making investment decisions.*
