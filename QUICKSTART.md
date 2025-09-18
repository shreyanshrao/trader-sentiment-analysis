# Quick Start Guide 🚀

## Prerequisites
- Python 3.7+ installed
- Git installed
- Your data files ready

## Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Your Data Files
Copy your CSV files to the `data/` directory:
- `fear_greed_index.csv` → `data/fear_greed_index.csv`
- `historical_data.csv` → `data/historical_data.csv`

### 3. Run Analysis
```bash
python trader_sentiment_analysis.py
```

That's it! 🎉

## What You'll Get

### Console Output
- Data loading and validation summaries
- Statistical insights by market sentiment
- Hidden pattern discoveries
- Actionable trading recommendations

### Generated Files
- `results/trader_sentiment_analysis.png` - Comprehensive dashboard
- `results/trading_insights_summary.json` - Key metrics
- `results/merged_sentiment_trader_data.csv` - Combined dataset

## Key Findings Preview

Based on your data (211,218 trades):
- **Best Sentiment**: Extreme Greed ($67.89 avg PnL, 46.49% win rate)
- **Strategy**: Short bias recommended (70% performance edge)
- **Opportunity**: Extreme sentiment periods outperform by 24%
- **Momentum**: Greed periods beat Fear periods consistently

## Next Steps

1. **Review Results**: Check the visualization dashboard
2. **Customize Analysis**: Modify the script for specific needs
3. **Extend Data**: Add more exchanges or time periods
4. **Build Strategy**: Implement insights in live trading

## Troubleshooting

### Common Issues

**File Not Found Error**
```
FileNotFoundError: data/fear_greed_index.csv
```
**Solution**: Make sure CSV files are in the `data/` directory with correct names

**Import Error**
```
ModuleNotFoundError: No module named 'pandas'
```
**Solution**: Run `pip install -r requirements.txt`

**Empty Results**
```
❌ No overlapping data found between sentiment and trader datasets!
```
**Solution**: Check date formats and ranges in both datasets

### Need Help?
- Check the detailed [README.md](README.md) for comprehensive documentation
- Review [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for analytical details
- Create an issue in the repository for specific problems

---

**⚡ Pro Tip**: Start with a subset of your data to validate the analysis before running the full dataset!
