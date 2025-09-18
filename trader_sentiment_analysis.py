import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_examine_data():
    """Load both datasets and examine their structure"""
    print("=" * 60)
    print("LOADING AND EXAMINING DATASETS")
    print("=" * 60)
    
    # Load Fear & Greed Index data
    sentiment_df = pd.read_csv('data/fear_greed_index.csv')
    print(f"\n📊 SENTIMENT DATA (Fear & Greed Index)")
    print(f"Shape: {sentiment_df.shape}")
    print(f"Columns: {list(sentiment_df.columns)}")
    print(f"Date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
    
    print("\nSample data:")
    print(sentiment_df.head())
    
    print(f"\nSentiment distribution:")
    print(sentiment_df['classification'].value_counts())
    
    # Load trader data
    trader_df = pd.read_csv('data/historical_data.csv')
    print(f"\n💹 TRADER DATA (Hyperliquid)")
    print(f"Shape: {trader_df.shape}")
    print(f"Columns: {list(trader_df.columns)}")
    
    print("\nSample data:")
    print(trader_df.head())
    
    print(f"\nTrading symbols:")
    print(trader_df['Coin'].value_counts())
    
    print(f"\nSide distribution:")
    print(trader_df['Side'].value_counts())
    
    return sentiment_df, trader_df

def clean_and_preprocess_data(sentiment_df, trader_df):
    """Clean and preprocess both datasets"""
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING AND CLEANING")
    print("=" * 60)
    
    # Clean sentiment data
    sentiment_clean = sentiment_df.copy()
    sentiment_clean['date'] = pd.to_datetime(sentiment_clean['date'])
    sentiment_clean['timestamp'] = pd.to_datetime(sentiment_clean['timestamp'], unit='s')
    
    # Clean trader data
    trader_clean = trader_df.copy()
    
    # Handle timestamp conversion for trader data
    try:
        trader_clean['trade_datetime'] = pd.to_datetime(trader_clean['Timestamp IST'], format='%d-%m-%Y %H:%M')
    except:
        try:
            trader_clean['trade_datetime'] = pd.to_datetime(trader_clean['Timestamp IST'])
        except:
            print("Warning: Could not parse trader timestamps")
            trader_clean['trade_datetime'] = pd.NaT
    
    # Extract date for merging
    trader_clean['trade_date'] = trader_clean['trade_datetime'].dt.date
    
    # Convert numeric columns
    numeric_cols = ['Execution Price', 'Size Tokens', 'Size USD', 'Closed PnL', 'Fee']
    for col in numeric_cols:
        if col in trader_clean.columns:
            trader_clean[col] = pd.to_numeric(trader_clean[col], errors='coerce')
    
    # Create derived features
    print("\n🔧 Creating derived features...")
    
    # For sentiment data
    sentiment_clean['sentiment_score'] = sentiment_clean['value']
    sentiment_clean['is_fear'] = sentiment_clean['classification'].isin(['Fear', 'Extreme Fear'])
    sentiment_clean['is_greed'] = sentiment_clean['classification'].isin(['Greed', 'Extreme Greed'])
    sentiment_clean['is_extreme'] = sentiment_clean['classification'].isin(['Extreme Fear', 'Extreme Greed'])
    
    # For trader data
    trader_clean['is_buy'] = trader_clean['Side'] == 'BUY'
    trader_clean['is_sell'] = trader_clean['Side'] == 'SELL'
    trader_clean['pnl_per_token'] = trader_clean['Closed PnL'] / trader_clean['Size Tokens']
    trader_clean['is_profitable'] = trader_clean['Closed PnL'] > 0
    trader_clean['trade_size_category'] = pd.cut(trader_clean['Size USD'], 
                                                bins=[0, 100, 500, 2000, float('inf')], 
                                                labels=['Small', 'Medium', 'Large', 'XLarge'])
    
    print(f"✅ Sentiment data cleaned: {sentiment_clean.shape[0]} records")
    print(f"✅ Trader data cleaned: {trader_clean.shape[0]} records")
    print(f"✅ Date range overlap check needed...")
    
    return sentiment_clean, trader_clean

def merge_datasets(sentiment_df, trader_df):
    """Merge datasets on date for correlation analysis"""
    print("\n" + "=" * 60)
    print("MERGING DATASETS ON TEMPORAL ALIGNMENT")
    print("=" * 60)
    
    # Convert sentiment date to date object for merging
    sentiment_df['merge_date'] = sentiment_df['date'].dt.date
    
    # Merge on date
    merged_df = trader_df.merge(
        sentiment_df[['merge_date', 'value', 'classification', 'sentiment_score', 
                     'is_fear', 'is_greed', 'is_extreme']], 
        left_on='trade_date', 
        right_on='merge_date', 
        how='inner'
    )
    
    print(f"✅ Merged dataset created: {merged_df.shape[0]} records")
    print(f"📅 Date range: {merged_df['trade_date'].min()} to {merged_df['trade_date'].max()}")
    
    # Check data availability
    print(f"\nData distribution by sentiment:")
    print(merged_df['classification'].value_counts())
    
    return merged_df

def perform_exploratory_analysis(merged_df):
    """Perform comprehensive exploratory data analysis"""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    print("\n📈 BASIC STATISTICS")
    print("-" * 40)
    
    # PnL statistics by sentiment
    pnl_by_sentiment = merged_df.groupby('classification').agg({
        'Closed PnL': ['count', 'mean', 'median', 'std', 'sum'],
        'Size USD': ['mean', 'median'],
        'is_profitable': 'mean'
    }).round(4)
    
    print("\n💰 PnL Statistics by Market Sentiment:")
    print(pnl_by_sentiment)
    
    # Trading volume by sentiment
    volume_by_sentiment = merged_df.groupby('classification').agg({
        'Size USD': 'sum',
        'Size Tokens': 'sum'
    }).round(2)
    
    print("\n📊 Trading Volume by Sentiment:")
    print(volume_by_sentiment)
    
    # Win rate analysis
    win_rate_analysis = merged_df.groupby(['classification', 'Side']).agg({
        'is_profitable': ['mean', 'count'],
        'Closed PnL': 'mean'
    }).round(4)
    
    print("\n🎯 Win Rate Analysis by Sentiment and Side:")
    print(win_rate_analysis)
    
    return {
        'pnl_stats': pnl_by_sentiment,
        'volume_stats': volume_by_sentiment,
        'win_rate_stats': win_rate_analysis
    }

def discover_hidden_patterns(merged_df):
    """Identify hidden patterns in trader behavior during different sentiment periods"""
    print("\n" + "=" * 60)
    print("PATTERN DISCOVERY AND INSIGHTS")
    print("=" * 60)
    
    insights = {}
    
    # Pattern 1: Risk-taking behavior in different sentiment periods
    print("\n🔍 PATTERN 1: Risk-Taking Behavior")
    print("-" * 40)
    
    risk_analysis = merged_df.groupby('classification').agg({
        'Size USD': ['mean', 'std', lambda x: (x > x.quantile(0.95)).mean()],  # % of very large trades
        'Closed PnL': ['mean', 'std'],
    }).round(4)
    
    print("Risk metrics by sentiment (avg trade size, volatility, % large trades):")
    print(risk_analysis)
    insights['risk_behavior'] = risk_analysis
    
    # Pattern 2: Sentiment momentum effect
    print("\n🔍 PATTERN 2: Sentiment Momentum Effect")
    print("-" * 40)
    
    # Group consecutive sentiment periods
    merged_df_sorted = merged_df.sort_values('trade_date')
    merged_df_sorted['prev_sentiment'] = merged_df_sorted['classification'].shift(1)
    merged_df_sorted['sentiment_change'] = (merged_df_sorted['classification'] != merged_df_sorted['prev_sentiment'])
    
    momentum_analysis = merged_df_sorted.groupby(['prev_sentiment', 'classification']).agg({
        'Closed PnL': 'mean',
        'is_profitable': 'mean',
        'Size USD': 'mean'
    }).round(4)
    
    print("Performance when sentiment changes vs stays same:")
    print(momentum_analysis)
    insights['momentum_effect'] = momentum_analysis
    
    # Pattern 3: Extreme sentiment contrarian opportunities
    print("\n🔍 PATTERN 3: Extreme Sentiment Analysis")
    print("-" * 40)
    
    extreme_analysis = merged_df.groupby('is_extreme').agg({
        'Closed PnL': ['mean', 'median'],
        'is_profitable': 'mean',
        'Size USD': 'mean'
    }).round(4)
    
    print("Performance during extreme vs normal sentiment:")
    print(extreme_analysis)
    insights['extreme_sentiment'] = extreme_analysis
    
    # Pattern 4: Time-based patterns
    print("\n🔍 PATTERN 4: Time-Based Patterns")
    print("-" * 40)
    
    # Add hour of day analysis if datetime is available
    if 'trade_datetime' in merged_df.columns and not merged_df['trade_datetime'].isna().all():
        merged_df['hour'] = merged_df['trade_datetime'].dt.hour
        hourly_performance = merged_df.groupby(['hour', 'classification']).agg({
            'Closed PnL': 'mean',
            'is_profitable': 'mean'
        }).round(4)
        
        print("Performance by hour and sentiment (sample):")
        print(hourly_performance.head(20))
        insights['hourly_patterns'] = hourly_performance
    
    # Pattern 5: Trading pair performance by sentiment
    print("\n🔍 PATTERN 5: Asset Performance by Sentiment")
    print("-" * 40)
    
    asset_sentiment_performance = merged_df.groupby(['Coin', 'classification']).agg({
        'Closed PnL': ['mean', 'count'],
        'is_profitable': 'mean',
        'Size USD': 'mean'
    }).round(4)
    
    print("Performance by asset and sentiment:")
    print(asset_sentiment_performance)
    insights['asset_performance'] = asset_sentiment_performance
    
    return insights

def create_visualizations(merged_df, insights):
    """Create comprehensive visualizations"""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Set up the plotting area
    plt.figure(figsize=(20, 16))
    
    # 1. PnL Distribution by Sentiment
    plt.subplot(3, 3, 1)
    sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    available_sentiments = [s for s in sentiment_order if s in merged_df['classification'].values]
    
    sns.boxplot(data=merged_df, x='classification', y='Closed PnL', order=available_sentiments)
    plt.title('PnL Distribution by Market Sentiment')
    plt.xticks(rotation=45)
    plt.ylabel('Closed PnL ($)')
    
    # 2. Win Rate by Sentiment
    plt.subplot(3, 3, 2)
    win_rates = merged_df.groupby('classification')['is_profitable'].mean()
    win_rates = win_rates.reindex(available_sentiments)
    win_rates.plot(kind='bar', color='skyblue')
    plt.title('Win Rate by Market Sentiment')
    plt.ylabel('Win Rate')
    plt.xticks(rotation=45)
    
    # 3. Average Trade Size by Sentiment
    plt.subplot(3, 3, 3)
    avg_size = merged_df.groupby('classification')['Size USD'].mean()
    avg_size = avg_size.reindex(available_sentiments)
    avg_size.plot(kind='bar', color='lightcoral')
    plt.title('Average Trade Size by Sentiment')
    plt.ylabel('Average Size (USD)')
    plt.xticks(rotation=45)
    
    # 4. Sentiment Score vs PnL Scatter
    plt.subplot(3, 3, 4)
    plt.scatter(merged_df['sentiment_score'], merged_df['Closed PnL'], alpha=0.5)
    plt.xlabel('Fear & Greed Index Score')
    plt.ylabel('Closed PnL ($)')
    plt.title('Sentiment Score vs Trading PnL')
    
    # 5. Trading Volume by Sentiment
    plt.subplot(3, 3, 5)
    volume_by_sentiment = merged_df.groupby('classification')['Size USD'].sum()
    volume_by_sentiment = volume_by_sentiment.reindex(available_sentiments)
    volume_by_sentiment.plot(kind='bar', color='gold')
    plt.title('Total Trading Volume by Sentiment')
    plt.ylabel('Total Volume (USD)')
    plt.xticks(rotation=45)
    
    # 6. Buy vs Sell Performance by Sentiment
    plt.subplot(3, 3, 6)
    buy_sell_performance = merged_df.groupby(['classification', 'Side'])['Closed PnL'].mean().unstack()
    buy_sell_performance = buy_sell_performance.reindex(available_sentiments)
    buy_sell_performance.plot(kind='bar')
    plt.title('Buy vs Sell Performance by Sentiment')
    plt.ylabel('Average PnL ($)')
    plt.xticks(rotation=45)
    plt.legend(title='Side')
    
    # 7. Risk Distribution (Trade Size Categories)
    plt.subplot(3, 3, 7)
    if 'trade_size_category' in merged_df.columns:
        risk_dist = merged_df.groupby(['classification', 'trade_size_category']).size().unstack(fill_value=0)
        risk_dist_pct = risk_dist.div(risk_dist.sum(axis=1), axis=0)
        risk_dist_pct = risk_dist_pct.reindex(available_sentiments)
        risk_dist_pct.plot(kind='bar', stacked=True)
        plt.title('Trade Size Distribution by Sentiment')
        plt.ylabel('Proportion')
        plt.xticks(rotation=45)
        plt.legend(title='Size Category', bbox_to_anchor=(1.05, 1))
    
    # 8. Extreme Sentiment Performance
    plt.subplot(3, 3, 8)
    extreme_perf = merged_df.groupby('is_extreme')[['Closed PnL', 'is_profitable']].mean()
    extreme_perf.plot(kind='bar')
    plt.title('Extreme vs Normal Sentiment Performance')
    plt.xticks([0, 1], ['Normal', 'Extreme'], rotation=0)
    plt.legend(['Avg PnL', 'Win Rate'])
    
    # 9. Asset Performance Heatmap
    plt.subplot(3, 3, 9)
    if len(merged_df['Coin'].unique()) > 1:
        asset_sentiment_matrix = merged_df.groupby(['Coin', 'classification'])['Closed PnL'].mean().unstack()
        if asset_sentiment_matrix.shape[0] > 0 and asset_sentiment_matrix.shape[1] > 0:
            sns.heatmap(asset_sentiment_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
            plt.title('Asset Performance Heatmap by Sentiment')
    else:
        plt.text(0.5, 0.5, 'Single Asset Dataset', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Asset Performance (Single Asset)')
    
    plt.tight_layout()
    plt.savefig('results/trader_sentiment_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations saved as 'results/trader_sentiment_analysis.png'")

def generate_trading_insights(merged_df, insights):
    """Generate actionable trading insights"""
    print("\n" + "=" * 60)
    print("ACTIONABLE TRADING INSIGHTS")
    print("=" * 60)
    
    # Key insights
    total_trades = len(merged_df)
    overall_win_rate = merged_df['is_profitable'].mean()
    overall_avg_pnl = merged_df['Closed PnL'].mean()
    
    print(f"\n📊 OVERVIEW")
    print(f"Total Trades Analyzed: {total_trades:,}")
    print(f"Overall Win Rate: {overall_win_rate:.2%}")
    print(f"Overall Average PnL: ${overall_avg_pnl:.2f}")
    
    # Sentiment-based insights
    print(f"\n💡 KEY INSIGHTS")
    print("-" * 40)
    
    # Best performing sentiment
    sentiment_performance = merged_df.groupby('classification')['Closed PnL'].mean().sort_values(ascending=False)
    best_sentiment = sentiment_performance.index[0]
    worst_sentiment = sentiment_performance.index[-1]
    
    print(f"🎯 Best Performing Sentiment: {best_sentiment}")
    print(f"   - Average PnL: ${sentiment_performance.iloc[0]:.2f}")
    print(f"   - Win Rate: {merged_df[merged_df['classification'] == best_sentiment]['is_profitable'].mean():.2%}")
    
    print(f"\n⚠️ Worst Performing Sentiment: {worst_sentiment}")
    print(f"   - Average PnL: ${sentiment_performance.iloc[-1]:.2f}")
    print(f"   - Win Rate: {merged_df[merged_df['classification'] == worst_sentiment]['is_profitable'].mean():.2%}")
    
    # Contrarian opportunities
    fear_performance = merged_df[merged_df['is_fear']]['Closed PnL'].mean()
    greed_performance = merged_df[merged_df['is_greed']]['Closed PnL'].mean()
    
    if fear_performance > greed_performance:
        print(f"\n🔄 CONTRARIAN SIGNAL DETECTED:")
        print(f"   Fear periods show better performance (${fear_performance:.2f}) than greed periods (${greed_performance:.2f})")
        print(f"   Consider buying during fear and selling during greed")
    else:
        print(f"\n📈 MOMENTUM SIGNAL DETECTED:")
        print(f"   Greed periods show better performance (${greed_performance:.2f}) than fear periods (${fear_performance:.2f})")
        print(f"   Consider following the trend")
    
    # Risk management insights
    extreme_performance = merged_df[merged_df['is_extreme']]['Closed PnL'].mean()
    normal_performance = merged_df[~merged_df['is_extreme']]['Closed PnL'].mean()
    
    print(f"\n⚡ EXTREME SENTIMENT ANALYSIS:")
    if extreme_performance > normal_performance:
        print(f"   Extreme sentiment periods offer better opportunities")
        print(f"   Extreme: ${extreme_performance:.2f} vs Normal: ${normal_performance:.2f}")
    else:
        print(f"   Normal sentiment periods are safer")
        print(f"   Normal: ${normal_performance:.2f} vs Extreme: ${extreme_performance:.2f}")
    
    # Trading recommendations
    print(f"\n🎯 TRADING RECOMMENDATIONS")
    print("-" * 40)
    
    # Asset-specific recommendations
    if len(merged_df['Coin'].unique()) > 1:
        asset_performance = merged_df.groupby('Coin')['Closed PnL'].mean().sort_values(ascending=False)
        print(f"1. Best performing asset: {asset_performance.index[0]} (${asset_performance.iloc[0]:.2f} avg PnL)")
        print(f"2. Focus on {asset_performance.index[0]} during {best_sentiment} periods")
    else:
        single_asset = merged_df['Coin'].iloc[0]
        print(f"1. Single asset analysis for {single_asset}")
        print(f"2. Best sentiment for {single_asset}: {best_sentiment}")
    
    # Side-specific recommendations
    buy_performance = merged_df[merged_df['Side'] == 'BUY']['Closed PnL'].mean()
    sell_performance = merged_df[merged_df['Side'] == 'SELL']['Closed PnL'].mean()
    
    if buy_performance > sell_performance:
        print(f"3. Long bias recommended (Buy avg: ${buy_performance:.2f}, Sell avg: ${sell_performance:.2f})")
    else:
        print(f"3. Short bias recommended (Sell avg: ${sell_performance:.2f}, Buy avg: ${buy_performance:.2f})")
    
    # Size-based recommendations
    large_trades = merged_df[merged_df['Size USD'] > merged_df['Size USD'].quantile(0.75)]
    small_trades = merged_df[merged_df['Size USD'] <= merged_df['Size USD'].quantile(0.25)]
    
    large_performance = large_trades['Closed PnL'].mean()
    small_performance = small_trades['Closed PnL'].mean()
    
    if large_performance > small_performance:
        print(f"4. Size up during favorable conditions (Large trades: ${large_performance:.2f})")
    else:
        print(f"4. Keep position sizes moderate (Small trades perform better: ${small_performance:.2f})")
    
    print(f"\n🚨 RISK WARNINGS")
    print("-" * 40)
    print(f"• Past performance doesn't guarantee future results")
    print(f"• Market conditions can change rapidly")
    print(f"• Always use proper risk management")
    print(f"• Consider transaction costs in real trading")
    
    # Export summary
    summary_stats = {
        'Total Trades': total_trades,
        'Overall Win Rate': f"{overall_win_rate:.2%}",
        'Overall Avg PnL': f"${overall_avg_pnl:.2f}",
        'Best Sentiment': best_sentiment,
        'Best Sentiment PnL': f"${sentiment_performance.iloc[0]:.2f}",
        'Worst Sentiment': worst_sentiment,
        'Worst Sentiment PnL': f"${sentiment_performance.iloc[-1]:.2f}",
        'Fear Avg PnL': f"${fear_performance:.2f}",
        'Greed Avg PnL': f"${greed_performance:.2f}",
        'Extreme Sentiment PnL': f"${extreme_performance:.2f}",
        'Normal Sentiment PnL': f"${normal_performance:.2f}"
    }
    
    return summary_stats

def main():
    """Main analysis pipeline"""
    print("🚀 BITCOIN MARKET SENTIMENT vs TRADER PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Load and examine data
    sentiment_df, trader_df = load_and_examine_data()
    
    # Clean and preprocess
    sentiment_clean, trader_clean = clean_and_preprocess_data(sentiment_df, trader_df)
    
    # Merge datasets
    merged_df = merge_datasets(sentiment_clean, trader_clean)
    
    if len(merged_df) == 0:
        print("❌ No overlapping data found between sentiment and trader datasets!")
        return
    
    # Perform exploratory analysis
    stats = perform_exploratory_analysis(merged_df)
    
    # Discover hidden patterns
    insights = discover_hidden_patterns(merged_df)
    
    # Create visualizations
    create_visualizations(merged_df, insights)
    
    # Generate trading insights
    summary = generate_trading_insights(merged_df, insights)
    
    # Save results
    merged_df.to_csv('results/merged_sentiment_trader_data.csv', index=False)
    
    import json
    with open('results/trading_insights_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Files created:")
    print(f"   - results/merged_sentiment_trader_data.csv")
    print(f"   - results/trading_insights_summary.json")
    print(f"   - results/trader_sentiment_analysis.png")

if __name__ == "__main__":
    main()
