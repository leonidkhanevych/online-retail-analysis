import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings('ignore')

# Load & clean
print("Loading data...")
df = pd.read_csv('online_retail_II.csv', encoding='utf-8')
print(f"  Raw rows: {len(df):,}")

# Parse dates
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Drop rows with no Customer ID 
df = df.dropna(subset=['Customer ID'])
df['Customer ID'] = df['Customer ID'].astype(int)

# Remove cancellations (Invoice starts with 'C')
df = df[~df['Invoice'].astype(str).str.startswith('C')]

# Remove rows with negative or zero quantity/price
df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]

# Revenue column
df['Revenue'] = df['Quantity'] * df['Price']

print(f"  Clean rows: {len(df):,}")
print(f"  Date range: {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
print(f"  Unique customers: {df['Customer ID'].nunique():,}")
print(f"  Unique products:  {df['StockCode'].nunique():,}")
print(f"  Countries:        {df['Country'].nunique():,}")
print(f"  Total revenue:    £{df['Revenue'].sum():,.2f}\n")

# Monthly revenue trend 
monthly = (df.groupby(df['InvoiceDate'].dt.to_period('M'))
             .agg(revenue=('Revenue', 'sum'),
                  orders=('Invoice', 'nunique'),
                  customers=('Customer ID', 'nunique'))
             .reset_index())
monthly['InvoiceDate'] = monthly['InvoiceDate'].dt.to_timestamp()
monthly['avg_order_value'] = monthly['revenue'] / monthly['orders']

print("── Monthly revenue ──")
print(monthly[['InvoiceDate','revenue','orders']].to_string(index=False))

# Top 10 products by revenue
top_products = (df.groupby('Description')['Revenue']
                  .sum()
                  .sort_values(ascending=False)
                  .head(10)
                  .reset_index())
top_products.columns = ['product', 'revenue']

print("\n── Top 10 products ──")
print(top_products.to_string(index=False))

# Revenue by country
by_country = (df.groupby('Country')['Revenue']
                .sum()
                .sort_values(ascending=False)
                .reset_index())
by_country_ex_uk = by_country[by_country['Country'] != 'United Kingdom'].head(10)

print("\n── Top 10 countries (ex UK) ──")
print(by_country_ex_uk.to_string(index=False))

# RFM Segmentation 
print("\n── Building RFM model ──")
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('Customer ID').agg(
    recency  =('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    frequency=('Invoice',     'nunique'),
    monetary =('Revenue',     'sum')
).reset_index()

# Score each dimension 1–5 (5 = best)
rfm['R'] = pd.qcut(rfm['recency'],   q=5, labels=[5,4,3,2,1]).astype(int)
rfm['F'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5]).astype(int)
rfm['M'] = pd.qcut(rfm['monetary'],  q=5, labels=[1,2,3,4,5]).astype(int)
rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
rfm['RFM_Total'] = rfm['R'] + rfm['F'] + rfm['M']

def segment(row):
    r, f, m = row['R'], row['F'], row['M']
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3:
        return 'Loyal Customers'
    elif r >= 4 and f <= 2:
        return 'New Customers'
    elif r >= 3 and f <= 2 and m >= 3:
        return 'Potential Loyalists'
    elif r == 2 and f >= 3:
        return 'At Risk'
    elif r <= 2 and f >= 3 and m >= 3:
        return 'Cannot Lose Them'
    elif r <= 2 and f <= 2:
        return 'Lost'
    else:
        return 'Needs Attention'

rfm['Segment'] = rfm.apply(segment, axis=1)

seg_summary = (rfm.groupby('Segment')
                  .agg(customers=('Customer ID','count'),
                       avg_recency=('recency','mean'),
                       avg_frequency=('frequency','mean'),
                       avg_monetary=('monetary','mean'))
                  .round(1)
                  .sort_values('avg_monetary', ascending=False)
                  .reset_index())

print(seg_summary.to_string(index=False))

# Plots
print("\nGenerating charts...")
BLUE   = '#1D4ED8'
COLORS = ['#1D4ED8','#2563EB','#3B82F6','#60A5FA','#93C5FD',
          '#BFDBFE','#1E40AF','#1E3A8A','#172554','#DBEAFE']
SEG_COLORS = {
    'Champions':          '#15803D',
    'Loyal Customers':    '#16A34A',
    'New Customers':      '#4ADE80',
    'Potential Loyalists':'#86EFAC',
    'At Risk':            '#F59E0B',
    'Cannot Lose Them':   '#EF4444',
    'Needs Attention':    '#F97316',
    'Lost':               '#9CA3AF',
}

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Online Retail II — Customer & Revenue Analysis\nUCI Dataset · UK Gift Retailer · Dec 2009 – Dec 2011',
             fontsize=15, fontweight='bold', y=1.01)

# Monthly revenue
ax = axes[0, 0]
ax.fill_between(monthly['InvoiceDate'], monthly['revenue'], alpha=0.15, color=BLUE)
ax.plot(monthly['InvoiceDate'], monthly['revenue'], color=BLUE, linewidth=2.5, marker='o', markersize=4)
ax.set_title('Monthly Revenue (£)', fontweight='bold')
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f'£{x/1000:.0f}k'))
ax.tick_params(axis='x', rotation=30)
ax.set_xlabel('')
ax.grid(axis='y', alpha=0.3)

# Monthly unique customers
ax = axes[0, 1]
ax.bar(monthly['InvoiceDate'], monthly['customers'], color=BLUE, alpha=0.8, width=20)
ax.set_title('Monthly Active Customers', fontweight='bold')
ax.tick_params(axis='x', rotation=30)
ax.grid(axis='y', alpha=0.3)

# Top 10 products
ax = axes[0, 2]
ax.barh(top_products['product'].str[:35], top_products['revenue'],
        color=COLORS[:10], alpha=0.9)
ax.set_title('Top 10 Products by Revenue', fontweight='bold')
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f'£{x/1000:.0f}k'))
ax.invert_yaxis()
ax.tick_params(axis='y', labelsize=8)

# Top countries ex UK
ax = axes[1, 0]
ax.bar(by_country_ex_uk['Country'], by_country_ex_uk['Revenue'],
       color=BLUE, alpha=0.85)
ax.set_title('Top 10 Countries by Revenue\n(excl. United Kingdom)', fontweight='bold')
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f'£{x/1000:.0f}k'))
ax.tick_params(axis='x', rotation=40)
ax.grid(axis='y', alpha=0.3)

# RFM segment sizes
ax = axes[1, 1]
seg_counts = rfm['Segment'].value_counts()
colors_pie  = [SEG_COLORS.get(s, '#ccc') for s in seg_counts.index]
wedges, texts, autotexts = ax.pie(
    seg_counts.values,
    labels=seg_counts.index,
    autopct='%1.1f%%',
    colors=colors_pie,
    startangle=140,
    textprops={'fontsize': 8}
)
ax.set_title('Customer Segments (RFM)', fontweight='bold')

# Avg revenue per segment
ax = axes[1, 2]
seg_rev = seg_summary.sort_values('avg_monetary', ascending=True)
bar_colors = [SEG_COLORS.get(s, '#ccc') for s in seg_rev['Segment']]
bars = ax.barh(seg_rev['Segment'], seg_rev['avg_monetary'], color=bar_colors, alpha=0.9)
ax.set_title('Avg Lifetime Revenue by Segment', fontweight='bold')
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f'£{x:,.0f}'))
for bar, val in zip(bars, seg_rev['avg_monetary']):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            f'£{val:,.0f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('retail_analysis.png', dpi=150, bbox_inches='tight')
print("Saved retail_analysis.png")

#  Save clean RFM table as CSV 
rfm.to_csv('rfm_segments.csv', index=False)
print("Saved rfm_segments.csv")

