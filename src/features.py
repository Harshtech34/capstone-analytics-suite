# src/features.py
def make_sales_monthly(df):
    df = df.copy()
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    agg = df.groupby(['year','month','Product','Region']).agg(
        quantity=('Quantity','sum'),
        avg_price=('Price','mean'),
        total_sales=('Total_Sales','sum')
    ).reset_index()
    return agg
