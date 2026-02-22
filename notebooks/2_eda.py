# notebooks/2_eda.py
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_sales
from src.cleaning import clean_sales

if __name__ == "__main__":
    sales = clean_sales(load_sales())
    prod = sales.groupby('Product')['Total_Sales'].sum().sort_values(ascending=False)
    plt.figure(figsize=(8,4))
    sns.barplot(x=prod.index, y=prod.values)
    plt.title("Total Sales by Product")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.savefig("reports/sales_by_product.png")
    print("Saved reports/sales_by_product.png")
