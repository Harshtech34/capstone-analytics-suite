# notebooks/1_data_cleaning.py
from src.data_loader import load_sales, load_house_prices, load_churn, save_processed
from src.cleaning import clean_sales, clean_house, clean_churn

if __name__ == "__main__":
    sales = load_sales()
    sales_clean = clean_sales(sales)
    save_processed(sales_clean, "sales_clean.csv")

    houses = load_house_prices()
    houses_clean = clean_house(houses)
    save_processed(houses_clean, "houses_clean.csv")

    churn = load_churn()
    churn_clean = clean_churn(churn)
    save_processed(churn_clean, "churn_clean.csv")

    print("Data cleaning finished.")
