#!/usr/bin/env python3
"""Explore FINMIND datasets and discover available data_ids."""
import requests
import json

BASE = "https://api.finmindtrade.com/api/v4"

def fetch_data(dataset, data_id=None, start_date=None, end_date=None, token=None):
    """Fetch data from FINMIND API."""
    params = {"dataset": dataset}
    if data_id:
        params["data_id"] = data_id
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(f"{BASE}/data", params=params, headers=headers, timeout=30)
    return resp.json()

def fetch_datalist(dataset):
    """Get list of data_ids for a dataset."""
    resp = requests.get(f"{BASE}/datalist", params={"dataset": dataset}, timeout=15)
    return resp.json()

# ============================================
# 1. Get stock list from TaiwanStockInfo
# ============================================
print("=" * 60)
print("1. Fetching TaiwanStockInfo (stock list)...")
result = fetch_data("TaiwanStockInfo")
if result.get("status") == 200:
    stocks = result["data"]
    stock_ids = set(s["stock_id"] for s in stocks)
    print(f"   Total stocks: {len(stocks)}")
    print(f"   Unique stock IDs: {len(stock_ids)}")
    # Show sample
    print(f"   Sample stocks: {stocks[0]}, {stocks[1]}, {stocks[2]}")
    # Save stock list
    with open("taiwan_stocks.json", "w") as f:
        json.dump(stocks, f, indent=2)
    print("   Saved to taiwan_stocks.json")
else:
    print(f"   Error: {result}")

# ============================================
# 2. Get futures contract IDs
# ============================================
print("\n" + "=" * 60)
print("2. Fetching TaiwanFuturesDaily sample (futures codes)...")
result = fetch_data("TaiwanFuturesDaily", start_date="2024-01-01")
if result.get("status") == 200:
    futures_data = result["data"]
    futures_ids = set(r.get("contract_id", "") for r in futures_data)
    print(f"   Futures contracts: {sorted(futures_ids)}")
    print(f"   Sample row: {futures_data[0]}")
else:
    print(f"   Error: {result}")

# ============================================
# 3. Get options contract IDs
# ============================================
print("\n" + "=" * 60)
print("3. Fetching TaiwanOptionDaily sample (options codes)...")
result = fetch_data("TaiwanOptionDaily", start_date="2024-01-01")
if result.get("status") == 200:
    options_data = result["data"]
    options_ids = set(r.get("contract_id", "") for r in options_data)
    print(f"   Options contracts: {sorted(options_ids)[:20]}...")  # limit output
    print(f"   Total unique: {len(options_ids)}")
    print(f"   Sample row: {options_data[0]}")
else:
    print(f"   Error: {result}")

# ============================================
# 4. Get data_ids for data_id-requiring datasets
# ============================================
print("\n" + "=" * 60)
print("4. Testing TaiwanFutOptTickInfo (single-day futures/options tick)...")
result = fetch_data("TaiwanFutOptTickInfo", start_date="2024-01-02")
if result.get("status") == 200:
    tick_data = result["data"]
    print(f"   Total records: {len(tick_data)}")
    print(f"   Sample row: {tick_data[0]}")
    # Get unique contract IDs
    contracts = set(r.get("contract_id", "") for r in tick_data)
    print(f"   Unique contracts: {len(contracts)}")
else:
    print(f"   Error: {result}")

# ============================================
# 5. Get exchange rate data_ids
# ============================================
print("\n" + "=" * 60)
print("5. Exchange rate currencies:")
result = fetch_datalist("TaiwanExchangeRate")
print(f"   {result['data']}")

result = fetch_datalist("ExchangeRate")
print(f"   Global exchange rates: {result['data']}")

# ============================================
# 6. Get interest rate data_ids
# ============================================
print("\n" + "=" * 60)
print("6. Interest rate central banks:")
result = fetch_datalist("InterestRate")
print(f"   {result['data']}")

# ============================================
# 7. Get US government bonds data_ids
# ============================================
print("\n" + "=" * 60)
print("7. US Government bond yields:")
result = fetch_datalist("GovernmentBondsYield")
print(f"   {result['data']}")

# ============================================
# 8. Test USStockPrice
# ============================================
print("\n" + "=" * 60)
print("8. Testing USStockPrice sample...")
result = fetch_data("USStockPrice", start_date="2024-01-01")
if result.get("status") == 200:
    us_data = result["data"]
    us_ids = set(r.get("stock_id", "") for r in us_data)
    print(f"   Total records: {len(us_data)}")
    print(f"   Unique US stock IDs (sample): {sorted(us_ids)[:20]}")
    print(f"   Total unique: {len(us_ids)}")
    print(f"   Sample row: {us_data[0]}")
    with open("us_stocks.json", "w") as f:
        json.dump(list(us_ids), f, indent=2)
else:
    print(f"   Error: {result}")

# ============================================
# 9. Test GoldPrice
# ============================================
print("\n" + "=" * 60)
print("9. Testing GoldPrice...")
result = fetch_data("GoldPrice", start_date="2016-01-01")
if result.get("status") == 200:
    gold_data = result["data"]
    print(f"   Total records: {len(gold_data)}")
    print(f"   Sample rows: {gold_data[:2]}")
else:
    print(f"   Error: {result}")

# ============================================
# 10. Test CnnFearGreedIndex
# ============================================
print("\n" + "=" * 60)
print("10. Testing CnnFearGreedIndex...")
result = fetch_data("CnnFearGreedIndex", start_date="2016-01-01")
if result.get("status") == 200:
    fgi_data = result["data"]
    print(f"   Total records: {len(fgi_data)}")
    print(f"   Sample rows: {fgi_data[:2]}")
else:
    print(f"   Error: {result}")

print("\n" + "=" * 60)
print("Exploration complete!")
