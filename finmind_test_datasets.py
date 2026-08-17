#!/usr/bin/env python3
"""Test which Free-tier datasets work without a token."""
import requests
import json

BASE = "https://api.finmindtrade.com/api/v4"

def fetch_data(dataset, data_id=None, start_date=None, end_date=None):
    params = {"dataset": dataset}
    if data_id:
        params["data_id"] = data_id
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    resp = requests.get(f"{BASE}/data", params=params, timeout=30)
    return resp.json()

# Test Free-tier datasets with a sample stock_id
sample_stock = "2330"

datasets_to_test = [
    # Technical
    ("TaiwanStockPrice", {"data_id": sample_stock, "start_date": "2016-01-01"}),
    ("TaiwanStockPriceAdj", {"data_id": sample_stock, "start_date": "2016-01-01"}),
    ("TaiwanStockPER", {"data_id": sample_stock, "start_date": "2016-01-01"}),
    ("TaiwanStockTotalReturnIndex", {"data_id": "TAIEX", "start_date": "2016-01-01"}),
    ("TaiwanStockTradingDate", {}),
    ("TaiwanStockDelisting", {}),

    # Chip / Institutional
    ("TaiwanStockInstitutionalInvestorsBuySell", {"data_id": sample_stock, "start_date": "2016-01-01"}),
    ("TaiwanStockMarginPurchaseShortSale", {"data_id": sample_stock, "start_date": "2016-01-01"}),
    ("TaiwanStockShareholding", {"data_id": sample_stock, "start_date": "2016-01-01"}),
    ("TaiwanStockSecuritiesLending", {"data_id": sample_stock, "start_date": "2016-01-01"}),
    ("TaiwanStockTotalInstitutionalInvestors", {"start_date": "2016-01-01"}),
    ("TaiwanStockTotalMarginPurchaseShortSale", {"start_date": "2016-01-01"}),
    ("TaiwanSecuritiesTraderInfo", {}),

    # Fundamental
    ("TaiwanStockFinancialStatements", {"data_id": sample_stock, "start_date": "2016-01-01"}),
    ("TaiwanStockBalanceSheet", {"data_id": sample_stock, "start_date": "2016-01-01"}),
    ("TaiwanStockCashFlowsStatement", {"data_id": sample_stock, "start_date": "2016-01-01"}),
    ("TaiwanStockDividend", {"data_id": sample_stock, "start_date": "2016-01-01"}),
    ("TaiwanStockMonthRevenue", {"data_id": sample_stock, "start_date": "2016-01-01"}),

    # Global Economic
    ("GoldPrice", {"start_date": "2016-01-01"}),
    ("TaiwanExchangeRate", {"data_id": "USD", "start_date": "2016-01-01"}),
    ("ExchangeRate", {"data_id": "Japan", "start_date": "2016-01-01"}),
    ("InterestRate", {"data_id": "FED", "start_date": "2016-01-01"}),
    ("GovernmentBondsYield", {"data_id": "United States 10-Year", "start_date": "2016-01-01"}),
    ("CrudeOilPrices", {"data_id": "Brent", "start_date": "2016-01-01"}),
    ("CurrencyCirculation", {"data_id": "US", "start_date": "2016-01-01"}),

    # Paid-tier (expected to fail)
    ("TaiwanFuturesDaily", {"data_id": "TX", "start_date": "2016-01-01"}),
    ("TaiwanOptionDaily", {"start_date": "2016-01-01"}),
    ("USStockInfo", {}),
    ("USStockPrice", {"data_id": "AAPL", "start_date": "2016-01-01"}),
    ("UKStockInfo", {}),
    ("EuropeStockInfo", {}),
    ("JapanStockInfo", {}),
    ("CnnFearGreedIndex", {"start_date": "2016-01-01"}),
]

results = {}
for ds, params in datasets_to_test:
    try:
        result = fetch_data(ds, **params)
        if result.get("status") == 200:
            data = result.get("data", [])
            n = len(data) if isinstance(data, list) else "N/A"
            results[ds] = {"status": "OK", "records": n}
            print(f"[OK]    {ds}: {n} records, sample={str(data[0])[:100] if data else 'none'}")
        else:
            results[ds] = {"status": "REQUIRES_TIER", "msg": result.get("msg", "unknown")}
            print(f"[TIER]  {ds}: {result.get('msg', 'requires higher tier')[:80]}")
    except Exception as e:
        results[ds] = {"status": "ERROR", "msg": str(e)[:100]}
        print(f"[ERR]   {ds}: {str(e)[:80]}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY:")
free_ok = [k for k, v in results.items() if v["status"] == "OK"]
requires_tier = [k for k, v in results.items() if v["status"] == "REQUIRES_TIER"]
errors = [k for k, v in results.items() if v["status"] == "ERROR"]
print(f"  Free tier (accessible): {len(free_ok)} datasets")
for d in free_ok:
    print(f"    - {d} ({results[d]['records']} records)")
print(f"  Requires Backer/Sponsor: {len(requires_tier)} datasets")
for d in requires_tier:
    print(f"    - {d}")
print(f"  Errors: {len(errors)} datasets")
for d in errors:
    print(f"    - {d}")

# Save results
with open("finmind_dataset_status.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to finmind_dataset_status.json")
