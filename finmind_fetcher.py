#!/usr/bin/env python3
"""
FINMIND Data Fetcher - Fetches 10 years of data across all categories.

Usage:
  python finmind_fetcher.py [--token YOUR_TOKEN] [--output ./finmind_data] [--max-stocks N] [--rate-limit N]

Without a token:
  - 300 requests/hour for stock-level datasets
  - Some datasets require Backer/Sponsor tier (marked as SKIP)
"""
import argparse
import json
import os
import time
import requests
import pandas as pd
from datetime import datetime, date
from pathlib import Path

BASE_URL = "https://api.finmindtrade.com/api/v4"


# ============================================================
# Dataset Registry
# ============================================================
# All datasets categorized. "tier" = access level required.
# "needs_data_id" = whether a data_id (stock/currency/etc) is needed.
# "data_id_source" = how to obtain the data_id list.

DATASETS = {
    "taiwan_technical": {
        "description": "Taiwan Market - Technical",
        "datasets": [
            {"name": "TaiwanStockInfo", "tier": "free", "needs_data_id": False, "type": "stock_list"},
            {"name": "TaiwanStockInfoWithWarrant", "tier": "sponsor", "needs_data_id": False, "type": "stock_list"},
            {"name": "TaiwanStockTradingDate", "tier": "free", "needs_data_id": False, "type": "dates"},
            {"name": "TaiwanStockPrice", "tier": "free", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockPriceAdj", "tier": "backer", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockPER", "tier": "free", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockDayTrading", "tier": "free", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockTotalReturnIndex", "tier": "free", "needs_data_id": True, "type": "index", "data_ids": ["TAIEX", "TPEx"]},
            {"name": "TaiwanVariousIndicators5Seconds", "tier": "free", "needs_data_id": False, "type": "market"},
            {"name": "TaiwanStockPriceLimit", "tier": "free", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStock10Year", "tier": "backer", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockWeekPrice", "tier": "backer", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockMonthPrice", "tier": "backer", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockKBar", "tier": "sponsor", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockDelisting", "tier": "free", "needs_data_id": False, "type": "delisting"},
        ],
    },
    "taiwan_chip_institutional": {
        "description": "Taiwan Market - Chip / Institutional",
        "datasets": [
            {"name": "TaiwanStockMarginPurchaseShortSale", "tier": "free", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockTotalMarginPurchaseShortSale", "tier": "free", "needs_data_id": False, "type": "market"},
            {"name": "TaiwanStockInstitutionalInvestorsBuySell", "tier": "free", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockInstitutionalInvestorsBuySellWide", "tier": "free", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockTotalInstitutionalInvestors", "tier": "free", "needs_data_id": False, "type": "market"},
            {"name": "TaiwanStockShareholding", "tier": "free", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockHoldingSharesPer", "tier": "backer", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockSecuritiesLending", "tier": "free", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockMarginShortSaleSuspension", "tier": "free", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanDailyShortSaleBalances", "tier": "free", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanSecuritiesTraderInfo", "tier": "free", "needs_data_id": False, "type": "brokers"},
            {"name": "TaiwanstockGovernmentBankBuySell", "tier": "sponsor", "needs_data_id": False, "type": "market"},
            {"name": "TaiwanTotalExchangeMarginMaintenance", "tier": "backer", "needs_data_id": False, "type": "market"},
            {"name": "TaiwanStockActiveETFInfo", "tier": "free", "needs_data_id": False, "type": "etf_list"},
            {"name": "TaiwanStockActiveETFHolding", "tier": "sponsor", "needs_data_id": True, "type": "etf", "data_id_source": "TaiwanStockActiveETFInfo"},
            {"name": "TaiwanStockActiveETFHoldingChange", "tier": "sponsor", "needs_data_id": True, "type": "etf", "data_id_source": "TaiwanStockActiveETFInfo"},
            {"name": "TaiwanStockIndustryChainMoneyFlow", "tier": "sponsor", "needs_data_id": False, "type": "daily"},
            {"name": "TaiwanStockMarginMaintenance", "tier": "sponsor", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockSuspended", "tier": "backer", "needs_data_id": False, "type": "daily"},
            {"name": "TaiwanStockDayTradingSuspension", "tier": "backer", "needs_data_id": False, "type": "daily"},
            {"name": "TaiwanStockTradingDailyReport", "tier": "sponsor", "needs_data_id": True, "type": "daily"},
            {"name": "TaiwanStockBlockTradingDailyReport", "tier": "sponsor", "needs_data_id": False, "type": "daily"},
            {"name": "TaiwanStockBlockTrade", "tier": "sponsor", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockLoanCollateralBalance", "tier": "sponsor", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockDispositionSecuritiesPeriod", "tier": "backer", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
        ],
    },
    "taiwan_fundamental": {
        "description": "Taiwan Market - Fundamental",
        "datasets": [
            {"name": "TaiwanStockFinancialStatements", "tier": "free", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockBalanceSheet", "tier": "free", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockCashFlowsStatement", "tier": "free", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockDividend", "tier": "free", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockDividendResult", "tier": "free", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockMonthRevenue", "tier": "free", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockCapitalReductionReferencePrice", "tier": "free", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockMarketValue", "tier": "backer", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockMarketValueWeight", "tier": "backer", "needs_data_id": True, "type": "stock", "data_id_source": "TaiwanStockInfo"},
        ],
    },
    "taiwan_derivatives": {
        "description": "Taiwan Market - Derivatives",
        "datasets": [
            {"name": "TaiwanFuturesDaily", "tier": "free", "needs_data_id": True, "type": "futures", "data_ids": ["TX", "MTX", "TXO", "TAIEX", "MiniTAIEX", "FTSETaiwan50"]},
            {"name": "TaiwanOptionDaily", "tier": "sponsor", "needs_data_id": True, "type": "options", "data_id_source": "TaiwanFutOptTickInfo"},
            {"name": "TaiwanFuturesInstitutionalInvestors", "tier": "free", "needs_data_id": True, "type": "futures_inst", "data_ids": ["MM", "RF", "ID", "ED", "FTSE", "TAIEX", "TX", "MTX", "TXO"]},
            {"name": "TaiwanOptionInstitutionalInvestors", "tier": "free", "needs_data_id": True, "type": "options_inst", "data_ids": ["IO", "TAIEX"]},
            {"name": "TaiwanFuturesDealerTradingVolumeDaily", "tier": "free", "needs_data_id": True, "type": "futures_dealer", "data_ids": ["MM", "RF"]},
            {"name": "TaiwanOptionDealerTradingVolumeDaily", "tier": "free", "needs_data_id": True, "type": "options_dealer", "data_ids": ["IO"]},
            {"name": "TaiwanFuturesSpreadTrading", "tier": "free", "needs_data_id": True, "type": "futures_spread", "data_ids": ["MM-TX", "RF-TX"]},
            {"name": "TaiwanOptionVIX", "tier": "free", "needs_data_id": True, "type": "options_vix", "data_ids": ["TXO"]},
            {"name": "TaiwanFuturesTick", "tier": "sponsor", "needs_data_id": True, "type": "futures_tick", "data_id_source": "TaiwanFuturesDaily"},
            {"name": "TaiwanOptionTick", "tier": "sponsor", "needs_data_id": True, "type": "options_tick", "data_id_source": "TaiwanFutOptTickInfo"},
            {"name": "TaiwanFuturesSpreadTick", "tier": "sponsor", "needs_data_id": True, "type": "futures_spread_tick", "data_ids": ["MM-TX", "RF-TX"]},
        ],
    },
    "taiwan_convertible_bond": {
        "description": "Taiwan Market - Convertible Bond",
        "datasets": [
            {"name": "TaiwanStockConvertibleBondInfo", "tier": "free", "needs_data_id": True, "type": "conv_bond", "data_id_source": "TaiwanStockInfo"},
            {"name": "TaiwanStockConvertibleBondDaily", "tier": "free", "needs_data_id": True, "type": "conv_bond", "data_id_source": "TaiwanStockConvertibleBondInfo"},
            {"name": "TaiwanStockConvertibleBondInstitutionalInvestors", "tier": "backer", "needs_data_id": True, "type": "conv_bond_inst", "data_id_source": "TaiwanStockConvertibleBondInfo"},
            {"name": "TaiwanStockConvertibleBondDailyOverview", "tier": "backer", "needs_data_id": False, "type": "conv_bond_overview"},
            {"name": "TaiwanStockConvertibleBondMonthlyAnalysis", "tier": "backer", "needs_data_id": True, "type": "conv_bond_monthly", "data_id_source": "TaiwanStockConvertibleBondInfo"},
            {"name": "TaiwanStockConvertibleBondPutProvision", "tier": "backer", "needs_data_id": True, "type": "conv_bond_put", "data_id_source": "TaiwanStockConvertibleBondInfo"},
        ],
    },
    "taiwan_news_indicator": {
        "description": "Taiwan Market - News & Indicators",
        "datasets": [
            {"name": "TaiwanStockNews", "tier": "free", "needs_data_id": False, "type": "news"},
            {"name": "TaiwanBusinessIndicator", "tier": "backer", "needs_data_id": False, "type": "indicator"},
            {"name": "TaiwanStockIndustryChain", "tier": "backer", "needs_data_id": False, "type": "industry"},
            {"name": "TaiwanStockDelisting", "tier": "free", "needs_data_id": False, "type": "delisting"},
        ],
    },
    "international_stocks": {
        "description": "International Markets",
        "datasets": [
            {"name": "USStockInfo", "tier": "free", "needs_data_id": False, "type": "stock_list"},
            {"name": "USStockPrice", "tier": "free", "needs_data_id": True, "type": "stock", "data_id_source": "USStockInfo"},
            {"name": "UKStockInfo", "tier": "free", "needs_data_id": False, "type": "stock_list"},
            {"name": "UKStockPrice", "tier": "backer", "needs_data_id": True, "type": "stock", "data_id_source": "UKStockInfo"},
            {"name": "EuropeStockInfo", "tier": "free", "needs_data_id": False, "type": "stock_list"},
            {"name": "EuropeStockPrice", "tier": "backer", "needs_data_id": True, "type": "stock", "data_id_source": "EuropeStockInfo"},
            {"name": "JapanStockInfo", "tier": "free", "needs_data_id": False, "type": "stock_list"},
            {"name": "JapanStockPrice", "tier": "backer", "needs_data_id": True, "type": "stock", "data_id_source": "JapanStockInfo"},
        ],
    },
    "global_economic": {
        "description": "Global Economic Data",
        "datasets": [
            {"name": "GoldPrice", "tier": "free", "needs_data_id": False, "type": "commodity"},
            {"name": "CrudeOilPrices", "tier": "free", "needs_data_id": True, "type": "commodity", "data_ids": ["Brent", "WTI"]},
            {"name": "TaiwanExchangeRate", "tier": "free", "needs_data_id": True, "type": "fx", "data_ids": ["USD", "EUR", "JPY", "GBP", "CNY", "HKD", "KRW", "SGD", "AUD", "CAD", "CHF", "NZD", "THB", "IDR", "MYR", "PHP", "VND", "ZAR", "SEK"]},
            {"name": "ExchangeRate", "tier": "free", "needs_data_id": True, "type": "fx", "data_ids": ["Canda", "China", "Euro", "Japan", "Taiwan", "UK"]},
            {"name": "InterestRate", "tier": "free", "needs_data_id": True, "type": "interest", "data_ids": ["FED", "ECB", "BOJ", "BOE", "PBOC", "RBA", "BOC", "RBNZ", "RBI", "CBR", "SNB", "BCB"]},
            {"name": "GovernmentBondsYield", "tier": "free", "needs_data_id": True, "type": "bonds", "data_ids": ["United States 1-Month", "United States 1-Year", "United States 2-Month", "United States 2-Year", "United States 3-Month", "United States 3-Year", "United States 5-Year", "United States 6-Month", "United States 7-Year", "United States 10-Year", "United States 20-Year", "United States 30-Year", "United States 4-Month"]},
            {"name": "CnnFearGreedIndex", "tier": "backer", "needs_data_id": False, "type": "sentiment"},
            {"name": "CurrencyCirculation", "tier": "backer", "needs_data_id": True, "type": "macro", "data_ids": ["US", "Europe", "Taiwan"]},
        ],
    },
}

# Flatten all datasets for easy lookup
ALL_DATASETS = {}
for category, info in DATASETS.items():
    for ds in info["datasets"]:
        ds["category"] = category
        ALL_DATASETS[ds["name"]] = ds


class FinMindFetcher:
    def __init__(self, token=None, output_dir="finmind_data", max_stocks=None, rate_limit=5):
        self.token = token
        self.output_dir = Path(output_dir)
        self.max_stocks = max_stocks
        self.rate_limit_delay = 12.0 if not token else 6.0  # 300/hour free, 600/hour with token
        self.session = requests.Session()
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.progress_file = self.output_dir / "progress.json"
        self.progress = self._load_progress()
        self.api_count = 0

    def _load_progress(self):
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                return json.load(f)
        return {"completed": [], "stock_ids": {}}

    def _save_progress(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def _rate_limit_delay(self):
        """Sleep to respect rate limits. 300/hour free = 12s delay, 600/hour token = 6s delay."""
        self.api_count += 1
        delay = self.rate_limit_delay
        if self.api_count % 50 == 0:
            print(f"  [API call #{self.api_count}] Sleeping {delay}s between requests (respecting rate limit)...")
        time.sleep(delay)

    def fetch_data(self, dataset, data_id=None, start_date=None, end_date=None):
        """Fetch data from FINMIND API with rate limiting."""
        params = {"dataset": dataset}
        if data_id:
            params["data_id"] = data_id
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        self._rate_limit_delay()
        resp = self.session.get(f"{BASE_URL}/data", params=params, timeout=60)
        result = resp.json()

        if result.get("status") != 200:
            if "Your level is free" in result.get("msg", ""):
                return {"status": "TIER_REQUIRED", "msg": result.get("msg", "")}
            return {"status": "ERROR", "msg": result.get("msg", str(resp.text))}

        return {"status": "OK", "data": result.get("data", [])}

    def fetch_datalist(self, dataset):
        """Get list of data_ids for a dataset."""
        self._rate_limit_delay()
        resp = self.session.get(f"{BASE_URL}/datalist", params={"dataset": dataset}, timeout=30)
        return resp.json()

    def get_stock_ids(self):
        """Get all Taiwan stock IDs."""
        cache_file = self.output_dir / "stock_ids.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        result = self.fetch_data("TaiwanStockInfo")
        if result["status"] != "OK":
            return []

        stocks = result["data"]
        stock_ids = sorted(set(s["stock_id"] for s in stocks))

        # Also save full stock info
        df = pd.DataFrame(stocks)
        stock_info_dir = self.output_dir / "taiwan_technical"
        stock_info_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(stock_info_dir / "TaiwanStockInfo.csv", index=False)

        with open(cache_file, "w") as f:
            json.dump(stock_ids, f)

        if self.max_stocks:
            stock_ids = stock_ids[:self.max_stocks]

        return stock_ids

    def get_international_stock_ids(self, dataset):
        """Get stock IDs for international markets."""
        cache_file = self.output_dir / f"{dataset}_ids.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        result = self.fetch_data(dataset)
        if result["status"] != "OK":
            return []

        stocks = result["data"]
        stock_ids = sorted(set(s["stock_id"] for s in stocks))

        with open(cache_file, "w") as f:
            json.dump(stock_ids, f)

        return stock_ids

    def save_data(self, category, dataset_name, data_id, data):
        """Save data to CSV file."""
        dir_path = self.output_dir / category
        dir_path.mkdir(parents=True, exist_ok=True)

        if not data:
            return

        df = pd.DataFrame(data)
        filename = f"{dataset_name}"
        if data_id:
            filename += f"_{data_id}"
        filename += ".csv"

        df.to_csv(dir_path / filename, index=False)
        print(f"    Saved: {dir_path / filename} ({len(data)} rows)")

    def fetch_no_data_id_dataset(self, ds_config):
        """Fetch data for a dataset that doesn't require data_id."""
        name = ds_config["name"]
        category = ds_config["category"]
        start_date = "2016-01-01"

        # Determine date range - use today as end date
        end_date = date.today().isoformat()

        if name in self.progress["completed"]:
            print(f"  [SKIP] {name} - already fetched")
            return

        print(f"  Fetching {name}...")
        result = self.fetch_data(name, start_date=start_date, end_date=end_date)

        if result["status"] == "TIER_REQUIRED":
            print(f"    [SKIP] {name} - requires Backer/Sponsor tier")
            return
        elif result["status"] != "OK":
            print(f"    [ERROR] {name}: {result.get('msg', 'unknown error')}")
            return

        self.save_data(category, name, None, result["data"])
        self.progress["completed"].append(name)
        self._save_progress()

    def fetch_with_data_ids(self, ds_config, data_ids, start_date="2016-01-01", end_date=None):
        """Fetch data for a dataset that requires data_id."""
        name = ds_config["name"]
        category = ds_config["category"]
        end_date = end_date or date.today().isoformat()

        if name in self.progress["completed"]:
            print(f"  [SKIP] {name} - already fetched for all IDs")
            return

        completed_key = f"{category}:{name}"
        completed_ids = set(self.progress.get("stock_ids", {}).get(completed_key, []))

        print(f"  Fetching {name} for {len(data_ids)} data_ids...")
        success_count = 0
        skip_count = 0
        error_count = 0

        for i, did in enumerate(data_ids):
            # Skip already completed
            if did in completed_ids:
                skip_count += 1
                continue

            # Check for tier requirement - if first attempt failed with tier error, skip all
            if self.progress.get("tier_limited", {}).get(name):
                skip_count += 1
                continue

            result = self.fetch_data(name, data_id=did, start_date=start_date, end_date=end_date)

            if result["status"] == "TIER_REQUIRED":
                print(f"    [TIER] {name} requires Backer/Sponsor - skipping all remaining")
                self.progress.setdefault("tier_limited", {})[name] = True
                skip_count += len(data_ids) - i
                break
            elif result["status"] != "OK":
                print(f"    [ERROR] {name} ({did}): {result.get('msg', 'unknown')}")
                error_count += 1
                continue

            self.save_data(category, name, did, result["data"])
            completed_ids.add(did)
            success_count += 1

            if (i + 1) % 10 == 0:
                self.progress.setdefault("stock_ids", {})[completed_key] = sorted(completed_ids)
                self._save_progress()

            if (i + 1) % 50 == 0:
                print(f"    Progress: {i+1}/{len(data_ids)} ({success_count} OK, {skip_count} skipped, {error_count} errors)")

        self.progress.setdefault("stock_ids", {})[completed_key] = sorted(completed_ids)
        if len(completed_ids) == len(data_ids):
            self.progress["completed"].append(name)
        self._save_progress()
        print(f"  Done: {success_count} OK, {skip_count} skipped, {error_count} errors")

    def run(self, categories=None, date_range=None):
        """Run the full data fetcher."""
        start_date, end_date = date_range or ("2016-01-01", date.today().isoformat())
        categories = categories or list(DATASETS.keys())

        print(f"\n{'='*60}")
        print(f"FINMIND Data Fetcher")
        print(f"Token: {'SET' if self.token else 'NOT SET (Free tier)'}")
        print(f"Rate limit: every {self.rate_limit_delay:.1f}s")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Output: {self.output_dir}")
        print(f"Max stocks per dataset: {self.max_stocks or 'ALL'}")
        print(f"{'='*60}\n")

        # Get stock IDs first
        stock_ids = self.get_stock_ids()
        print(f"\nTotal Taiwan stock IDs available: {len(stock_ids)}")
        if self.max_stocks:
            stock_ids = stock_ids[:self.max_stocks]
        print(f"Will fetch for {len(stock_ids)} stocks")

        # Get international stock IDs
        intl_stocks = {}
        for ds_info in DATASETS["international_stocks"]["datasets"]:
            if ds_info["needs_data_id"] and ds_info.get("data_id_source", "").startswith("USStock"):
                intl_stocks["USStockInfo"] = self.get_international_stock_ids("USStockInfo")

        print(f"US stock IDs available: {len(intl_stocks.get('USStockInfo', []))}")

        # Process datasets
        for category in categories:
            if category not in DATASETS:
                continue

            cat_info = DATASETS[category]
            print(f"\n{'='*60}")
            print(f"Category: {cat_info['description']}")
            print(f"{'='*60}")

            for ds_config in cat_info["datasets"]:
                name = ds_config["name"]
                tier = ds_config["tier"]

                # Check tier access
                if tier != "free" and not self.token:
                    print(f"\n  [SKIP] {name} - requires {tier} tier (Free account)")
                    continue
                if tier == "backer" and self.token:
                    print(f"\n  [NOTE] {name} - may require Backer/Sponsor tier")

                if not ds_config["needs_data_id"]:
                    self.fetch_no_data_id_dataset(ds_config)
                else:
                    # Determine data_ids
                    if "data_ids" in ds_config:
                        data_ids = ds_config["data_ids"]
                    elif ds_config.get("type") == "stock":
                        data_ids = stock_ids
                    elif ds_config.get("data_id_source") == "USStockInfo":
                        data_ids = intl_stocks.get("USStockInfo", [])
                    elif ds_config.get("data_id_source") == "TaiwanStockInfo":
                        data_ids = stock_ids
                    elif ds_config.get("data_id_source") == "TaiwanStockActiveETFInfo":
                        # Need to fetch active ETF list first
                        etf_result = self.fetch_data("TaiwanStockActiveETFInfo")
                        if etf_result["status"] == "OK":
                            data_ids = sorted(set(d.get("stock_id", "") for d in etf_result["data"] if d.get("stock_id")))
                        else:
                            data_ids = []
                    else:
                        data_ids = ds_config.get("data_ids", [])

                    if data_ids:
                        self.fetch_with_data_ids(ds_config, data_ids, start_date, end_date)

        print(f"\n{'='*60}")
        print(f"Fetch complete! Total API calls: {self.api_count}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}")

        # Generate summary
        self._generate_summary()

    def _generate_summary(self):
        """Generate a summary of downloaded data."""
        summary_file = self.output_dir / "summary.json"
        summary = {
            "fetch_date": date.today().isoformat(),
            "total_api_calls": self.api_count,
            "token_used": bool(self.token),
            "date_range": "2016-01-01 to present",
            "categories": {}
        }

        for category in DATASETS:
            cat_dir = self.output_dir / category
            if cat_dir.exists():
                files = list(cat_dir.rglob("*.csv"))
                total_rows = 0
                file_info = []
                for f in files:
                    size = f.stat().st_size
                    file_info.append({"file": str(f.relative_to(self.output_dir)), "size_bytes": size})
                    total_rows += sum(1 for _ in open(f, encoding="utf-8", errors="replace")) - 1  # count lines minus header
                summary["categories"][category] = {
                    "num_files": len(files),
                    "total_rows": total_rows,
                    "total_size_bytes": sum(f["size_bytes"] for f in file_info),
                    "files": file_info
                }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Fetch FINMIND data for past 10 years across all categories")
    parser.add_argument("--token", default=None, help="FINMIND API token (optional, increases rate limit to 600/hr)")
    parser.add_argument("--output", default="finmind_data", help="Output directory")
    parser.add_argument("--max-stocks", type=int, default=None, help="Maximum number of stocks to fetch per dataset (default: all)")
    parser.add_argument("--rate-limit", type=int, default=None, help="Max requests per hour (default: 300 free / 600 with token)")
    parser.add_argument("--categories", nargs="+", choices=list(DATASETS.keys()), default=list(DATASETS.keys()), help="Categories to fetch")
    parser.add_argument("--start-date", default="2016-01-01", help="Start date (default: 2016-01-01)")
    parser.add_argument("--end-date", default=None, help="End date (default: today)")

    args = parser.parse_args()
    end_date = args.end_date or date.today().isoformat()

    fetcher = FinMindFetcher(
        token=args.token,
        output_dir=args.output,
        max_stocks=args.max_stocks,
        rate_limit=args.rate_limit or (600 if args.token else 300)
    )

    fetcher.run(
        categories=args.categories,
        date_range=(args.start_date, end_date)
    )


if __name__ == "__main__":
    main()
