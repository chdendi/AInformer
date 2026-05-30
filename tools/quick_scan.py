"""
快速市值扫描器

快速扫描港股中市值的标的，用于初步筛选。

用法:
  python -m tools.quick_scan                        # 默认 20~70亿区间
  python -m tools.quick_scan --min 10 --max 100     # 自定义区间

作为库使用:
  from tools.quick_scan import QuickMarketCapScanner
  scanner = QuickMarketCapScanner(min_mc=20, max_mc=70)
  results = scanner.scan()
"""

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")


@dataclass
class QuickScanResult:
    code: str
    name: str
    market_cap: float      # 亿HKD
    price: float
    avg_volume: int


class QuickMarketCapScanner:
    """快速港股市值扫描 (数据源: Yahoo Finance)"""

    def __init__(self, min_mc: float = 20, max_mc: float = 70,
                 data_dir: Optional[Path] = None, batch_size: int = 100):
        self.min_mc = min_mc
        self.max_mc = max_mc
        self.data_dir = data_dir or Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

    def scan(self) -> list[QuickScanResult]:
        import yfinance as yf

        tickers = self._generate_tickers()
        print(f"快速市值扫描: {self.min_mc}~{self.max_mc}亿 HKD, 共 {len(tickers)} 只待扫...")

        found = []
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i + self.batch_size]
            bn = i // self.batch_size + 1
            total = (len(tickers) - 1) // self.batch_size + 1

            try:
                ts = yf.Tickers(" ".join(batch))
                batch_count = 0
                for tkr in batch:
                    try:
                        t = ts.tickers.get(tkr)
                        if t is None:
                            continue
                        info = t.info
                        mc = info.get("marketCap")
                        if mc and self.min_mc * 1e8 <= mc <= self.max_mc * 1e8:
                            name = info.get("shortName") or info.get("longName") or tkr
                            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
                            vol = info.get("averageVolume") or 0
                            found.append(
                                QuickScanResult(
                                    code=tkr.replace(".HK", ""),
                                    name=str(name)[:50],
                                    market_cap=round(mc / 1e8, 1),
                                    price=price,
                                    avg_volume=vol,
                                )
                            )
                            batch_count += 1
                    except Exception:
                        continue
                if bn % 10 == 1 or bn == total:
                    print(f"  Batch {bn}/{total}: {batch_count} found, total={len(found)}")
            except Exception as e:
                if bn % 10 == 1:
                    print(f"  Batch {bn}/{total}: error - {str(e)[:50]}")
            time.sleep(0.1)

        print(f"\n共找到 {len(found)} 只标的在 {self.min_mc}~{self.max_mc}亿 HKD 区间")
        return found

    @staticmethod
    def _generate_tickers() -> list[str]:
        tickers = []
        for i in range(1, 2501):
            tickers.append(f"{i:04d}.HK")
        for i in range(6000, 7001):
            tickers.append(f"{i:04d}.HK")
        for i in range(8001, 8501):
            tickers.append(f"{i:04d}.HK")
        for i in range(9600, 9999):
            tickers.append(f"{i:04d}.HK")
        return sorted(set(tickers))

    def to_json(self, results: list[QuickScanResult]) -> str:
        data = []
        for r in results:
            data.append({
                "代码": r.code,
                "名称": r.name,
                "当前市值(亿)": r.market_cap,
                "股价": r.price,
                "日均成交量": r.avg_volume,
            })
        out = self.data_dir / "quick_scan.json"
        with open(out, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(out)

    @staticmethod
    def print_results(results: list[QuickScanResult]):
        if not results:
            print("未找到符合条件的标的")
            return
        results.sort(key=lambda r: r.market_cap, reverse=True)
        print("\n按当前市值排序:")
        for r in results:
            print(f"  {r.code:>8s}  {r.name[:30]:<30s}  {r.market_cap:>6.1f}亿")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="快速港股市值扫描")
    parser.add_argument("--min", type=float, default=20, help="最低市值 (亿HKD)")
    parser.add_argument("--max", type=float, default=70, help="最高市值 (亿HKD)")
    args = parser.parse_args()

    scanner = QuickMarketCapScanner(min_mc=args.min, max_mc=args.max)
    results = scanner.scan()
    scanner.print_results(results)

    path = scanner.to_json(results)
    print(f"\n已保存到 {path}")


if __name__ == "__main__":
    main()
