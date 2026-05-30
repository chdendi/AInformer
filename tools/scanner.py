"""
港股通潜在纳入扫描器

扫描全港股，筛选日均市值接近港股通门槛 (50亿 HKD) 的潜在纳入标的。

港股通小型股纳入条件:
  1. 恒生综合小型股指数成分股
  2. 过去12个月日均市值 ≥ 50亿港元
  3. 日均换手率 ≥ 0.05%

用法:
  python -m tools.scanner                    # 扫描，输出结果
  python -m tools.scanner --min-mc 30        # 自定义最低市值阈值
  python -m tools.scanner --output results   # 自定义输出文件名

作为库使用:
  from tools.scanner import StockConnectScanner
  scanner = StockConnectScanner(min_mc=20, max_mc=70)
  results = scanner.scan()
  df = scanner.to_dataframe(results)
"""

import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class StockResult:
    code: str
    name: str
    current_mc: float           # 当前市值 (亿HKD)
    avg_mc_12m: float            # 12月日均市值 (亿HKD)
    avg_turnover: float          # 日均换手率 (%)
    gap_to_threshold: float      # 距目标门槛差值 (亿)
    gap_pct: float               # 距目标门槛百分比
    turnover_达标: str           # 换手率是否达标


@dataclass
class ScannerConfig:
    """扫描器配置"""
    min_mc: float = 20            # 初筛最低市值 (亿HKD)
    max_mc: float = 70            # 初筛最高市值 (亿HKD)
    connect_threshold: float = 50 # 港股通门槛
    turnover_threshold: float = 0.05  # 换手率门槛 (%)
    lookback_days: int = 400      # 回看天数
    min_history_days: int = 100   # 最少历史数据天数
    data_dir: Path = field(default_factory=lambda: Path("data"))
    request_delay: float = 0.08   # API 请求间隔


class StockConnectScanner:
    """港股通潜在纳入扫描器

    使用 akshare 获取全港股实时行情进行初筛，
    再用 akshare 获取历史数据精确计算 12 月日均市值和换手率。
    若 akshare 不可用则回退到基于代码范围 + yfinance 的方案。
    """

    def __init__(self, config: Optional[ScannerConfig] = None, **kwargs):
        self.config = config or ScannerConfig(**kwargs)
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

    def scan(self) -> list[StockResult]:
        candidates = self._initial_screen()
        if not candidates:
            print("初筛无结果，尝试扩大范围...")
            self.config.min_mc = max(10, self.config.min_mc - 10)
            self.config.max_mc = min(100, self.config.max_mc + 10)
            candidates = self._initial_screen()

        results = self._precise_calculation(candidates)
        return results

    def _initial_screen(self) -> list[dict]:
        """第一步：快速获取全港股行情，按市值初筛"""
        candidates = []
        try:
            candidates = self._screen_via_akshare()
        except Exception:
            candidates = self._screen_via_fallback()
        return candidates

    def _screen_via_akshare(self) -> list[dict]:
        import akshare as ak

        print("第一步：通过 akshare 获取全港股实时行情...")
        df_all = None
        for attempt in range(5):
            try:
                df_all = ak.stock_hk_spot_em()
                if df_all is not None and not df_all.empty:
                    break
            except Exception as e:
                print(f"  尝试 {attempt+1}/5 失败: {e}")
                time.sleep(3)

        if df_all is None or df_all.empty:
            raise RuntimeError("akshare 获取行情失败")

        print(f"获取到 {len(df_all)} 只港股行情")

        mc_col = None
        for col in df_all.columns:
            if "总市值" in str(col) or "市值" in str(col):
                mc_col = col
                break

        if not mc_col:
            raise RuntimeError("未找到市值字段")

        df_all["市值亿"] = pd.to_numeric(df_all[mc_col], errors="coerce")
        if df_all["市值亿"].isna().all():
            df_all["市值亿"] = df_all[mc_col].apply(self._parse_market_cap)

        df_valid = df_all.dropna(subset=["市值亿"])
        screened = df_valid[
            (df_valid["市值亿"] >= self.config.min_mc)
            & (df_valid["市值亿"] <= self.config.max_mc)
        ]
        print(f"市值 {self.config.min_mc}~{self.config.max_mc}亿 区间: {len(screened)} 只")

        return [
            {
                "code": str(row["代码"]),
                "name": str(row.get("名称", "")),
                "current_mc": float(row["市值亿"]),
            }
            for _, row in screened.iterrows()
        ]

    def _screen_via_fallback(self) -> list[dict]:
        print("回退方案：扫描代码范围...")
        codes = []
        for i in range(1, 2501):
            codes.append(f"{i:04d}")
        for i in range(6000, 7001):
            codes.append(f"{i:04d}")
        for i in range(8001, 8501):
            codes.append(f"{i:04d}")
        for i in range(9000, 10000):
            codes.append(f"{i:04d}")

        print(f"共 {len(codes)} 只待扫描 (将跳过不存在/无数据的标的)")
        return [{"code": c, "name": "", "current_mc": 0} for c in codes]

    def _precise_calculation(self, candidates: list[dict]) -> list[StockResult]:
        import akshare as ak

        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=self.config.lookback_days)).strftime("%Y%m%d")

        print(f"\n第二步：精确计算 12 月日均市值和换手率 ({len(candidates)} 只)...")

        results = []
        success = 0
        skipped = 0

        for idx, c in enumerate(candidates):
            code = c["code"]
            name = c.get("name", "")
            current_mc = c.get("current_mc", 0)

            if (idx + 1) % 50 == 1 or idx == len(candidates) - 1:
                print(
                    f"  [{idx+1}/{len(candidates)}] {code} {name[:20]}...",
                    end=" ",
                    flush=True,
                )

            try:
                hist = ak.stock_hk_hist(
                    symbol=code, period="daily",
                    start_date=start, end_date=end, adjust="qfq",
                )
                if hist is None or hist.empty or len(hist) < self.config.min_history_days:
                    skipped += 1
                    if (idx + 1) % 50 == 1 or idx == len(candidates) - 1:
                        print("跳过 (数据不足)")
                    continue

                close = hist["收盘"].values
                volume = hist["成交量"].values

                latest_close = close[-1]
                if latest_close <= 0:
                    skipped += 1
                    continue

                if current_mc > 0:
                    total_shares = current_mc * 1e8 / latest_close
                else:
                    skipped += 1
                    continue

                daily_mc = close * total_shares / 1e8
                recent_252 = daily_mc[-252:] if len(daily_mc) >= 252 else daily_mc
                avg_mc = recent_252.mean()

                if avg_mc < self.config.min_mc or avg_mc > self.config.max_mc:
                    skipped += 1
                    continue

                daily_turnover = volume / total_shares * 100
                recent_to = daily_turnover[-252:] if len(daily_turnover) >= 252 else daily_turnover
                avg_turnover = recent_to.mean()

                if not name:
                    name = code

                gap = self.config.connect_threshold - avg_mc
                gap_pct = gap / self.config.connect_threshold * 100

                results.append(
                    StockResult(
                        code=code,
                        name=name,
                        current_mc=round(current_mc, 1),
                        avg_mc_12m=round(avg_mc, 1),
                        avg_turnover=round(avg_turnover, 4),
                        gap_to_threshold=round(gap, 1),
                        gap_pct=round(gap_pct, 1),
                        turnover_达标="✓" if avg_turnover >= self.config.turnover_threshold else "✗",
                    )
                )
                success += 1

                if (idx + 1) % 50 == 1 or idx == len(candidates) - 1:
                    print(f"✓ 日均市值{avg_mc:.1f}亿 换手率{avg_turnover:.4f}%")

            except Exception as e:
                skipped += 1
                if (idx + 1) % 50 == 1 or idx == len(candidates) - 1:
                    print(f"失败: {str(e)[:40]}")

            time.sleep(self.config.request_delay)

        print(f"\n扫描完成: 成功 {success}, 跳过 {skipped}")
        return results

    @staticmethod
    def to_dataframe(results: list[StockResult]) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in results])

    @staticmethod
    def _parse_market_cap(val) -> Optional[float]:
        try:
            if isinstance(val, str):
                val = val.replace(",", "").replace("亿", "")
                val = val.replace("万", "e-4").replace("元", "")
                v = float(val)
                return v if v < 1000 else v / 1e8
            v = float(val)
            return v / 1e8 if v > 1e10 else v
        except (ValueError, TypeError):
            return None

    def print_report(self, results: list[StockResult]):
        df = self.to_dataframe(results)
        if df.empty:
            print("\n未获取到有效数据")
            return

        df = df.sort_values("avg_mc_12m", ascending=False)

        threshold = self.config.connect_threshold
        target = df[
            (df["avg_mc_12m"] >= threshold * 0.7)
            & (df["avg_mc_12m"] < threshold)
        ]

        print("\n" + "=" * 115)
        print(f"📊 港股通潜在纳入标的 — 差 10~30% (日均市值 {threshold*0.7:.0f}~{threshold:.0f}亿 HKD)")
        print(f"  港股通门槛: 日均市值≥{threshold:.0f}亿 + 换手率≥{self.config.turnover_threshold}% + 恒生综合指数成分")
        print("=" * 115)

        if target.empty:
            target = df[
                (df["avg_mc_12m"] >= threshold * 0.6)
                & (df["avg_mc_12m"] < threshold * 1.1)
            ]

        for _, row in target.iterrows():
            print(
                f"  {row['code']:>8s} {str(row['name'])[:20]:<20s} "
                f"日均市值: {row['avg_mc_12m']:>6.1f}亿 "
                f"差{row['gap_to_threshold']:>5.1f}亿({row['gap_pct']:>4.0f}%) "
                f"换手率: {row['avg_turnover']:>7.4f}% {row['turnover_达标']}"
            )

        self._export(df)

    def _export(self, df: pd.DataFrame):
        base = self.config.data_dir / "hkstock_connect_scan"
        df.to_json(f"{base}.json", orient="records", force_ascii=False, indent=2)
        df.to_csv(f"{base}.csv", index=False, encoding="utf-8-sig")
        print(f"\n✅ 导出: {base}.json / {base}.csv ({len(df)} 只)")

        self._print_stats(df)

    def _print_stats(self, df: pd.DataFrame):
        threshold = self.config.connect_threshold
        bands = [
            (threshold, 999, "≥50亿 (已达标)"),
            (threshold * 0.9, threshold, "45~50亿 (差0~10%)"),
            (threshold * 0.8, threshold * 0.9, "40~45亿 (差10~20%)"),
            (threshold * 0.7, threshold * 0.8, "35~40亿 (差20~30%)"),
            (threshold * 0.6, threshold * 0.7, "30~35亿 (差30~40%)"),
        ]
        print("\n📈 分布统计:")
        for low, high, label in bands:
            cnt = len(df[(df["avg_mc_12m"] >= low) & (df["avg_mc_12m"] < high)])
            print(f"   {label}: {cnt} 只")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="港股通潜在纳入扫描器")
    parser.add_argument("--min-mc", type=float, default=20, help="初筛最低市值 (亿HKD)")
    parser.add_argument("--max-mc", type=float, default=70, help="初筛最高市值 (亿HKD)")
    parser.add_argument("--threshold", type=float, default=50, help="港股通门槛 (亿HKD)")
    parser.add_argument("--output", type=str, default=None, help="输出文件名前缀")
    args = parser.parse_args()

    config = ScannerConfig(
        min_mc=args.min_mc,
        max_mc=args.max_mc,
        connect_threshold=args.threshold,
    )

    scanner = StockConnectScanner(config)
    results = scanner.scan()
    scanner.print_report(results)


if __name__ == "__main__":
    main()
