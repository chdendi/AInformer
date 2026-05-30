"""
基本面分析工具

获取指定港股的基本面数据: 估值、盈利能力、成长性、财务健康、现金流、股息。

用法:
  python -m tools.fundamental 0700.HK 9988.HK     # 分析指定股票
  python -m tools.fundamental --batch stocks.txt   # 批量分析

作为库使用:
  from tools.fundamental import FundamentalAnalyzer
  analyzer = FundamentalAnalyzer()
  report = analyzer.analyze("0700.HK")
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")


@dataclass
class FundamentalReport:
    ticker: str
    name: str
    sector: str = ""
    industry: str = ""
    market_cap: float = 0
    current_price: float = 0
    low_52w: float = 0
    high_52w: float = 0
    beta: Optional[float] = None
    pe: Optional[float] = None
    pb: Optional[float] = None
    ps: Optional[float] = None
    ev_ebitda: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    profit_margin: Optional[float] = None
    gross_margin: Optional[float] = None
    rev_growth: Optional[float] = None
    earn_growth: Optional[float] = None
    debt_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    revenue: float = 0
    net_income: float = 0
    fcf: Optional[float] = None
    op_cf: Optional[float] = None
    div_yield: Optional[float] = None
    div_rate: Optional[float] = None
    payout_ratio: Optional[float] = None
    employees: int = 0
    description: str = ""
    error: str = ""


class FundamentalAnalyzer:
    """港股基本面分析器 (数据源: Yahoo Finance)"""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, ticker: str) -> FundamentalReport:
        import yfinance as yf

        try:
            t = yf.Ticker(ticker)
            info = t.info
        except Exception as e:
            return FundamentalReport(ticker=ticker, name=ticker, error=str(e)[:100])

        return FundamentalReport(
            ticker=ticker.replace(".HK", ""),
            name=info.get("shortName") or info.get("longName") or ticker,
            sector=info.get("sector", ""),
            industry=info.get("industry", ""),
            market_cell=(info.get("marketCap") or 0) / 1e8,
            current_price=info.get("currentPrice") or info.get("regularMarketPrice") or 0,
            low_52w=info.get("fiftyTwoWeekLow") or 0,
            high_52w=info.get("fiftyTwoWeekHigh") or 0,
            beta=info.get("beta"),
            pe=info.get("trailingPE") or info.get("forwardPE"),
            pb=info.get("priceToBook"),
            ps=info.get("priceToSalesTrailing12Months"),
            ev_ebitda=info.get("enterpriseToEbitda"),
            roe=info.get("returnOnEquity"),
            roa=info.get("returnOnAssets"),
            profit_margin=info.get("profitMargins"),
            gross_margin=info.get("grossMargins"),
            rev_growth=info.get("revenueGrowth"),
            earn_growth=info.get("earningsGrowth"),
            debt_equity=info.get("debtToEquity"),
            current_ratio=info.get("currentRatio"),
            quick_ratio=info.get("quickRatio"),
            revenue=(info.get("totalRevenue") or 0) / 1e8,
            net_income=(info.get("netIncomeToCommon") or info.get("netIncome") or 0) / 1e8,
            fcf=info.get("freeCashflow"),
            op_cf=info.get("operatingCashflow"),
            div_yield=info.get("dividendYield"),
            div_rate=info.get("dividendRate"),
            payout_ratio=info.get("payoutRatio"),
            employees=info.get("fullTimeEmployees") or 0,
            description=(info.get("longBusinessSummary") or "")[:500],
        )

    def analyze_batch(self, tickers: list[str]) -> list[FundamentalReport]:
        return [self.analyze(t) for t in tickers]

    @staticmethod
    def print_report(report: FundamentalReport):
        if report.error:
            print(f"\n{'='*70}\n  {report.name} ({report.ticker}) - 数据获取失败: {report.error}\n{'='*70}")
            return

        print(f"\n{'='*70}")
        print(f"  {report.name} ({report.ticker}.HK)")
        print(f"{'='*70}")

        if report.description:
            desc = report.description[:300]
            desc += "..." if len(report.description) > 300 else ""
            print(f"\n  📋 业务: {desc}")

        print(f"\n  🏭 行业: {report.sector} / {report.industry}")
        print(f"  💰 当前市值: {report.market_cell:.1f}亿 HKD")
        if report.employees:
            print(f"  👥 员工: {report.employees:,}人")

        cur = report.current_price
        lo, hi = report.low_52w, report.high_52w
        if cur:
            print(f"  📈 最新价: {cur:.2f} HKD")
        if lo and hi and hi != lo:
            pct = (cur - lo) / (hi - lo) * 100
            print(f"  📊 52周: {lo:.2f} ~ {hi:.2f} (当前处于{pct:.0f}%分位)")
        if report.beta is not None:
            b = report.beta
            label = "高波动" if abs(b) > 1.5 else ("中等" if abs(b) > 0.8 else "低波动")
            print(f"  📉 Beta: {b:.2f} ({label})")

        if report.revenue:
            print(f"\n  📊 最近财年营收: {report.revenue:.1f}亿 HKD")
            print(f"  📊 最近财年净利润: {report.net_income:.1f}亿 HKD")

        if any([report.pe, report.pb, report.ps, report.ev_ebitda]):
            print(f"\n  💎 估值指标:")
            _p("PE(TTM)", report.pe, "x")
            _p("PB", report.pb, "x")
            _p("PS", report.ps, "x")
            _p("EV/EBITDA", report.ev_ebitda, "x")

        if any([report.roe, report.roa, report.profit_margin, report.gross_margin]):
            print(f"\n  📈 盈利能力:")
            _ppct("ROE", report.roe)
            _ppct("ROA", report.roa)
            _ppct("净利率", report.profit_margin)
            _ppct("毛利率", report.gross_margin)

        if any([report.rev_growth, report.earn_growth]):
            print(f"\n  🚀 成长性:")
            _ppct("营收增速(YoY)", report.rev_growth)
            _ppct("盈利增速(YoY)", report.earn_growth)

        if any([report.debt_equity, report.current_ratio, report.quick_ratio]):
            print(f"\n  🏦 财务健康:")
            if report.debt_equity is not None:
                print(f"     负债权益比: {report.debt_equity}")
            if report.current_ratio:
                print(f"     流动比率: {report.current_ratio:.2f}")
            if report.quick_ratio:
                print(f"     速动比率: {report.quick_ratio:.2f}")

        if report.fcf or report.op_cf:
            print(f"\n  💵 现金流:")
            if report.op_cf:
                print(f"     经营现金流: {report.op_cf/1e8:.1f}亿 HKD")
            if report.fcf:
                print(f"     自由现金流: {report.fcf/1e8:.1f}亿 HKD")

        if report.div_yield or report.div_rate:
            print(f"\n  💸 股息:")
            if report.div_yield and report.div_yield > 0:
                print(f"     股息率: {report.div_yield*100:.2f}%")
            if report.div_rate and report.div_rate > 0:
                print(f"     每股股息: {report.div_rate:.4f} HKD")


def _p(label: str, value, unit: str = ""):
    if value is not None:
        print(f"     {label}: {value:.1f}{unit}" if isinstance(value, (int, float)) else f"     {label}: {value}")


def _ppct(label: str, value):
    if value is not None:
        print(f"     {label}: {value*100:+.1f}%" if isinstance(value, (int, float)) else f"     {label}: {value}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="港股基本面分析")
    parser.add_argument("tickers", nargs="*", help="港股 ticker (如 0700.HK)")
    parser.add_argument("--batch", type=str, help="批量文件 (每行一个 ticker)")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    tickers = args.tickers
    if args.batch:
        with open(args.batch) as f:
            tickers = [line.strip() for line in f if line.strip()]

    if not tickers:
        print("请提供 ticker 列表")
        return

    analyzer = FundamentalAnalyzer()
    reports = analyzer.analyze_batch(tickers)

    if args.json:
        print(json.dumps(
            [{k: v for k, v in r.__dict__.items() if not k.startswith("_")} for r in reports],
            ensure_ascii=False, indent=2,
        ))
    else:
        for r in reports:
            FundamentalAnalyzer.print_report(r)

        print(f"\n{'='*70}")
        print("  ⚠ 数据说明")
        print("=" * 70)
        print("  • 数据源: Yahoo Finance")
        print("  • PE/PB 等基于最近可得的财务数据，可能延迟")
        print("  • 以上不构成投资建议")


if __name__ == "__main__":
    main()
