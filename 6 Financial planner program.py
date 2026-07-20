"""
Personal Financial Planner — PRO EDITION
========================================
Projects your net worth over time based on:
  - Income sources (salary, rent, other) with recurrence
  - Expenses with recurrence
  - Any number of investment assets (ETFs / stocks), entered MANUALLY

HOW IT WORKS
------------
1. You enter income, expenses and the INVESTMENT HORIZON first.
2. Based on the horizon the program tells you EXACTLY which numbers to
   read on Yahoo Finance (free, no premium) and where to find them:
     short  horizon (<= 3 years)  -> use the "3 Years"  column / 3y ago
     medium horizon (<= 7 years)  -> use the "5 Years"  column / 5y ago
     long   horizon (>  7 years)  -> use the "10 Years" column / 10y ago
3. You copy the raw numbers; ALL calculations (annualization, CAGR,
   volatility estimation) are done by the program, never by you.

THE THREE METHODOLOGIES (all fed by free Yahoo Finance data)
------------------------------------------------------------
  METHOD A — RISK STATISTICS TAB (best for ETFs and funds)
      Left column of the product page -> "Risk" -> "Risk Statistics"
      table. You copy "Mean Annual Return" and "Standard Deviation"
      from the column matching your horizon.
      IMPORTANT QUIRK: despite its name, Yahoo's "Mean Annual Return"
      is the average MONTHLY return. The program annualizes it (x12).
      "Standard Deviation" is already annualized.

  METHOD B — PRICE HISTORY (works for ANY stock or ETF)
      Current price + price N years ago (N chosen by the program from
      your horizon) + 52-Week Range + dividend yield. The program
      computes:
        return     = CAGR over N years + dividend yield
        volatility = Parkinson estimator from the 52-week High/Low:
                     sigma = ln(High/Low) / (2 * sqrt(ln 2))
      The 52-Week Range is the LOWEST and HIGHEST price touched in
      the LAST 12 MONTHS (it is past history, not today's price), so
      the volatility is a 1-year estimate applied to the whole
      horizon (stated in the report).

  METHOD C — CAPM (market model)
      return = risk-free + Beta x (market return - risk-free)
      Beta comes from the SAME Risk Statistics table as Method A.
      The market return/volatility are read with the SAME procedure
      as Method A but on the page of a broad MARKET ETF (e.g. VOO or
      SPY for the S&P 500, IWDA.L or URTH for the world market) —
      Yahoo does not publish "the market return" as a single number.
      Volatility = |Beta| x market volatility (SYSTEMATIC risk only:
      it understates the total risk of a single stock; the report
      says so explicitly).

OUTPUT (all in files, saved on your DESKTOP)
--------------------------------------------
  Financial_Report_<date>.pdf   ONE professional PDF containing:
                                cover page, then for each methodology
                                the projection chart + the full report
                                (input summary, evolution table,
                                projection summary), and a final
                                comparison page. No other file is
                                produced.

DISCLAIMER (printed to the user and on every report):
  WARNING: the investment options are estimated by looking at PAST
  returns over the selected time horizon. Past returns are NOT
  indicative of future returns. We recommend investing for the long
  term whenever possible.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe; saved files are opened by the OS
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


DISCLAIMER = (
    "WARNING: the investment options are estimated by looking at PAST\n"
    "returns over the selected time horizon. Past returns are NOT\n"
    "indicative of future returns. We recommend investing for the long\n"
    "term whenever possible."
)

RECURRENCE_TO_MONTHLY = {
    "daily": 365.0 / 12.0,
    "weekly": 52.0 / 12.0,
    "monthly": 1.0,
    "yearly": 1.0 / 12.0,
}
VALID_RECURRENCES = list(RECURRENCE_TO_MONTHLY.keys())

N_SIMULATIONS = 5000
RANDOM_SEED = 42
Z_95 = 1.9600   # normal quantile for a 95% two-sided interval (2.5% per tail)

METHODS = {
    "A_risktab": ("Method A — Risk Statistics tab",
                  "Yahoo 'Risk' tab: mean monthly return (annualized x12 "
                  "by the program) and annualized standard deviation."),
    "B_history": ("Method B — Price history",
                  "CAGR from two prices + dividend yield; volatility from "
                  "the 52-week High/Low range (Parkinson estimator)."),
    "C_capm": ("Method C — CAPM",
               "Return = risk-free + Beta x market premium; volatility = "
               "|Beta| x market volatility (systematic risk only)."),
}


# ---------------------------------------------------------------------------
# Horizon profile: decides which Yahoo time windows the user must read
# ---------------------------------------------------------------------------

@dataclass
class Horizon:
    months: int

    @property
    def years(self) -> float:
        return self.months / 12.0

    @property
    def label(self) -> str:
        if self.months <= 36:
            return "SHORT"
        if self.months <= 84:
            return "MEDIUM"
        return "LONG"

    @property
    def window_label(self) -> str:
        """Column of the Yahoo 'Risk Statistics' table to read."""
        return {"SHORT": "3 Years", "MEDIUM": "5 Years",
                "LONG": "10 Years"}[self.label]

    @property
    def n_years_back(self) -> int:
        """How many years back to read the old price (Method B)."""
        return {"SHORT": 3, "MEDIUM": 5, "LONG": 10}[self.label]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class CashFlow:
    name: str
    amount: float
    recurrence: str

    @property
    def monthly_amount(self) -> float:
        return self.amount * RECURRENCE_TO_MONTHLY[self.recurrence]


@dataclass
class Asset:
    """One investment asset. The label (e.g. the ISIN) is just a name.

    `estimates` maps a method key ("A_risktab" / "B_history" / "C_capm")
    to a tuple (annual_return_pct, annual_vol_pct, note).
    """
    label: str
    initial_capital: float
    contribution: float
    recurrence: str              # daily / weekly / monthly
    ter_pct: float               # total expense ratio, % per year
    tax_rate_pct: float          # capital gains tax, %
    estimates: dict = field(default_factory=dict)

    @property
    def monthly_contribution(self) -> float:
        return self.contribution * RECURRENCE_TO_MONTHLY[self.recurrence]

    def annual_return_pct(self, method: str) -> float:
        return self.estimates[method][0]

    def annual_vol_pct(self, method: str) -> float:
        return self.estimates[method][1]

    def note(self, method: str) -> str:
        return self.estimates[method][2]

    def mu_monthly_net(self, method: str) -> float:
        """Mean monthly log-return, net of TER, for the given method."""
        net_annual = (1.0 + self.annual_return_pct(method) / 100.0) \
                     / (1.0 + self.ter_pct / 100.0)
        return np.log(net_annual) / 12.0

    def sigma_monthly(self, method: str) -> float:
        return (self.annual_vol_pct(method) / 100.0) / np.sqrt(12.0)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def ask_float(prompt: str, default: Optional[float] = None,
              min_value: Optional[float] = None,
              max_value: Optional[float] = None,
              allow_skip: bool = False) -> Optional[float]:
    """Ask for a float. If allow_skip and the user presses ENTER with no
    default, returns None (meaning: this datum is not available)."""
    while True:
        suffix = f" [{default}]" if default is not None else \
                 (" [ENTER to skip]" if allow_skip else "")
        raw = input(f"{prompt}{suffix}: ").strip().replace(",", ".")
        if raw == "":
            if default is not None:
                return default
            if allow_skip:
                return None
        try:
            value = float(raw)
            if min_value is not None and value < min_value:
                print(f"  -> Value must be at least {min_value}. Try again.")
                continue
            if max_value is not None and value > max_value:
                print(f"  -> Value must be at most {max_value}. Try again.")
                continue
            return value
        except ValueError:
            print("  -> Invalid number. Try again.")


def ask_int(prompt: str, min_value: int = 1) -> int:
    while True:
        raw = input(f"{prompt}: ").strip()
        try:
            value = int(raw)
            if value < min_value:
                print(f"  -> Value must be at least {min_value}. Try again.")
                continue
            return value
        except ValueError:
            print("  -> Invalid integer. Try again.")


def ask_choice(prompt: str, allowed: list) -> str:
    options = " / ".join(allowed)
    shortcuts = {a[0]: a for a in allowed}
    while True:
        raw = input(f"{prompt} ({options}): ").strip().lower()
        if raw in allowed:
            return raw
        if raw in shortcuts:
            return shortcuts[raw]
        print(f"  -> Please choose one of: {options}")


def ask_yes_no(prompt: str, default: Optional[bool] = None) -> bool:
    suffix = " (y/n)" if default is None else (" (Y/n)" if default else " (y/N)")
    while True:
        raw = input(f"{prompt}{suffix}: ").strip().lower()
        if raw == "" and default is not None:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  -> Please answer y or n.")


# ---------------------------------------------------------------------------
# Data guide — WHERE to find every number (personalized on the horizon)
# ---------------------------------------------------------------------------

def print_data_guide(h: Horizon) -> None:
    W = h.window_label
    N = h.n_years_back
    print("\n" + "#" * 68)
    print("HOW TO FIND THE INVESTMENT DATA ON YAHOO FINANCE (free)")
    print("#" * 68)
    print(f"""
Your horizon: {h.months} months (~{h.years:.1f} years) -> {h.label} term.
Whenever a time window is needed, use the "{W}" data as told below.

Everything is on https://finance.yahoo.com — no account, no premium.
Type the NAME, ISIN or TICKER of your fund/stock in the search bar
(e.g. "IWDA", "Apple") and open its page. You will be asked for the
data of up to THREE methods; fill in what you find and press ENTER to
skip the rest. AT LEAST ONE method is required per asset.
The program does ALL the math — you only copy numbers.

METHOD A — RISK STATISTICS TAB  (easiest for ETFs and funds)
  1. On the product page, in the LEFT column, click "Risk"
     (NOT "Summary"). A table called "Risk Statistics" appears.
  2. Go to the "{W}" column of that table.
  3. Row "Mean Annual Return": copy the number exactly as shown
     (e.g. 1.13). NOTE: despite the name, Yahoo reports here the
     average MONTHLY return — the program annualizes it (x12) for
     you. Do NOT multiply anything yourself.
  4. Row "Standard Deviation": copy as shown (e.g. 15.19). This one
     IS already the annual volatility in %.
  5. While you are there, also note the row "BETA" ("{W}" column):
     you will need it for Method C.

METHOD B — PRICE HISTORY  (works for ANY stock or ETF)
  1. On the main "Summary" page, the big number at the top is the
     CURRENT price. Copy it.
  2. Open the price chart, select the {N}Y period (or MAX), move the
     mouse to about {N} years ago and copy the price back then.
     The program computes the annualized growth (CAGR) itself.
  3. On the Summary page find "52 Week Range": two numbers, e.g.
     "110.00 - 160.00". They are the LOWEST and HIGHEST price touched
     during the LAST 12 MONTHS — it is past history, not today's
     price. Copy both. (The program turns this range into a 1-year
     volatility estimate.)
  4. Dividend yield:
       - for a STOCK: Summary page, "Forward Dividend & Yield" — the
         percentage in parentheses, e.g. "(2.10%)" -> enter 2.10;
       - for an ETF/FUND: Summary page, row "Yield" (also visible in
         "Profile" -> ETF Overview -> "Yield").
     If it shows 0.00% (typical of ACCUMULATING ETFs) enter 0: the
     dividends are already reinvested inside the price, so the CAGR
     already contains them.

METHOD C — CAPM  (market model, mainly for individual stocks)
  1. BETA: same "Risk Statistics" table as Method A, "{W}" column,
     row "BETA" (for stocks it is also on the "Statistics" tab as
     "Beta (5Y Monthly)").
  2. Risk-free rate: search the ticker  ^TNX  on Yahoo Finance; the
     quoted value IS the 10-year US Treasury yield in %
     (e.g. 4.25 means 4.25%).
  3. Market return and volatility: Yahoo does NOT publish "the market
     return" as a single number. Read it EXACTLY like Method A, but
     on the page of a broad MARKET ETF:
       e.g.  VOO or SPY  for the S&P 500,
             IWDA.L or URTH for the world market.
     Open that ETF -> "Risk" -> "{W}" column:
       "Mean Annual Return" -> enter as shown (program annualizes x12)
       "Standard Deviation" -> the market volatility.
     If unsure, press ENTER to keep the defaults (8%/y and 15%/y,
     common long-run figures for the world stock market).
  NOTE: the CAPM volatility covers only market-driven risk, so for a
  single stock it UNDERESTIMATES total risk. The report says this.
""")
    print(DISCLAIMER)
    print("#" * 68)


# ---------------------------------------------------------------------------
# Worked example (so the tool is not a black box)
# ---------------------------------------------------------------------------

def print_worked_example() -> None:
    """
    Show, with simple numbers, exactly how every output figure is
    computed — including the confidence intervals — across 3 horizons,
    so the user can verify the numbers by hand.
    """
    print("\n" + "#" * 68)
    print("WORKED EXAMPLE — how every number is computed")
    print("#" * 68)

    print("""
STEP 1 — Cash flows (income - expenses)
---------------------------------------
Everything is converted to a MONTHLY basis first:
    daily   amount x 365/12  (~30.42 per month)
    weekly  amount x 52/12   (~4.33  per month)
    monthly amount x 1
    yearly  amount / 12

Example inputs:
    Income  A: 1,000 EUR / month
    Cost    C:   200 EUR / month

Net monthly cash flow = 1,000 - 200 = 800 EUR
The cash account is simply cumulative (no interest on cash):
    month 1:   800      month 2: 1,600      month 3: 2,400 ...
    year  1: 9,600      year  2: 19,200     year  3: 28,800 ...

STEP 2 — Estimating return & volatility (the three methods)
-----------------------------------------------------------
METHOD A example: Yahoo Risk tab shows Mean Annual Return 1.13 and
Standard Deviation 14.89 (10 Years column):
    return     = 1.13 x 12 = 13.56% per year
                 (Yahoo's figure is a MONTHLY mean despite the name)
    volatility = 14.89% per year (already annualized)

METHOD B example: price today 150, price 5 years ago 100,
dividend yield 2%:
    CAGR = (150/100)^(1/5) - 1 = 8.45%   ->  return = 8.45 + 2 = 10.45%
52-week range 110 - 160 (lowest/highest of the last 12 months):
    volatility = ln(160/110) / (2 x sqrt(ln 2)) = 0.375/1.665 = 22.5%

METHOD C example: beta 1.2, risk-free 4%, market return 8%,
market volatility 15%:
    return     = 4 + 1.2 x (8 - 4)  = 8.8%
    volatility = 1.2 x 15           = 18%   (systematic risk only)

STEP 3 — Investment growth (central estimate)
---------------------------------------------
Example asset: 10,000 EUR lump sum, NO recurring contributions,
expected return 6%/year, volatility 15%/year, TER 0.2%/year.

(a) Net annual growth factor (return net of fees):
        (1 + 0.06) / (1 + 0.002) = 1.05788  ->  +5.788% per year
(b) Convert to a mean MONTHLY log-return:
        mu = ln(1.05788) / 12 = 0.004689
(c) Monthly volatility:
        sigma = 0.15 / sqrt(12) = 0.043301
(d) Central estimate after t months (compound growth):
        V(t) = 10,000 x exp(mu x t)

STEP 4 — Confidence intervals
-----------------------------
Monthly log-returns are modeled as Normal(mu, sigma). Over t months
the TOTAL log-return is Normal(mu x t, sigma x sqrt(t)), so for a
lump sum the interval bounds have a closed formula you can verify:

        V_bound(t) = 10,000 x exp( mu x t  ±  z x sigma x sqrt(t) )

        z = 1.960 for the 95% interval (2.5% in each tail)
""")

    v0, mu, sigma = 10_000.0, np.log(1.06 / 1.002) / 12.0, 0.15 / np.sqrt(12.0)
    print("Applying the formulas on 3 horizons (values in EUR):\n")
    header = (f"{'Horizon':<10}{'t':>5}{'Central':>12}"
              f"{'95% low':>12}{'95% high':>12}")
    print(header)
    print("-" * len(header))
    for label, t in (("1 year", 12), ("5 years", 60), ("10 years", 120)):
        central = v0 * np.exp(mu * t)
        lo95 = v0 * np.exp(mu * t - Z_95 * sigma * np.sqrt(t))
        hi95 = v0 * np.exp(mu * t + Z_95 * sigma * np.sqrt(t))
        print(f"{label:<10}{t:>5}{central:>12,.0f}"
              f"{lo95:>12,.0f}{hi95:>12,.0f}")

    print("""
STEP 5 — Recurring contributions and multiple assets
----------------------------------------------------
With recurring contributions the closed formula above no longer
exists (a sum of lognormals), so the program runs a MONTE CARLO
simulation: 5,000 random paths where each month
        value_new = (value_old + contribution) x exp(r),
        r drawn from Normal(mu, sigma),
and the confidence band is the 2.5th/97.5th percentile (95% CI)
across the 5,000 paths.
The central curve grows deterministically at exp(mu) — it is the
MEDIAN outcome, a prudent central estimate.

With MULTIPLE assets, each asset has its own mu and sigma, and the
monthly random shocks are CORRELATED across assets (you choose the
average correlation; equity ETFs typically move together, 0.7-0.9).

Taxes: at the end of the horizon the capital gain of each asset
(value - contributed capital) is taxed at its tax rate, assuming
full liquidation.
""")
    print("#" * 68)


# ---------------------------------------------------------------------------
# Methodology estimators (ALL math done by the program)
# ---------------------------------------------------------------------------

def estimate_from_risktab(mean_monthly_pct: float, sd_annual_pct: float,
                          window: str) -> tuple:
    """METHOD A: annualize Yahoo's monthly mean (x12); SD is annual."""
    ret = mean_monthly_pct * 12.0
    note = (f"Yahoo Risk tab, {window} column: 'Mean Annual Return' "
            f"{mean_monthly_pct:g} is a MONTHLY mean -> x12 = {ret:.2f}%/y; "
            f"Standard Deviation {sd_annual_pct:g}% already annual")
    return ret, sd_annual_pct, note


def estimate_from_history(p_now: float, p_then: float, years: float,
                          hi52: float, lo52: float,
                          div_yield_pct: float) -> tuple:
    """METHOD B: CAGR + dividend yield; Parkinson vol from 52w range."""
    cagr = ((p_now / p_then) ** (1.0 / years) - 1.0) * 100.0
    ret = cagr + div_yield_pct
    vol = np.log(hi52 / lo52) / (2.0 * np.sqrt(np.log(2.0))) * 100.0
    note = (f"return = CAGR {cagr:+.2f}% over {years:g}y + dividend "
            f"{div_yield_pct:.2f}%; vol = ln({hi52:g}/{lo52:g})/(2*sqrt(ln2)) "
            f"[Parkinson on last-12-months range, 1y estimate]")
    return ret, vol, note


def estimate_from_capm(beta: float, rf_pct: float, rm_pct: float,
                       market_vol_pct: float) -> tuple:
    """METHOD C: CAPM return; vol = |beta| x market vol (systematic only)."""
    ret = rf_pct + beta * (rm_pct - rf_pct)
    vol = abs(beta) * market_vol_pct
    note = (f"return = {rf_pct:g}% + {beta:g} x ({rm_pct:g}% - {rf_pct:g}%); "
            f"vol = |{beta:g}| x {market_vol_pct:g}% market vol "
            f"(SYSTEMATIC RISK ONLY — understates single-stock risk)")
    return ret, vol, note


# ---------------------------------------------------------------------------
# Input sections
# ---------------------------------------------------------------------------

def input_cash_flows(kind: str) -> list:
    items = []
    print(f"\n--- {kind.upper()} ---")
    print(f"Add your {kind} items one by one. "
          f"Press ENTER (or type 'done') when finished.")
    while True:
        i = len(items) + 1
        print(f"\n{kind.capitalize()} #{i}")
        name = input("  Name (ENTER to finish): ").strip()
        if name == "" or name.lower() in ("done", "exit", "q"):
            break
        amount = ask_float("  Amount per occurrence (EUR)", min_value=0.0)
        recurrence = ask_choice("  Recurrence", VALID_RECURRENCES)
        items.append(CashFlow(name=name, amount=amount, recurrence=recurrence))
        print(f"  Added: {name} — {amount:,.2f} EUR ({recurrence})")
    print(f"Total {kind} items added: {len(items)}")
    return items


def input_horizon() -> Horizon:
    print("\n--- INVESTMENT HORIZON ---")
    print("How long do you plan to keep the money invested?")
    print("(Minimum 1 year: shorter horizons make historical estimates")
    print(" meaningless. This choice also decides which time windows you")
    print(" will be asked to read on Yahoo Finance.)")
    while True:
        raw = input("Horizon unit (months / years): ").strip().lower()
        if raw in ("months", "month", "m"):
            months = ask_int("Number of months", min_value=12)
            break
        if raw in ("years", "year", "y"):
            months = ask_int("Number of years", min_value=1) * 12
            break
        print("  -> Please type 'months' or 'years'.")
    h = Horizon(months)
    print(f"\nHorizon set: {h.months} months (~{h.years:.1f} years) "
          f"-> {h.label} term.")
    print(f"You will be told to read the \"{h.window_label}\" data and the "
          f"price {h.n_years_back} years ago.")
    return h


def input_method_a(h: Horizon) -> Optional[tuple]:
    W = h.window_label
    print(f"\n  METHOD A — Risk Statistics tab (left column -> Risk)")
    v = ask_float(f"    'Mean Annual Return', {W} column, AS SHOWN "
                  f"(e.g. 1.13)", allow_skip=True)
    if v is None:
        print("    -> Method A skipped for this asset.")
        return None
    sd = ask_float(f"    'Standard Deviation', {W} column, AS SHOWN "
                   f"(e.g. 15.19)", min_value=0.0)
    ret, vol, note = estimate_from_risktab(v, sd, W)
    print(f"    -> Computed: return {ret:+.2f}%/y (= {v:g} x 12), "
          f"volatility {vol:.2f}%/y")
    return ret, vol, note


def input_method_b(h: Horizon) -> Optional[tuple]:
    N = h.n_years_back
    print("\n  METHOD B — Price history (Summary page + chart)")
    p_now = ask_float("    Current price (big number at the top)",
                      min_value=0.0001, allow_skip=True)
    if p_now is None:
        print("    -> Method B skipped for this asset.")
        return None
    p_then = ask_float(f"    Price about {N} years ago (from the {N}Y chart)",
                       min_value=0.0001)
    hi52 = ask_float("    52 Week Range — HIGHEST value (last 12 months)",
                     min_value=0.0001)
    lo52 = ask_float("    52 Week Range — LOWEST value (last 12 months)",
                     min_value=0.0001)
    if lo52 > hi52:
        lo52, hi52 = hi52, lo52
        print("    (swapped: low must be below high)")
    dy = ask_float("    Dividend yield (%) — 0 for accumulating ETFs",
                   default=0.0, min_value=0.0)
    ret, vol, note = estimate_from_history(p_now, p_then, float(N),
                                           hi52, lo52, dy)
    print(f"    -> Computed: return {ret:+.2f}%/y, volatility {vol:.2f}%/y")
    return ret, vol, note


def input_method_c(h: Horizon) -> Optional[tuple]:
    W = h.window_label
    print("\n  METHOD C — CAPM (Beta + market data)")
    beta = ask_float(f"    BETA (Risk Statistics table, {W} column)",
                     allow_skip=True)
    if beta is None:
        print("    -> Method C skipped for this asset.")
        return None
    rf = ask_float("    Risk-free rate, ^TNX quote (%)", default=4.0)
    print("    Market data: open a broad market ETF (VOO/SPY or IWDA.L)")
    print(f"    -> Risk tab -> {W} column. ENTER keeps the defaults.")
    m_mean = ask_float("    Market 'Mean Annual Return' AS SHOWN "
                       "(monthly mean)", allow_skip=True)
    rm = m_mean * 12.0 if m_mean is not None else 8.0
    if m_mean is not None:
        print(f"      -> market return = {m_mean:g} x 12 = {rm:.2f}%/y")
    else:
        print("      -> default market return 8%/y")
    mv = ask_float("    Market 'Standard Deviation' AS SHOWN (annual %)",
                   default=15.0, min_value=0.0)
    ret, vol, note = estimate_from_capm(beta, rf, rm, mv)
    print(f"    -> Computed: return {ret:+.2f}%/y, volatility {vol:.2f}%/y")
    return ret, vol, note


def input_assets(h: Horizon) -> tuple:
    """Collect any number of investment assets, all parameters manual."""
    assets = []
    print("\n--- INVESTMENTS ---")
    print("Add as many assets as you want. Press ENTER (or 'done') to finish.")
    print("For each asset you can provide the data of up to 3 estimation")
    print("methods (see the guide above). Skip what you did not find;")
    print("AT LEAST ONE method is required per asset.")
    while True:
        i = len(assets) + 1
        print(f"\nAsset #{i}")
        label = input("  Name/ISIN (identifier only, ENTER to finish): ").strip()
        if label == "" or label.lower() in ("done", "exit", "q"):
            break
        initial = ask_float("  Initial capital already invested (EUR)",
                            default=0.0, min_value=0.0)
        contribution = ask_float("  Recurring contribution amount (EUR)",
                                 default=0.0, min_value=0.0)
        recurrence = "monthly"
        if contribution > 0:
            recurrence = ask_choice("  Contribution recurrence",
                                    ["daily", "weekly", "monthly"])
        ter = ask_float("  TER (% per year)", default=0.20, min_value=0.0)
        tax = ask_float("  Capital gains tax rate (%)", default=26.0,
                        min_value=0.0)

        estimates = {}
        while not estimates:
            a = input_method_a(h)
            if a is not None:
                estimates["A_risktab"] = a
            b = input_method_b(h)
            if b is not None:
                estimates["B_history"] = b
            c = input_method_c(h)
            if c is not None:
                estimates["C_capm"] = c
            if not estimates:
                print("\n  !! You skipped all three methods. At least one is")
                print("  required — please fill in the data for one of them.")

        assets.append(Asset(
            label=label, initial_capital=initial, contribution=contribution,
            recurrence=recurrence, ter_pct=ter, tax_rate_pct=tax,
            estimates=estimates,
        ))
        found = ", ".join(METHODS[k][0] for k in estimates)
        print(f"  Added: {label} — {contribution:,.2f} EUR ({recurrence}), "
              f"TER {ter}%, tax {tax}%")
        print(f"  Methods available: {found}")
    print(f"Total assets added: {len(assets)}")

    correlation = 0.0
    if len(assets) > 1:
        print("\nAverage correlation between the assets' monthly returns.")
        print("Guideline: equity ETFs on similar markets 0.7-0.9; "
              "stocks vs bonds 0.0-0.3.")
        correlation = ask_float("Correlation (0 to 1)", default=0.7,
                                min_value=0.0, max_value=1.0)
    return assets, correlation


# ---------------------------------------------------------------------------
# Simulation (one run per methodology)
# ---------------------------------------------------------------------------

def simulate(incomes: list, expenses: list, assets: list,
             correlation: float, months: int, method: str) -> dict:
    """
    Month-by-month projection for ONE methodology.

    Cash:    cumulative (income - expenses - total contributions).
    Assets:  central path compounding at exp(mu_net) per month;
             Monte Carlo (correlated shocks across assets) for the
             portfolio confidence bands.
    """
    monthly_income = sum(cf.monthly_amount for cf in incomes)
    monthly_expenses = sum(cf.monthly_amount for cf in expenses)
    monthly_contrib = sum(a.monthly_contribution for a in assets)
    monthly_net_cash = monthly_income - monthly_expenses - monthly_contrib

    t = np.arange(months + 1)
    cash_path = monthly_net_cash * t

    result = {
        "method": method,
        "months": months,
        "monthly_income": monthly_income,
        "monthly_expenses": monthly_expenses,
        "monthly_contrib": monthly_contrib,
        "monthly_net_cash": monthly_net_cash,
        "cash_path": cash_path,
        "assets": assets,
        "correlation": correlation,
    }

    n_assets = len(assets)
    if n_assets == 0:
        zero = np.zeros(months + 1)
        result.update({
            "asset_expected": [], "portfolio_expected": zero,
            "contributed": zero, "band95": None,
            "total_expected": cash_path.copy(),
            "taxes": 0.0, "net_portfolio_value": 0.0,
            "final_wealth": cash_path[-1],
            "gross_gain": 0.0, "total_contributed": 0.0, "gross_value": 0.0,
        })
        return result

    C = np.array([a.monthly_contribution for a in assets])
    mu = np.array([a.mu_monthly_net(method) for a in assets])
    sigma = np.array([a.sigma_monthly(method) for a in assets])
    V0 = np.array([a.initial_capital for a in assets])

    # --- Central (median) path per asset: deterministic compounding ---
    asset_expected = np.empty((months + 1, n_assets))
    contributed = np.empty((months + 1, n_assets))
    asset_expected[0] = V0
    contributed[0] = V0
    g = np.exp(mu)
    for m in range(1, months + 1):
        asset_expected[m] = (asset_expected[m - 1] + C) * g
        contributed[m] = contributed[m - 1] + C
    portfolio_expected = asset_expected.sum(axis=1)

    # --- Monte Carlo with equicorrelated shocks across assets ---
    # shock_i = sqrt(rho) * common + sqrt(1-rho) * idiosyncratic_i
    rng = np.random.default_rng(RANDOM_SEED)
    rho = correlation if n_assets > 1 else 0.0
    values = np.tile(V0, (N_SIMULATIONS, 1))          # (sims, assets)
    portfolio_paths = np.empty((months + 1, N_SIMULATIONS))
    portfolio_paths[0] = values.sum(axis=1)
    for m in range(1, months + 1):
        common = rng.standard_normal((N_SIMULATIONS, 1))
        idio = rng.standard_normal((N_SIMULATIONS, n_assets))
        z = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio
        r = mu + sigma * z                             # (sims, assets)
        values = (values + C) * np.exp(r)
        portfolio_paths[m] = values.sum(axis=1)

    band95 = (np.percentile(portfolio_paths, 2.5, axis=1),
              np.percentile(portfolio_paths, 97.5, axis=1))

    # --- Taxes at end of horizon on the central estimate, per asset ---
    taxes = 0.0
    for j, a in enumerate(assets):
        gain = asset_expected[-1, j] - contributed[-1, j]
        if gain > 0:
            taxes += gain * a.tax_rate_pct / 100.0

    gross_value = portfolio_expected[-1]
    total_contributed = contributed[-1].sum()
    net_portfolio_value = gross_value - taxes

    result.update({
        "asset_expected": asset_expected,
        "portfolio_expected": portfolio_expected,
        "contributed": contributed,
        "band95": band95,
        "total_expected": cash_path + portfolio_expected,
        "gross_value": gross_value,
        "total_contributed": total_contributed,
        "gross_gain": gross_value - total_contributed,
        "taxes": taxes,
        "net_portfolio_value": net_portfolio_value,
        "final_wealth": cash_path[-1] + net_portfolio_value,
    })
    return result


# ---------------------------------------------------------------------------
# Output: report text (per methodology)
# ---------------------------------------------------------------------------

def format_method_header(method: str) -> str:
    title, desc = METHODS[method]
    L = []
    L.append("=" * 68)
    L.append(f"METHODOLOGY: {title}")
    L.append("=" * 68)
    L.append(desc)
    return "\n".join(L)


def format_input_summary(incomes: list, expenses: list, assets: list,
                         correlation: float, method: str) -> str:
    L = []
    L.append("=" * 68)
    L.append("INPUT SUMMARY")
    L.append("=" * 68)

    L.append("")
    L.append("Income:")
    if not incomes:
        L.append("  (none)")
    for cf in incomes:
        L.append(f"  {cf.name:<24}{cf.amount:>12,.2f} EUR  {cf.recurrence:<8}"
                 f"(= {cf.monthly_amount:>10,.2f} EUR/month)")

    L.append("")
    L.append("Cost:")
    if not expenses:
        L.append("  (none)")
    for cf in expenses:
        L.append(f"  {cf.name:<24}{cf.amount:>12,.2f} EUR  {cf.recurrence:<8}"
                 f"(= {cf.monthly_amount:>10,.2f} EUR/month)")

    L.append("")
    L.append("Investments:")
    if not assets:
        L.append("  (none)")
    for a in assets:
        L.append(f"  {a.label:<24}initial {a.initial_capital:>10,.2f} EUR, "
                 f"{a.contribution:>8,.2f} EUR {a.recurrence}")
        L.append(f"  {'':<24}return {a.annual_return_pct(method):+.2f}%/y, "
                 f"vol {a.annual_vol_pct(method):.2f}%/y, TER {a.ter_pct}%, "
                 f"tax {a.tax_rate_pct}%")
        # how the numbers were obtained (wrapped to fit the page)
        note = a.note(method)
        while note:
            L.append(f"  {'':<24}| {note[:64]}")
            note = note[64:]
    if len(assets) > 1:
        L.append("")
        L.append(f"  Assumed correlation between assets: {correlation:.2f}")
    return "\n".join(L)


def format_evolution_table(r: dict) -> str:
    """
    Evolution table: cash, contributed capital, expected investment
    value, 95% band, total. Monthly rows for horizons up to 24 months,
    yearly rows otherwise.
    """
    months = r["months"]
    has_assets = len(r["assets"]) > 0

    if months <= 24:
        steps = list(range(1, months + 1))
    else:
        steps = list(range(12, months + 1, 12))
        if steps[-1] != months:
            steps.append(months)
    unit = "Month"

    L = []
    L.append("=" * 68)
    L.append("EVOLUTION TABLE  (EUR)")
    L.append("=" * 68)
    if has_assets:
        header = (f"{unit:>6} | {'Income-Cost':>12} | {'Contributed':>12} | "
                  f"{'Invest.(exp)':>12} | {'95% low':>11} | {'95% high':>11} | "
                  f"{'Total(exp)':>12}")
    else:
        header = f"{unit:>6} | {'Income-Cost':>12} | {'Total':>12}"
    L.append(header)
    L.append("-" * len(header))

    for m in steps:
        cash = r["cash_path"][m]
        if has_assets:
            contrib = r["contributed"][m].sum()
            inv = r["portfolio_expected"][m]
            lo95 = r["band95"][0][m]
            hi95 = r["band95"][1][m]
            total = r["total_expected"][m]
            L.append(f"{m:>6} | {cash:>12,.0f} | {contrib:>12,.0f} | "
                     f"{inv:>12,.0f} | {lo95:>11,.0f} | {hi95:>11,.0f} | "
                     f"{total:>12,.0f}")
        else:
            L.append(f"{m:>6} | {cash:>12,.0f} | {cash:>12,.0f}")

    L.append("-" * len(header))
    L.append("Income-Cost = cumulative (income - expenses - contributions).")
    if has_assets:
        L.append("Invest.(exp) = central (median) portfolio value before taxes.")
        L.append("Total(exp)   = Income-Cost + Invest.(exp), before taxes.")
    return "\n".join(L)


def format_summary(r: dict) -> str:
    m = r["months"]
    assets = r["assets"]
    method = r["method"]
    L = []
    L.append("=" * 68)
    L.append(f"PROJECTION SUMMARY  ({m} months = {m / 12:.1f} years)  "
             f"[{METHODS[method][0]}]")
    L.append("=" * 68)
    L.append(f"Monthly income:              {r['monthly_income']:>14,.2f} EUR")
    L.append(f"Monthly expenses:            {r['monthly_expenses']:>14,.2f} EUR")
    L.append(f"Monthly invested (plans):    {r['monthly_contrib']:>14,.2f} EUR")
    L.append(f"Monthly net cash flow:       {r['monthly_net_cash']:>14,.2f} EUR")
    if r["monthly_net_cash"] < 0:
        L.append("  WARNING: negative net cash flow — you spend more than you")
        L.append("  earn after investment contributions.")
    L.append("-" * 68)
    L.append(f"Cash accumulated:            {r['cash_path'][-1]:>14,.2f} EUR")

    if assets:
        lo95, hi95 = r["band95"][0][-1], r["band95"][1][-1]
        L.append("")
        L.append(f"Portfolio ({len(assets)} asset(s)):")
        for j, a in enumerate(assets):
            L.append(f"  {a.label}: contributed "
                     f"{r['contributed'][-1][j]:,.2f} EUR -> expected "
                     f"{r['asset_expected'][-1][j]:,.2f} EUR")
        L.append("")
        L.append(f"  Total contributed:         {r['total_contributed']:>14,.2f} EUR")
        L.append(f"  Expected gross value:      {r['gross_value']:>14,.2f} EUR")
        L.append(f"  95% confidence interval:   "
                 f"{lo95:>14,.2f} — {hi95:,.2f} EUR")
        L.append(f"  Expected gross gain:       {r['gross_gain']:>14,.2f} EUR")
        L.append(f"  Capital gains tax:         {r['taxes']:>14,.2f} EUR")
        L.append(f"  Expected net value:        {r['net_portfolio_value']:>14,.2f} EUR")
        L.append("-" * 68)
        L.append(f"EXPECTED FINAL NET WORTH:    {r['final_wealth']:>14,.2f} EUR")
    else:
        L.append("-" * 68)
        L.append(f"FINAL NET WORTH (no invest): {r['cash_path'][-1]:>14,.2f} EUR")
    L.append("=" * 68)
    L.append("")
    L.append(DISCLAIMER)
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Output: figures and PDF assembly
# ---------------------------------------------------------------------------

A4_PORTRAIT = (8.27, 11.69)
A4_LANDSCAPE = (11.69, 8.27)


def desktop_dir() -> Path:
    """Save on the user's Desktop; fall back to the current folder."""
    d = Path.home() / "Desktop"
    return d if d.is_dir() else Path.cwd()


def build_projection_fig(r: dict) -> plt.Figure:
    m = r["months"]
    x = np.arange(m + 1)
    assets = r["assets"]
    method = r["method"]

    fig, ax = plt.subplots(figsize=A4_LANDSCAPE)

    if assets:
        lo95, hi95 = r["band95"]
        ax.fill_between(x, lo95, hi95, alpha=0.18, color="tab:green",
                        label="Portfolio 95% CI")

    ax.plot(x, r["cash_path"], linewidth=1.8, linestyle="--",
            color="tab:orange", label="Cash: income − expenses")

    if assets:
        ax.plot(x, r["portfolio_expected"], linewidth=1.8, color="tab:green",
                label="Investments (expected)")
        ax.plot(x, r["contributed"].sum(axis=1), linewidth=1.0,
                linestyle=":", color="gray", label="Contributed capital")

    ax.plot(x, r["total_expected"], linewidth=2.6, color="tab:blue",
            label="Total: cash + investments")

    ax.set_xlabel("Months")
    ax.set_ylabel("EUR")
    title = f"Wealth projection — {METHODS[method][0]}"
    if assets:
        labels = ", ".join(a.label for a in assets[:3])
        if len(assets) > 3:
            labels += ", ..."
        title += f"  |  {labels}"
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8)

    fig.text(0.5, 0.005,
             "Estimates based on past returns — past performance is not "
             "indicative of future results.",
             ha="center", fontsize=8, style="italic", color="dimgray")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def build_comparison_fig(results: dict) -> plt.Figure:
    """One chart with the central INVESTMENT curve of each methodology."""
    fig, ax = plt.subplots(figsize=A4_LANDSCAPE)
    colors = {"A_risktab": "tab:green", "B_history": "tab:purple",
              "C_capm": "tab:red"}
    contributed_drawn = False
    for mk, r in results.items():
        x = np.arange(r["months"] + 1)
        ax.plot(x, r["portfolio_expected"], linewidth=2.2,
                color=colors.get(mk, None), label=METHODS[mk][0])
        lo95, hi95 = r["band95"]
        ax.fill_between(x, lo95, hi95, alpha=0.10,
                        color=colors.get(mk, None))
        if not contributed_drawn:
            ax.plot(x, r["contributed"].sum(axis=1), linewidth=1.0,
                    linestyle=":", color="gray", label="Contributed capital")
            contributed_drawn = True

    ax.set_xlabel("Months")
    ax.set_ylabel("EUR")
    ax.set_title("Investment value — comparison of the methodologies "
                 "(central estimate with 95% CI)", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8)
    fig.text(0.5, 0.005,
             "Estimates based on past returns — past performance is not "
             "indicative of future results.",
             ha="center", fontsize=8, style="italic", color="dimgray")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def build_cover_fig(h: Horizon, assets: list, method_keys: list) -> plt.Figure:
    fig = plt.figure(figsize=A4_PORTRAIT)
    fig.text(0.5, 0.86, "PERSONAL FINANCIAL PLAN", ha="center",
             fontsize=26, fontweight="bold")
    fig.text(0.5, 0.82, "Multi-methodology wealth projection, project by Edoardo Maira", ha="center",
             fontsize=13, color="dimgray")
    fig.lines.append(plt.Line2D([0.15, 0.85], [0.79, 0.79],
                                transform=fig.transFigure,
                                color="black", linewidth=1.2))

    stamp = datetime.now().strftime("%d %B %Y, %H:%M")
    info = [
        ("Generated", stamp),
        ("Horizon", f"{h.months} months (~{h.years:.1f} years, "
                    f"{h.label.lower()} term)"),
        ("Yahoo data window", h.window_label),
        ("Assets", ", ".join(a.label for a in assets) if assets
                   else "(none — cash only)"),
        ("Methodologies", ""),
    ]
    # Manual wrapping: matplotlib's wrap=True is unreliable, so every
    # value is wrapped with textwrap to a fixed character width that
    # fits between x=0.38 and the right margin at the given font size.
    y = 0.74
    for k, v in info:
        fig.text(0.15, y, f"{k}:", fontsize=11, fontweight="bold")
        v_lines = textwrap.wrap(v, width=46) or [""]
        for line in v_lines:
            fig.text(0.38, y, line, fontsize=11)
            y -= 0.030
        y -= 0.012
    for mk in method_keys:
        fig.text(0.38, y, f"• {METHODS[mk][0]}", fontsize=11)
        y -= 0.028
        for line in textwrap.wrap(METHODS[mk][1], width=58):
            fig.text(0.40, y, line, fontsize=8.5, color="dimgray")
            y -= 0.021
        y -= 0.016

    fig.text(0.5, 0.16, DISCLAIMER, ha="center", fontsize=9,
             style="italic", color="dimgray",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="whitesmoke",
                       edgecolor="lightgray"))
    fig.text(0.5, 0.05, "Data source: Yahoo Finance (finance.yahoo.com), "
             "free public pages.", ha="center", fontsize=8, color="gray")
    return fig


def add_text_pages(pdf: PdfPages, text: str, lines_per_page: int = 84) -> None:
    """Render a monospace text report as one or more A4 PDF pages."""
    lines = text.split("\n")
    for i in range(0, len(lines), lines_per_page):
        chunk = "\n".join(lines[i:i + lines_per_page])
        fig = plt.figure(figsize=A4_PORTRAIT)
        fig.text(0.07, 0.965, chunk, va="top", ha="left",
                 family="monospace", fontsize=7.5)
        pdf.savefig(fig)
        plt.close(fig)


def export_outputs(h: Horizon, assets: list, results: dict,
                   reports: dict) -> list:
    """Build the single PDF report on the Desktop (no other file)."""
    out = desktop_dir()
    stamp = datetime.now().strftime("%Y-%m-%d")
    pdf_path = out / f"Financial_Report_{stamp}.pdf"
    generated = []

    with PdfPages(pdf_path) as pdf:
        cover = build_cover_fig(h, assets, list(results.keys()))
        pdf.savefig(cover)
        plt.close(cover)

        for mk, r in results.items():
            fig = build_projection_fig(r)     # page 1: the chart
            pdf.savefig(fig)
            plt.close(fig)
            add_text_pages(pdf, reports[mk])  # page 2+: data + full output

        if len(results) > 1 and assets:
            fig = build_comparison_fig(results)
            pdf.savefig(fig)
            plt.close(fig)

        meta = pdf.infodict()
        meta["Title"] = "Personal Financial Plan"
        meta["Author"] = "Personal Financial Planner"
        meta["CreationDate"] = datetime.now()
    print(f"\nPDF report saved to: {pdf_path}")
    generated.append(pdf_path)
    return generated


def open_files(paths: list) -> None:
    """Open the saved files with the OS default viewer."""
    import os
    import subprocess
    for p in paths:
        path = str(Path(p).resolve())
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", path], check=False,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            elif sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", path], check=False,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 68)
    print("PERSONAL FINANCIAL PLANNER — PRO EDITION")
    print("=" * 68)

    if ask_yes_no("\nShow the worked example (how all numbers are computed)?",
                  default=True):
        print_worked_example()

    incomes = input_cash_flows("income")
    expenses = input_cash_flows("expense")

    # Horizon FIRST: it decides which Yahoo windows the user must read.
    h = input_horizon()

    # The guide is shown BEFORE asking for the investment inputs.
    print_data_guide(h)

    assets, correlation = input_assets(h)

    # A methodology is run only if EVERY asset has data for it, so that
    # each report describes the whole portfolio consistently.
    if assets:
        available = [mk for mk in METHODS
                     if all(mk in a.estimates for a in assets)]
        skipped = [mk for mk in METHODS if mk not in available
                   and any(mk in a.estimates for a in assets)]
        for mk in skipped:
            print(f"\nNOTE: {METHODS[mk][0]} was provided only for some "
                  f"assets, so it is skipped (each methodology needs data "
                  f"for ALL assets).")
        if not available:
            print("\nNo methodology has data for all assets. Exiting.")
            return
    else:
        available = ["A_risktab"]   # cash-only run, method is irrelevant

    results = {}
    reports = {}
    for mk in available:
        r = simulate(incomes, expenses, assets, correlation, h.months, mk)
        results[mk] = r
        report = "\n".join([
            format_method_header(mk),
            "",
            format_input_summary(incomes, expenses, assets, correlation, mk),
            "",
            format_evolution_table(r),
            "",
            format_summary(r),
        ])
        reports[mk] = report
        print("\n" + report + "\n")

    generated = export_outputs(h, assets, results, reports)

    print("\n" + "=" * 68)
    print("GENERATED FILES (on your Desktop)")
    print("=" * 68)
    for f in generated:
        print(f"  {f}")

    if ask_yes_no("\nOpen the generated files now?", default=True):
        open_files(generated)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
        sys.exit(0)
