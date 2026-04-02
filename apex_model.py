"""
APEX DCF++™ — Logique métier pure
Importable depuis Streamlit, Colab ou tout script Python.
Aucune dépendance à ipywidgets ou à Streamlit ici.
"""

import numpy as np
import pandas as pd
import numpy_financial as npf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy.optimize import brentq
import calendar
import logging
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger("APEX")


# ================================================================
#  DATACLASSES — Hypothèses typées avec validation
# ================================================================

@dataclass
class MarketAssumptions:
    """Hypothèses macro-économiques et fiscales (France)."""
    revpar_growth_mean: float = 0.025
    revpar_growth_std:  float = 0.015
    opex_inflation_mean: float = 0.030
    opex_inflation_std:  float = 0.010
    exit_cap_mean: float = 0.055
    exit_cap_std:  float = 0.005
    market_volatility_revpar: float = 0.10
    corporate_tax_rate: float = 0.25
    capital_gains_tax_rate: float = 0.25

    def __post_init__(self):
        assert 0 < self.exit_cap_mean < 0.30, "exit_cap_mean hors plage [0–30%]"
        assert 0 <= self.corporate_tax_rate <= 1, "Taux IS invalide"
        assert self.revpar_growth_std >= 0, "σ RevPAR ne peut pas être négatif"


@dataclass
class HotelOperatingInputs:
    """Données opérationnelles de l'actif hôtelier (norme USALI)."""
    rooms: int
    purchase_price: float
    base_revpar: float
    base_occupancy: float
    hotel_name: str = "Hôtel"

    land_value_pct: float = 0.20
    asset_useful_life_years: int = 25

    rev_segmentation: Dict[str, float] = field(default_factory=lambda: {
        "Rooms": 0.65, "F&B": 0.25, "Other": 0.10
    })
    opex_ratios: Dict[str, float] = field(default_factory=lambda: {
        "Rooms": 0.25, "F&B": 0.70, "Admin": 0.08,
        "Sales": 0.06, "Utilities": 0.04, "Property Tax": 0.03, "Insurance": 0.01
    })
    fixed_opex_per_room_annual: float = 5_000
    management_fee_base_pct: float = 0.03
    management_fee_incentive_pct: float = 0.10
    management_fee_incentive_threshold_pct: float = 0.35
    renovation_capex: List[Tuple[int, float]] = field(default_factory=list)
    ffe_reserve_pct: float = 0.04

    dso_days: int = 30
    dpo_days: int = 45
    dio_days: int = 15
    min_cash_days_opex: int = 30

    monthly_seasonality_factors: Dict[int, float] = field(default_factory=lambda: {
        1: 0.75, 2: 0.80, 3: 0.95, 4: 1.05, 5: 1.10, 6: 1.15,
        7: 1.20, 8: 1.18, 9: 1.12, 10: 1.05, 11: 0.90, 12: 0.85
    })

    def __post_init__(self):
        assert self.rooms > 0
        assert 0 < self.base_occupancy <= 1.0
        assert self.base_revpar > 0
        assert self.purchase_price > 0
        assert abs(sum(self.rev_segmentation.values()) - 1.0) < 0.01, \
            "Les segments de revenus doivent totaliser 100%"


@dataclass
class DebtStructure:
    """Structure financière dette senior + mezzanine."""
    senior_loan_amount: float
    senior_interest_rate: float
    senior_term_years: int
    senior_amortization_years: int
    mezzanine_loan_amount: float = 0.0
    mezzanine_interest_rate: float = 0.0

    def __post_init__(self):
        assert 0 < self.senior_interest_rate < 0.25
        assert self.senior_loan_amount > 0
        if self.mezzanine_loan_amount > 0:
            assert self.mezzanine_interest_rate > self.senior_interest_rate, \
                "Taux mezzanine doit être > taux senior"


@dataclass
class ESGTrajectory:
    """Trajectoire ESG annuelle (scores 0–100) conforme GRESB."""
    esg_scores_per_year: List[float]

    def __post_init__(self):
        assert all(0 <= s <= 100 for s in self.esg_scores_per_year)
        assert len(self.esg_scores_per_year) >= 1


# ================================================================
#  ESG INTEGRATOR
# ================================================================

class APEX_ESG_Integrator:
    """Traduit le score ESG en impacts financiers quantifiés."""

    _THRESHOLDS = [
        (85, 0.08, 0.060, -50, "Leader ESG (AAA)"),
        (75, 0.05, 0.040, -35, "Avancé ESG (AA)"),
        (65, 0.03, 0.025, -20, "Engagé ESG (A)"),
        ( 0, 0.00, 0.000,   0, "Standard ESG"),
    ]

    def __init__(self, esg_trajectory: ESGTrajectory):
        self.esg_scores_per_year = esg_trajectory.esg_scores_per_year

    def get_score(self, year_idx: int) -> float:
        return self.esg_scores_per_year[min(year_idx, len(self.esg_scores_per_year) - 1)]

    def get_financial_impacts(self, year_idx: int) -> Dict[str, float]:
        score = self.get_score(year_idx)
        for threshold, revpar_premium, opex_reduction, cap_bps, label in self._THRESHOLDS:
            if score >= threshold:
                return {
                    "esg_score": score,
                    "esg_label": label,
                    "revpar_premium": revpar_premium,
                    "opex_reduction": opex_reduction,
                    "cap_rate_compression_bps": cap_bps,
                    "volatility_reduction_factor": max(0.0, (100 - score) / 100),
                }

    def get_rating_summary(self) -> pd.DataFrame:
        rows = []
        for i, score in enumerate(self.esg_scores_per_year):
            impacts = self.get_financial_impacts(i)
            rows.append({
                "Année": i + 1,
                "Score ESG": score,
                "Label": impacts["esg_label"],
                "Prime RevPAR (%)": impacts["revpar_premium"] * 100,
                "Réduction OpEx (%)": impacts["opex_reduction"] * 100,
                "Compression Cap Rate (bps)": impacts["cap_rate_compression_bps"],
            })
        return pd.DataFrame(rows).set_index("Année")


# ================================================================
#  CORE MODELER
# ================================================================

class APEX_Modeler:
    """Modèle financier institutionnel — granularité mixte mensuel/annuel."""

    def __init__(
        self,
        hotel_ops: HotelOperatingInputs,
        market_ass: MarketAssumptions,
        debt_struct: DebtStructure,
        esg_integrator: APEX_ESG_Integrator,
        holding_period: int = 10,
        monthly_detail_years: int = 3,
        start_year: int = 2025,
        seed: Optional[int] = None,
    ):
        self.hotel_ops = hotel_ops
        self.market_ass = market_ass
        self.debt_struct = debt_struct
        self.esg_integrator = esg_integrator
        self.holding_period = holding_period
        self.monthly_detail_years = monthly_detail_years
        self.start_year = start_year
        self.rng = np.random.default_rng(seed)

    def generate_annual_cash_flows(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        monthly_dfs, annual_rows = [], []

        dep_val    = self.hotel_ops.purchase_price * (1 - self.hotel_ops.land_value_pct)
        accum_depr = 0.0
        bal_senior = self.debt_struct.senior_loan_amount
        bal_mezz   = self.debt_struct.mezzanine_loan_amount
        revpar_base = self.hotel_ops.base_revpar
        opex_base   = self.hotel_ops.fixed_opex_per_room_annual
        bfr_prev    = 0.0

        for year in range(1, self.holding_period + 1):
            if year <= self.monthly_detail_years:
                mdf = self._gen_monthly_year(
                    year, revpar_base, opex_base,
                    dep_val, accum_depr, bal_senior, bal_mezz, bfr_prev
                )
                monthly_dfs.append(mdf)
                last = mdf.iloc[-1]
                revpar_base  = last["RevPAR"] / last["Monthly_SF"]
                opex_base    = last["Fixed_OpEx_Per_Room_Base"]
                accum_depr   = last["Accumulated_Depreciation"]
                bal_senior   = last["Loan_Balance_Senior"]
                bal_mezz     = last["Loan_Balance_Mezzanine"]
                bfr_prev     = last["BFR"]
            else:
                esg = self.esg_integrator.get_financial_impacts(year - 1)
                revpar_base *= (1 + self.rng.normal(
                    self.market_ass.revpar_growth_mean,
                    self.market_ass.revpar_growth_std * esg["volatility_reduction_factor"]
                ) + esg["revpar_premium"])
                opex_base *= (1 + self.rng.normal(
                    self.market_ass.opex_inflation_mean,
                    self.market_ass.opex_inflation_std
                ))
                row, bal_senior, bal_mezz, accum_depr, bfr_prev = self._gen_annual_year(
                    year, revpar_base, opex_base,
                    dep_val, accum_depr, bal_senior, bal_mezz, bfr_prev, esg
                )
                annual_rows.append(row)

        AGG = {
            "RevPAR": "mean", "Occupancy": "mean",
            "Total_Revenue": "sum", "Total_OpEx": "sum",
            "Fixed_OpEx": "sum", "Variable_OpEx": "sum",
            "EBITDA": "sum", "EBITDA_Margin": "mean",
            "Depreciation": "sum", "Interest_Expense": "sum",
            "EBT": "sum", "Corporate_Tax": "sum", "Net_Income": "sum",
            "FFES_Reserve": "sum", "Major_CAPEX": "sum", "Total_CAPEX": "sum",
            "Delta_BFR": "sum", "CFADS": "sum",
            "Total_Debt_Service": "sum", "DSCR": "mean", "FCFE": "sum",
            "Loan_Balance_Senior": "last", "Loan_Balance_Mezzanine": "last",
            "Total_Debt_Outstanding": "last",
            "Accumulated_Depreciation": "last", "Gross_Book_Value": "last",
        }

        parts = []
        if monthly_dfs:
            parts.append(pd.concat(monthly_dfs).groupby("Year").agg(AGG))
        if annual_rows:
            parts.append(pd.DataFrame(annual_rows).set_index("Year"))

        full_df = pd.concat(parts).sort_index()
        full_df.index.name = "Year"

        esg_exit  = self.esg_integrator.get_financial_impacts(self.holding_period - 1)
        exit_cap  = self.market_ass.exit_cap_mean + esg_exit["cap_rate_compression_bps"] / 10_000
        t_noi     = full_df.loc[self.holding_period, "EBITDA"] - full_df.loc[self.holding_period, "FFES_Reserve"]
        tv_gross  = t_noi / exit_cap
        nbv       = full_df.loc[self.holding_period, "Gross_Book_Value"] - full_df.loc[self.holding_period, "Accumulated_Depreciation"]
        cap_tax   = max(0, (tv_gross - nbv) * self.market_ass.capital_gains_tax_rate)
        tv_net    = tv_gross - cap_tax
        debt_exit = full_df.loc[self.holding_period, "Total_Debt_Outstanding"]

        full_df.loc[self.holding_period, "Terminal_Value_Gross"] = tv_gross
        full_df.loc[self.holding_period, "Capital_Gains_Tax"]    = cap_tax
        full_df.loc[self.holding_period, "Terminal_Value_Net"]   = tv_net
        full_df.loc[self.holding_period, "Asset_Value"]          = tv_gross
        full_df.loc[self.holding_period, "FCFE_Terminal"]        = (
            full_df.loc[self.holding_period, "FCFE"] + tv_net - debt_exit
        )

        eq0 = self.hotel_ops.purchase_price - (
            self.debt_struct.senior_loan_amount + self.debt_struct.mezzanine_loan_amount
        )
        df_irr = pd.DataFrame(index=range(self.holding_period + 1))
        df_irr.loc[0, "FCFE"] = -eq0
        df_irr.loc[1:, "FCFE"] = full_df.loc[1:, "FCFE"].values
        df_irr.loc[self.holding_period, "FCFE"] = full_df.loc[self.holding_period, "FCFE_Terminal"]

        df_irr.loc[0, "FCF_Unlevered"] = -self.hotel_ops.purchase_price
        df_irr.loc[1:, "FCF_Unlevered"] = (
            full_df.loc[1:, "EBITDA"]
            - full_df.loc[1:, "Total_CAPEX"]
            - full_df.loc[1:, "Corporate_Tax"]
            - full_df.loc[1:, "Delta_BFR"]
        ).values
        df_irr.loc[self.holding_period, "FCF_Unlevered"] = tv_gross - cap_tax

        return full_df, df_irr

    def _gen_monthly_year(self, year, revpar_base, opex_base,
                          dep_val, accum_depr, bal_senior, bal_mezz, bfr_prev):
        idx = pd.date_range(
            start=f"{self.start_year + year - 1}-01-01", periods=12, freq="ME"
        )
        mdf = pd.DataFrame(index=idx)
        mdf["Year"] = year

        sf_raw = np.array(list(self.hotel_ops.monthly_seasonality_factors.values()), dtype=float)
        sf = sf_raw / (sf_raw.sum() / 12)

        esg = self.esg_integrator.get_financial_impacts(year - 1)
        revpar_eff = revpar_base * (
            1 + self.rng.normal(
                self.market_ass.revpar_growth_mean,
                self.market_ass.revpar_growth_std * esg["volatility_reduction_factor"]
            ) + esg["revpar_premium"]
        )
        opex_eff = opex_base * (
            1 + self.rng.normal(self.market_ass.opex_inflation_mean, self.market_ass.opex_inflation_std)
        )

        ann_depr  = dep_val / self.hotel_ops.asset_useful_life_years
        ann_princ = (self.debt_struct.senior_loan_amount / self.debt_struct.senior_amortization_years
                     if year <= self.debt_struct.senior_amortization_years else 0.0)

        for m in range(12):
            d  = idx[m]
            dm = calendar.monthrange(self.start_year + year - 1, m + 1)[1]

            rev_m   = revpar_eff * sf[m] * self.hotel_ops.rooms * dm
            occ_m   = np.clip(self.hotel_ops.base_occupancy * sf[m], 0.01, 0.99)

            var_opex = sum(
                rev_m * self.hotel_ops.rev_segmentation.get(seg, 0) * ratio
                for seg, ratio in self.hotel_ops.opex_ratios.items()
                if seg in self.hotel_ops.rev_segmentation
            ) * (1 - esg["opex_reduction"])
            fix_opex = (opex_eff / 12) * self.hotel_ops.rooms * (1 - esg["opex_reduction"])
            tot_opex = var_opex + fix_opex

            gop  = rev_m - tot_opex
            bmf  = rev_m * self.hotel_ops.management_fee_base_pct
            ebi  = gop - bmf
            thr  = rev_m * self.hotel_ops.management_fee_incentive_threshold_pct
            imf  = max(0, (ebi - thr) * self.hotel_ops.management_fee_incentive_pct)
            ebitda_m = ebi - imf

            depr_m   = ann_depr / 12
            accum_depr += depr_m

            ffe_m   = rev_m * self.hotel_ops.ffe_reserve_pct
            major_m = next(
                (amt for (y2, amt) in self.hotel_ops.renovation_capex if y2 == year and m == 5), 0
            )
            tot_capex_m = ffe_m + major_m

            intr_s_m = bal_senior * (self.debt_struct.senior_interest_rate / 12)
            prin_s_m = ann_princ / 12
            intr_z_m = bal_mezz * (self.debt_struct.mezzanine_interest_rate / 12)
            prin_z_m = bal_mezz if (year == self.debt_struct.senior_term_years and m == 11) else 0
            tot_ds_m = intr_s_m + prin_s_m + intr_z_m + prin_z_m
            bal_senior = max(0, bal_senior - prin_s_m)
            bal_mezz   = max(0, bal_mezz   - prin_z_m)

            ar       = (rev_m / dm) * self.hotel_ops.dso_days
            ap       = (tot_opex / dm) * self.hotel_ops.dpo_days
            inv      = (tot_opex * (self.hotel_ops.opex_ratios.get("F&B", 0)
                        + self.hotel_ops.opex_ratios.get("Utilities", 0)) / dm) * self.hotel_ops.dio_days
            cash_min = (tot_opex / dm) * self.hotel_ops.min_cash_days_opex
            bfr_m    = ar + inv + cash_min - ap
            delta_bfr = bfr_m - (bfr_prev if m == 0 else mdf.iloc[m - 1]["BFR"])

            int_exp_m = intr_s_m + intr_z_m
            ebt_m     = (ebitda_m - depr_m) - int_exp_m
            tax_m     = max(0, ebt_m) * self.market_ass.corporate_tax_rate
            cfads_m   = ebitda_m - tot_capex_m - delta_bfr
            dscr_m    = cfads_m / tot_ds_m if tot_ds_m > 0 else np.inf
            fcfe_m    = cfads_m - tot_ds_m - tax_m

            for k, v in {
                "RevPAR": revpar_eff * sf[m], "Occupancy": occ_m, "Monthly_SF": sf[m],
                "Total_Revenue": rev_m, "Variable_OpEx": var_opex,
                "Fixed_OpEx": fix_opex, "Total_OpEx": tot_opex,
                "Fixed_OpEx_Per_Room_Base": opex_eff,
                "EBITDA": ebitda_m, "EBITDA_Margin": ebitda_m / rev_m if rev_m else 0,
                "Depreciation": depr_m, "Accumulated_Depreciation": accum_depr,
                "Gross_Book_Value": dep_val,
                "FFES_Reserve": ffe_m, "Major_CAPEX": major_m, "Total_CAPEX": tot_capex_m,
                "Total_Debt_Service": tot_ds_m,
                "Loan_Balance_Senior": bal_senior, "Loan_Balance_Mezzanine": bal_mezz,
                "Total_Debt_Outstanding": bal_senior + bal_mezz,
                "BFR": bfr_m, "Delta_BFR": delta_bfr,
                "Interest_Expense": int_exp_m, "EBT": ebt_m,
                "Corporate_Tax": tax_m, "Net_Income": ebt_m - tax_m,
                "CFADS": cfads_m, "DSCR": dscr_m, "FCFE": fcfe_m,
            }.items():
                mdf.at[d, k] = v

        return mdf

    def _gen_annual_year(self, year, revpar_base, opex_base,
                         dep_val, accum_depr, bal_senior, bal_mezz, bfr_prev, esg):
        rev = revpar_base * self.hotel_ops.rooms * 365

        var_opex = sum(
            rev * self.hotel_ops.rev_segmentation.get(seg, 0) * ratio
            for seg, ratio in self.hotel_ops.opex_ratios.items()
            if seg in self.hotel_ops.rev_segmentation
        )
        fix_opex  = opex_base * self.hotel_ops.rooms
        tot_opex  = (var_opex + fix_opex) * (1 - esg["opex_reduction"])

        gop  = rev - tot_opex
        bmf  = rev * self.hotel_ops.management_fee_base_pct
        ebi  = gop - bmf
        thr  = rev * self.hotel_ops.management_fee_incentive_threshold_pct
        imf  = max(0, (ebi - thr) * self.hotel_ops.management_fee_incentive_pct)
        ebitda = ebi - imf

        ann_depr  = dep_val / self.hotel_ops.asset_useful_life_years
        accum_depr += ann_depr
        ffe   = rev * self.hotel_ops.ffe_reserve_pct
        major = next((amt for (y2, amt) in self.hotel_ops.renovation_capex if y2 == year), 0)
        capex = ffe + major

        ann_princ = (self.debt_struct.senior_loan_amount / self.debt_struct.senior_amortization_years
                     if year <= self.debt_struct.senior_amortization_years else 0.0)
        intr_s  = bal_senior * self.debt_struct.senior_interest_rate
        intr_z  = bal_mezz   * self.debt_struct.mezzanine_interest_rate
        prin_z  = self.debt_struct.mezzanine_loan_amount if year == self.debt_struct.senior_term_years else 0
        tot_ds  = ann_princ + intr_s + intr_z + prin_z
        bal_senior = max(0, bal_senior - ann_princ)
        bal_mezz   = max(0, bal_mezz   - prin_z)

        ebt  = (ebitda - ann_depr) - (intr_s + intr_z)
        tax  = max(0, ebt) * self.market_ass.corporate_tax_rate
        cfads = ebitda - capex
        dscr  = cfads / tot_ds if tot_ds > 0 else np.inf
        fcfe  = cfads - tot_ds - tax

        return {
            "Year": year,
            "RevPAR": revpar_base, "Occupancy": self.hotel_ops.base_occupancy,
            "Total_Revenue": rev, "Variable_OpEx": var_opex,
            "Fixed_OpEx": fix_opex, "Total_OpEx": tot_opex,
            "EBITDA": ebitda, "EBITDA_Margin": ebitda / rev if rev else 0,
            "Depreciation": ann_depr, "Accumulated_Depreciation": accum_depr,
            "Gross_Book_Value": dep_val,
            "FFES_Reserve": ffe, "Major_CAPEX": major, "Total_CAPEX": capex,
            "Total_Debt_Service": tot_ds,
            "Loan_Balance_Senior": bal_senior, "Loan_Balance_Mezzanine": bal_mezz,
            "Total_Debt_Outstanding": bal_senior + bal_mezz,
            "Interest_Expense": intr_s + intr_z,
            "EBT": ebt, "Corporate_Tax": tax, "Net_Income": ebt - tax,
            "Delta_BFR": 0.0, "BFR": bfr_prev,
            "CFADS": cfads, "DSCR": dscr, "FCFE": fcfe,
        }, bal_senior, bal_mezz, accum_depr, bfr_prev


# ================================================================
#  MONTE CARLO SIMULATOR
# ================================================================

class APEX_Simulator:
    FINANCE_RATE  = 0.06
    REINVEST_RATE = 0.08

    def __init__(self, modeler: APEX_Modeler):
        self.modeler = modeler

    def run_monte_carlo(self, n_simulations: int = 1_000) -> Dict[str, np.ndarray]:
        results = {k: [] for k in [
            "irr_levered", "mirr_levered", "payback_levered", "equity_multiple",
            "irr_unlevered", "mirr_unlevered", "dscr_min", "dscr_avg", "ltv_max"
        ]}

        for _ in range(n_simulations):
            try:
                full_df, df_irr = self.modeler.generate_annual_cash_flows()
            except Exception:
                continue

            cf_l = df_irr["FCFE"].values
            cf_u = df_irr["FCF_Unlevered"].values

            try:   irr_l = npf.irr(cf_l)
            except: irr_l = np.nan
            try:   irr_u = npf.irr(cf_u)
            except: irr_u = np.nan

            init_eq = -cf_l[0]
            em = np.sum(cf_l[cf_l > 0]) / init_eq if init_eq > 0 else np.nan
            prop_val = full_df["EBITDA"] / self.modeler.market_ass.exit_cap_mean
            ltv = (full_df["Total_Debt_Outstanding"] / prop_val).replace([np.inf, -np.inf], np.nan).max()

            results["irr_levered"].append(irr_l)
            results["mirr_levered"].append(self._mirr(cf_l))
            results["payback_levered"].append(self._payback(cf_l))
            results["equity_multiple"].append(em)
            results["irr_unlevered"].append(irr_u)
            results["mirr_unlevered"].append(self._mirr(cf_u))
            results["dscr_min"].append(full_df["DSCR"].replace(np.inf, np.nan).min())
            results["dscr_avg"].append(full_df["DSCR"].replace(np.inf, np.nan).mean())
            results["ltv_max"].append(ltv)

        return {k: np.array(v, dtype=float) for k, v in results.items()}

    def _mirr(self, cfs):
        cfs = np.asarray(cfs, dtype=float)
        n   = len(cfs)
        pv_neg = sum(cf / (1 + self.FINANCE_RATE) ** i for i, cf in enumerate(cfs) if cf < 0)
        fv_pos = sum(cf * (1 + self.REINVEST_RATE) ** (n - 1 - i) for i, cf in enumerate(cfs) if cf > 0)
        if pv_neg == 0 or n < 2:
            return np.nan
        return (fv_pos / -pv_neg) ** (1 / (n - 1)) - 1

    def _payback(self, cfs):
        cfs = np.asarray(cfs, dtype=float)
        cum = np.cumsum(cfs)
        if cfs[0] >= 0 or cum[-1] < 0:
            return np.inf
        for i in range(1, len(cum)):
            if cum[i] >= 0:
                return (i - 1) + (-cum[i - 1]) / cfs[i]
        return np.inf


# ================================================================
#  STRESS TESTER
# ================================================================

class APEX_StressTester:
    PREDEFINED_SCENARIOS = [
        ("Base Case",           0.00,  0.00,   0),
        ("Récession Modérée",  -0.15,  0.10,  50),
        ("Crise Sévère",       -0.25,  0.20, 100),
        ("Choc Énergétique",   -0.05,  0.30,  25),
        ("Reprise Forte",       0.10, -0.05, -25),
    ]

    def __init__(self, simulator: APEX_Simulator):
        self.simulator = simulator
        self.modeler   = simulator.modeler

    def run_scenario(self, name, revpar_mult, opex_mult, cap_bps, n_sims=500):
        m = self.modeler.market_ass
        orig = (m.revpar_growth_mean, m.opex_inflation_mean, m.exit_cap_mean)
        try:
            m.revpar_growth_mean  = orig[0] * (1 + revpar_mult)
            m.opex_inflation_mean = orig[1] * (1 + opex_mult)
            m.exit_cap_mean       = orig[2] + cap_bps / 10_000
            return self.simulator.run_monte_carlo(n_sims)
        finally:
            m.revpar_growth_mean, m.opex_inflation_mean, m.exit_cap_mean = orig

    def run_all_predefined(self, n_sims=500):
        return {
            name: self.run_scenario(name, rm, om, cb, n_sims)
            for name, rm, om, cb in self.PREDEFINED_SCENARIOS
        }


# ================================================================
#  CALCULATOR
# ================================================================

class APEX_Calculator:
    DSCR_COVENANT = 1.25

    def __init__(self, mc_results, stress_results, holding_period):
        self.mc = mc_results
        self.stress = stress_results
        self.hp = holding_period

    def calculate_safety_margin(self, hotel_ops, debt_struct, market_ass):
        var_ratio = sum(
            hotel_ops.opex_ratios.get(s, 0) * hotel_ops.rev_segmentation.get(s, 0)
            for s in hotel_ops.rev_segmentation
        )
        fix_opex_yr1 = hotel_ops.fixed_opex_per_room_annual * hotel_ops.rooms * (1 + market_ass.opex_inflation_mean)
        avg_ds       = (debt_struct.senior_loan_amount * debt_struct.senior_interest_rate
                        + debt_struct.mezzanine_loan_amount * debt_struct.mezzanine_interest_rate)
        avg_capex    = hotel_ops.base_revpar * hotel_ops.base_occupancy * hotel_ops.rooms * 365 * hotel_ops.ffe_reserve_pct
        target_cfads = avg_ds * self.DSCR_COVENANT

        def f(occ):
            rev = hotel_ops.base_revpar * occ * hotel_ops.rooms * 365
            return (rev - fix_opex_yr1 - rev * var_ratio) - avg_capex - target_cfads

        try:
            be_occ = brentq(f, 0.01, 0.99)
            margin = (hotel_ops.base_occupancy - be_occ) / hotel_ops.base_occupancy * 100
        except ValueError:
            be_occ, margin = np.nan, np.nan

        return {"breakeven_occupancy": be_occ, "safety_margin_pct": margin}

    def calculate_resilience_score(self):
        def median_irr(res):
            v = res["irr_levered"]
            v = v[~np.isnan(v) & ~np.isinf(v)]
            return np.median(v) if len(v) else np.nan

        base_irr  = median_irr(self.mc)
        worst_irr = min(median_irr(r) for r in self.stress.values())
        score = 100.0

        if base_irr != 0 and not np.isnan(base_irr):
            drop = (base_irr - worst_irr) / abs(base_irr)
            score -= min(40, max(0, drop * 40))

        worst_dscr5 = min(np.nanpercentile(r["dscr_min"], 5) for r in self.stress.values())
        if worst_dscr5 < 1.0:
            score -= 30
        elif worst_dscr5 < 1.25:
            score -= (1.25 - worst_dscr5) * 100

        worst_prob = max(np.mean(r["irr_levered"][~np.isnan(r["irr_levered"])] < 0)
                         for r in self.stress.values())
        score -= worst_prob * 30

        resilience = max(0, min(100, score))
        label = (
            "AAA — Résilience Exceptionnelle" if resilience >= 85 else
            "AA  — Résilience Forte"          if resilience >= 70 else
            "A   — Résilience Modérée"        if resilience >= 55 else
            "BBB — Résilience Faible"         if resilience >= 40 else
            "BB  — Risque Élevé"
        )
        return {"resilience_score": resilience, "resilience_label": label}


# ================================================================
#  REPORTER
# ================================================================

class APEX_Reporter:
    COLORS = {
        "primary": "#002F6C", "secondary": "#00A3E0", "success": "#78BE20",
        "warning": "#FFB81C", "danger":  "#C8102E",  "dark":    "#1C1C1C",
        "light":   "#F7F7F7", "gold":    "#D4AF37",
    }

    def __init__(self, hotel_name: str, holding_period: int = 10):
        self.hotel_name     = hotel_name
        self.holding_period = holding_period
        self.rng            = np.random.default_rng(42)

    def _safe(self, arr, lo=-0.5, hi=1.5):
        v = arr[np.isfinite(arr)]
        return v[(v >= lo) & (v <= hi)]

    def generate_dashboard(self, mc, stress, esg_integrator, modeler, safety, resilience):
        C  = self.COLORS
        HP = self.holding_period

        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                "Distribution TRI Levered", "Distribution Equity Multiple",
                "Évolution DSCR (Cas de Base)", "Distribution Payback Period",
                "TRI vs MIRR Levered", "Trajectoire ESG & Impacts",
                "Stress — TRI Levered Médian", "Stress — DSCR Min (5e pct)",
                "Indicateurs Clés (KPI)",
            ),
            specs=[
                [{"type": "histogram"}, {"type": "histogram"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "scatter"},   {"type": "bar"}],
                [{"type": "bar"},       {"type": "bar"},       {"type": "table"}],
            ],
            vertical_spacing=0.10, horizontal_spacing=0.08,
        )

        irr_v = self._safe(mc["irr_levered"]) * 100
        fig.add_trace(go.Histogram(x=irr_v, nbinsx=50, marker_color=C["primary"], showlegend=False), row=1, col=1)
        fig.add_vline(x=float(np.median(irr_v)), line_dash="dash", line_color=C["gold"], row=1, col=1)
        fig.update_xaxes(title_text="TRI (%)", row=1, col=1)

        em_v = self._safe(mc["equity_multiple"], 0, 10)
        fig.add_trace(go.Histogram(x=em_v, nbinsx=40, marker_color=C["secondary"], showlegend=False), row=1, col=2)
        fig.add_vline(x=float(np.median(em_v)), line_dash="dash", line_color=C["gold"], row=1, col=2)
        fig.update_xaxes(title_text="Equity Multiple (x)", row=1, col=2)

        base_df, _ = modeler.generate_annual_cash_flows()
        dscr_plot  = base_df["DSCR"].replace(np.inf, np.nan).clip(upper=5)
        fig.add_trace(go.Scatter(x=base_df.index, y=dscr_plot, mode="lines+markers",
                                 line=dict(color=C["success"], width=2), showlegend=False), row=1, col=3)
        fig.add_hline(y=1.25, line_dash="dot", line_color=C["danger"],
                      annotation_text="Covenant 1.25x", row=1, col=3)
        fig.update_xaxes(title_text="Année", row=1, col=3)
        fig.update_yaxes(title_text="DSCR (x)", row=1, col=3)

        pb_v = mc["payback_levered"][np.isfinite(mc["payback_levered"]) & (mc["payback_levered"] < HP * 2)]
        fig.add_trace(go.Histogram(x=pb_v, nbinsx=HP, marker_color=C["warning"], showlegend=False), row=2, col=1)
        fig.add_vline(x=float(np.median(pb_v)), line_dash="dash", line_color=C["dark"], row=2, col=1)
        fig.update_xaxes(title_text="Payback (années)", row=2, col=1)

        mask   = np.isfinite(mc["irr_levered"]) & np.isfinite(mc["mirr_levered"])
        irr_f  = mc["irr_levered"][mask] * 100
        mirr_f = mc["mirr_levered"][mask] * 100
        if len(irr_f) > 800:
            idx = self.rng.choice(len(irr_f), 800, replace=False)
            irr_f, mirr_f = irr_f[idx], mirr_f[idx]
        fig.add_trace(go.Scatter(x=irr_f, y=mirr_f, mode="markers",
                                 marker=dict(color=C["primary"], opacity=0.4, size=4), showlegend=False), row=2, col=2)
        lim = [float(irr_f.min()), float(irr_f.max())]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                 line=dict(dash="dash", color="gray"), showlegend=False), row=2, col=2)
        fig.update_xaxes(title_text="TRI (%)", row=2, col=2)
        fig.update_yaxes(title_text="MIRR (%)", row=2, col=2)

        years  = list(range(1, HP + 1))
        scores = [esg_integrator.get_score(i) for i in range(HP)]
        prems  = [esg_integrator.get_financial_impacts(i)["revpar_premium"] * 100 for i in range(HP)]
        fig.add_trace(go.Bar(x=years, y=scores, marker_color=C["secondary"], showlegend=False), row=2, col=3)
        fig.add_trace(go.Scatter(x=years, y=prems, mode="lines+markers",
                                 line=dict(color=C["success"]), showlegend=False), row=2, col=3)
        fig.update_xaxes(title_text="Année", row=2, col=3)
        fig.update_yaxes(title_text="Score / Impact (%)", row=2, col=3)

        base_irr_med = float(np.median(irr_v))
        s_names  = list(stress.keys())
        s_irrs   = [float(np.median(self._safe(stress[n]["irr_levered"]) * 100)) for n in s_names]
        s_colors = [C["success"] if v >= base_irr_med else C["danger"] for v in s_irrs]
        fig.add_trace(go.Bar(x=s_names, y=s_irrs, marker_color=s_colors, showlegend=False), row=3, col=1)
        fig.add_hline(y=base_irr_med, line_dash="dash", line_color=C["dark"], row=3, col=1)
        fig.update_yaxes(title_text="TRI Médian (%)", row=3, col=1)

        s_dscrs  = [float(np.nanpercentile(stress[n]["dscr_min"], 5)) for n in s_names]
        s_colors2 = [C["success"] if v >= 1.25 else C["danger"] for v in s_dscrs]
        fig.add_trace(go.Bar(x=s_names, y=s_dscrs, marker_color=s_colors2, showlegend=False), row=3, col=2)
        fig.add_hline(y=1.25, line_dash="dot", line_color=C["danger"], row=3, col=2)
        fig.update_yaxes(title_text="DSCR Min — 5e Pct (x)", row=3, col=2)

        sm_pct = safety.get("safety_margin_pct", float("nan"))
        be_occ = safety.get("breakeven_occupancy", float("nan"))
        rs     = resilience.get("resilience_score", float("nan"))
        rs_lbl = resilience.get("resilience_label", "N/A")

        kpi_rows = [
            ["TRI Levered (Médiane)",    f"{np.median(irr_v):.1f}%",                       "≥ 15%"],
            ["Equity Multiple",          f"{np.median(em_v):.2f}x",                        "≥ 2.0x"],
            ["Payback (Médiane)",        f"{np.median(pb_v):.1f} ans",                     "≤ 7 ans"],
            ["DSCR Min — 5e Pct",        f"{np.nanpercentile(mc['dscr_min'], 5):.2f}x",    "≥ 1.25x"],
            ["LTV Max — 95e Pct",        f"{np.nanpercentile(mc['ltv_max'], 95)*100:.1f}%","≤ 65%"],
            ["Safety Margin",            f"{sm_pct:.1f}%" if np.isfinite(sm_pct) else "N/A", "≥ 20%"],
            ["Breakeven Occupancy",      f"{be_occ*100:.1f}%" if np.isfinite(be_occ) else "N/A", ""],
            ["Score de Résilience",      f"{rs:.0f}/100 {rs_lbl}" if np.isfinite(rs) else "N/A", "≥ 80/100"],
        ]

        fig.add_trace(go.Table(
            header=dict(values=["<b>Indicateur</b>", "<b>Valeur</b>", "<b>Benchmark</b>"],
                        fill_color=C["primary"], font=dict(color="white", size=10), align="left"),
            cells=dict(values=list(zip(*kpi_rows)),
                       fill_color=[C["light"], "white", "white"],
                       align="left", font=dict(size=9))
        ), row=3, col=3)

        fig.update_layout(
            title=dict(text=f"<b>APEX DCF++™</b> — {self.hotel_name}",
                       font=dict(size=18, color=C["primary"])),
            height=1100, showlegend=False,
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Arial", size=10, color=C["dark"]),
        )
        return fig


# ================================================================
#  ORCHESTRATEUR PRINCIPAL
# ================================================================

class APEX_DCF_Plus:
    """Point d'entrée unique — instanciation, calcul et rapport."""

    def __init__(self, hotel_ops, market_ass, debt_struct, esg_traj,
                 holding_period=10, monthly_detail_years=3):
        self.hotel_ops   = hotel_ops
        self.market_ass  = market_ass
        self.debt_struct = debt_struct
        self.esg_traj    = esg_traj
        self.hp          = holding_period

        self.esg_integrator = APEX_ESG_Integrator(esg_traj)
        self.modeler  = APEX_Modeler(
            hotel_ops, market_ass, debt_struct,
            self.esg_integrator, holding_period, monthly_detail_years
        )
        self.simulator     = APEX_Simulator(self.modeler)
        self.stress_tester = APEX_StressTester(self.simulator)
        self.reporter      = APEX_Reporter(hotel_ops.hotel_name, holding_period)

        self.mc_results  = None
        self.stress_results = {}
        self.safety      = {}
        self.resilience  = {}

    def run(self, n_mc=1_000, n_stress=500):
        self.mc_results     = self.simulator.run_monte_carlo(n_mc)
        self.stress_results = self.stress_tester.run_all_predefined(n_stress)

        calc = APEX_Calculator(self.mc_results, self.stress_results, self.hp)
        self.safety     = calc.calculate_safety_margin(self.hotel_ops, self.debt_struct, self.market_ass)
        self.resilience = calc.calculate_resilience_score()
        return self

    def dashboard(self):
        if self.mc_results is None:
            raise RuntimeError("Lancez .run() avant .dashboard()")
        return self.reporter.generate_dashboard(
            self.mc_results, self.stress_results,
            self.esg_integrator, self.modeler,
            self.safety, self.resilience
        )

    def esg_summary(self):
        return self.esg_integrator.get_rating_summary()

    def base_case_financials(self):
        df, _ = self.modeler.generate_annual_cash_flows()
        cols = ["Total_Revenue", "Total_OpEx", "EBITDA", "EBITDA_Margin",
                "Total_CAPEX", "CFADS", "Total_Debt_Service", "DSCR",
                "Corporate_Tax", "FCFE", "Total_Debt_Outstanding"]
        return df[[c for c in cols if c in df.columns]]
