"""
APEX DCF++™ — Interface Streamlit
Déployable sur Streamlit Community Cloud via GitHub.

Structure du dépôt :
  apex_model.py        ← logique métier (partagée avec Colab)
  app.py               ← ce fichier
  requirements.txt     ← dépendances
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.io as pio

from apex_model import (
    HotelOperatingInputs, MarketAssumptions, DebtStructure,
    ESGTrajectory, APEX_DCF_Plus
)

# ── Configuration de la page ──────────────────────────────────────
st.set_page_config(
    page_title="APEX DCF++™",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ──────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { min-width: 340px; max-width: 380px; }
  .kpi-card {
    background: #f0f4ff;
    border: 1px solid #c0cfe8;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
  }
  .kpi-label { font-size: 12px; color: #555; margin-bottom: 4px; }
  .kpi-value { font-size: 22px; font-weight: 700; }
  .kpi-good  { color: #1a7a3f; }
  .kpi-warn  { color: #b35c00; }
  .kpi-bad   { color: #c0392b; }
  .section-header {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: #002F6C;
    border-bottom: 2px solid #002F6C;
    padding-bottom: 3px; margin: 12px 0 8px 0;
  }
  div[data-testid="metric-container"] { background: #f8faff; border-radius: 8px; padding: 8px; }
</style>
""", unsafe_allow_html=True)


# ================================================================
#  SIDEBAR — Tous les inputs
# ================================================================

with st.sidebar:
    st.markdown("## APEX DCF++™")
    st.caption("Modèle institutionnel d'évaluation hôtelière")
    st.divider()

    # ── Actif hôtelier ───────────────────────────────────────────
    st.markdown('<div class="section-header">Actif hôtelier</div>', unsafe_allow_html=True)

    hotel_name  = st.text_input("Nom de l'hôtel", value="Azure Beach Resort")
    rooms       = st.number_input("Chambres", min_value=10, max_value=3000, value=220, step=10)
    price       = st.number_input("Prix d'acquisition (€)", min_value=1_000_000,
                                   max_value=2_000_000_000, value=75_000_000, step=500_000,
                                   format="%d")
    revpar      = st.number_input("RevPAR de base (€)", min_value=20.0, max_value=1000.0,
                                   value=165.0, step=5.0)
    occ         = st.slider("Occupation de base (%)", min_value=20, max_value=99,
                             value=74, step=1) / 100
    fixed_opex  = st.number_input("OpEx fixes / chambre / an (€)", min_value=0,
                                   max_value=100_000, value=15_000, step=500)

    col1, col2 = st.columns(2)
    with col1:
        land_pct   = st.slider("Part terrain (%)", 0, 50, 20, 1) / 100
        ffe_pct    = st.slider("Réserve FF&E (%)", 1, 10, 4, 1) / 100
    with col2:
        useful_life = st.number_input("Durée amort. (ans)", 10, 50, 25, 1)

    st.markdown('<div class="section-header">CAPEX de rénovation</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    cap1_yr  = c1.number_input("Réno 1 — Année",     1, 20, 3, 1)
    cap1_amt = c2.number_input("Réno 1 — Montant (€)", 0, 50_000_000, 2_000_000, 100_000, format="%d")
    c3, c4 = st.columns(2)
    cap2_yr  = c3.number_input("Réno 2 — Année",     1, 20, 7, 1)
    cap2_amt = c4.number_input("Réno 2 — Montant (€)", 0, 50_000_000, 1_500_000, 100_000, format="%d")

    # ── Structure financière ──────────────────────────────────────
    st.markdown('<div class="section-header">Structure financière</div>', unsafe_allow_html=True)

    senior_amt   = st.number_input("Dette senior (€)", 0, 1_000_000_000, 45_000_000, 500_000, format="%d")
    senior_rate  = st.slider("Taux senior (%)", 0.5, 15.0, 4.2, 0.1) / 100
    senior_term  = st.number_input("Maturité senior (ans)", 1, 30, 7, 1)
    senior_amort = st.number_input("Amortissement senior (ans)", 5, 40, 25, 1)
    mezz_amt     = st.number_input("Mezzanine (€)", 0, 500_000_000, 10_000_000, 500_000, format="%d")
    mezz_rate    = st.slider("Taux mezzanine (%)", 0.0, 25.0, 8.5, 0.5) / 100

    ltv = (senior_amt + mezz_amt) / price * 100 if price > 0 else 0
    ltv_color = "green" if ltv <= 65 else ("orange" if ltv <= 75 else "red")
    st.markdown(
        f"**LTV totale : :{ltv_color}[{ltv:.1f}%]** "
        f"(senior {senior_amt/price*100:.1f}% + mezz {mezz_amt/price*100:.1f}%)"
    )

    # ── Marché & fiscalité ────────────────────────────────────────
    st.markdown('<div class="section-header">Marché & fiscalité</div>', unsafe_allow_html=True)

    ca, cb = st.columns(2)
    revpar_g = ca.slider("RevPAR μ (%/an)", -5.0, 10.0, 2.5, 0.5) / 100
    revpar_s = cb.slider("RevPAR σ (%)", 0.0, 10.0, 1.5, 0.5) / 100
    cc, cd = st.columns(2)
    opex_g   = cc.slider("OpEx μ (%/an)", 0.0, 10.0, 3.0, 0.5) / 100
    opex_s   = cd.slider("OpEx σ (%)", 0.0, 5.0, 1.0, 0.5) / 100
    ce, cf_ = st.columns(2)
    cap_rate = ce.slider("Cap rate μ (%)", 2.0, 12.0, 5.5, 0.25) / 100
    cap_std  = cf_.slider("Cap rate σ (%)", 0.0, 3.0, 0.5, 0.1) / 100
    cg, ch = st.columns(2)
    is_rate  = cg.slider("IS France (%)", 10, 35, 25, 1) / 100
    cg_rate  = ch.slider("Taxe plus-value (%)", 0, 35, 25, 1) / 100

    # ── Période & ESG ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Période & ESG</div>', unsafe_allow_html=True)

    holding      = st.number_input("Durée de détention (ans)", 3, 20, 10, 1)
    monthly_det  = st.number_input("Années détail mensuel", 1, 5, 3, 1)

    st.caption("Trajectoire ESG (score /100 par année) — GRESB")
    esg_defaults = [65, 70, 75, 78, 80, 82, 83, 84, 85, 85] + [85] * 10
    esg_scores = []
    for i in range(holding):
        default = esg_defaults[i] if i < len(esg_defaults) else 85
        val = st.slider(f"An {i+1}", 0, 100, default, 1, key=f"esg_{i}")
        esg_scores.append(val)

    # ── Simulation ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Simulation</div>', unsafe_allow_html=True)

    n_mc     = st.select_slider("Simulations Monte Carlo",
                                 options=[500, 1000, 2000, 5000], value=1000)
    n_stress = st.select_slider("Simulations / stress test",
                                 options=[200, 500, 1000], value=500)

    st.divider()
    run_btn = st.button("Lancer APEX DCF++", type="primary", use_container_width=True)


# ================================================================
#  ZONE PRINCIPALE
# ================================================================

st.title(f"APEX DCF++™ — {hotel_name}")
st.caption("Modèle institutionnel · GRESB | TCFD | SFDR | EU Taxonomy | USALI")

if not run_btn:
    st.info(
        "Paramétrez votre actif dans la sidebar, puis cliquez sur **Lancer APEX DCF++** "
        "pour démarrer l'analyse Monte Carlo et les stress tests."
    )
    st.stop()

# ── Lancement de l'analyse ────────────────────────────────────────
capex_list = []
if cap1_amt > 0:
    capex_list.append((int(cap1_yr), float(cap1_amt)))
if cap2_amt > 0:
    capex_list.append((int(cap2_yr), float(cap2_amt)))

with st.spinner(f"Monte Carlo — {n_mc:,} simulations en cours…"):
    try:
        apex = APEX_DCF_Plus(
            hotel_ops=HotelOperatingInputs(
                hotel_name=hotel_name,
                rooms=int(rooms),
                purchase_price=float(price),
                base_revpar=float(revpar),
                base_occupancy=float(occ),
                land_value_pct=float(land_pct),
                asset_useful_life_years=int(useful_life),
                fixed_opex_per_room_annual=float(fixed_opex),
                ffe_reserve_pct=float(ffe_pct),
                renovation_capex=capex_list,
            ),
            market_ass=MarketAssumptions(
                revpar_growth_mean=float(revpar_g),
                revpar_growth_std=float(revpar_s),
                opex_inflation_mean=float(opex_g),
                opex_inflation_std=float(opex_s),
                exit_cap_mean=float(cap_rate),
                exit_cap_std=float(cap_std),
                corporate_tax_rate=float(is_rate),
                capital_gains_tax_rate=float(cg_rate),
            ),
            debt_struct=DebtStructure(
                senior_loan_amount=float(senior_amt),
                senior_interest_rate=float(senior_rate),
                senior_term_years=int(senior_term),
                senior_amortization_years=int(senior_amort),
                mezzanine_loan_amount=float(mezz_amt),
                mezzanine_interest_rate=float(mezz_rate) if mezz_amt > 0 else 0.0,
            ),
            esg_traj=ESGTrajectory(esg_scores_per_year=esg_scores),
            holding_period=int(holding),
            monthly_detail_years=int(monthly_det),
        ).run(n_mc=int(n_mc), n_stress=int(n_stress))

    except (AssertionError, ValueError) as e:
        st.error(f"Paramètre invalide : {e}")
        st.stop()
    except Exception as e:
        st.error(f"Erreur inattendue : {type(e).__name__}: {e}")
        st.stop()

st.success("Analyse terminée")

# ── KPIs ──────────────────────────────────────────────────────────
mc  = apex.mc_results
sm  = apex.safety
res = apex.resilience

irr_v = mc["irr_levered"][np.isfinite(mc["irr_levered"])] * 100
em_v  = mc["equity_multiple"][np.isfinite(mc["equity_multiple"])]
pb_v  = mc["payback_levered"][np.isfinite(mc["payback_levered"])]

def kpi_color(val, good, warn):
    if val >= good: return "normal"
    if val >= warn: return "off"
    return "inverse"

st.subheader("Métriques clés")
k1, k2, k3, k4, k5, k6, k7 = st.columns(7)

irr_med  = float(np.median(irr_v))
em_med   = float(np.median(em_v))
pb_med   = float(np.median(pb_v))
dscr5    = float(np.nanpercentile(mc["dscr_min"], 5))
ltv95    = float(np.nanpercentile(mc["ltv_max"], 95)) * 100
sm_pct   = sm.get("safety_margin_pct", float("nan"))
rs       = res.get("resilience_score", float("nan"))

k1.metric("TRI Levered (médiane)", f"{irr_med:.1f}%",
          f"[{np.percentile(irr_v,5):.1f}%–{np.percentile(irr_v,95):.1f}%]")
k2.metric("Equity Multiple",  f"{em_med:.2f}x")
k3.metric("Payback (médiane)", f"{pb_med:.1f} ans")
k4.metric("DSCR Min — 5e Pct", f"{dscr5:.2f}x",
          delta="≥1.25x requis", delta_color="off")
k5.metric("LTV Max — 95e Pct", f"{ltv95:.1f}%",
          delta="≤65% requis", delta_color="off")
k6.metric("Safety Margin",
          f"{sm_pct:.1f}%" if np.isfinite(sm_pct) else "N/A")
k7.metric("Score Résilience",
          f"{rs:.0f}/100" if np.isfinite(rs) else "N/A",
          delta=res.get("resilience_label", ""), delta_color="off")

# ── Tabs : Dashboard | Financials | ESG ──────────────────────────
tab1, tab2, tab3 = st.tabs(["Dashboard", "Tableau financier", "Résumé ESG"])

with tab1:
    st.subheader("Dashboard institutionnel")
    fig = apex.dashboard()
    st.plotly_chart(fig, use_container_width=True)

    # Export HTML
    html_bytes = pio.to_html(fig, full_html=True, include_plotlyjs="cdn").encode("utf-8")
    st.download_button(
        label="Télécharger le rapport HTML",
        data=html_bytes,
        file_name=f"APEX_{hotel_name.replace(' ','_')}.html",
        mime="text/html",
    )

with tab2:
    st.subheader("Cas de base — tableau annuel")
    df_fin = apex.base_case_financials()

    fmt = {
        "Total_Revenue":         "{:,.0f} €",
        "Total_OpEx":            "{:,.0f} €",
        "EBITDA":                "{:,.0f} €",
        "EBITDA_Margin":         "{:.1%}",
        "Total_CAPEX":           "{:,.0f} €",
        "CFADS":                 "{:,.0f} €",
        "Total_Debt_Service":    "{:,.0f} €",
        "DSCR":                  "{:.2f}x",
        "Corporate_Tax":         "{:,.0f} €",
        "FCFE":                  "{:,.0f} €",
        "Total_Debt_Outstanding":"{:,.0f} €",
    }
    st.dataframe(
        df_fin.style.format({k: v for k, v in fmt.items() if k in df_fin.columns}),
        use_container_width=True,
        height=420,
    )

    csv = df_fin.to_csv().encode("utf-8")
    st.download_button("Exporter CSV", csv,
                       f"APEX_financials_{hotel_name.replace(' ','_')}.csv",
                       "text/csv")

with tab3:
    st.subheader("Trajectoire ESG & impacts financiers")
    st.dataframe(apex.esg_summary(), use_container_width=True)
    st.caption(
        "Seuils GRESB : ≥85 Leader (AAA) · ≥75 Avancé (AA) · ≥65 Engagé (A) · <65 Standard"
    )
