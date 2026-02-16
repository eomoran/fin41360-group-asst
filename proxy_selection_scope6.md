### Practical ETF proxies for Fama–French 5 factors (Scope 6)

This note records the chosen **investable proxies** (one per Fama–French factor) and why each is preferred to common alternatives. All returns should be treated as **total returns (reinvested distributions)** via adjusted-close or provider total-return indices.

---

## 1. Market factor \(R_m - R_f\) → **VTI** (Vanguard Total Stock Market ETF)

**Rationale**

- Fama–French “market” is a **value‑weighted broad US market** (all CRSP stocks), not just large caps.
- **VTI** tracks the CRSP US Total Market Index → coverage is close to the FF market: large, mid, and small caps.
- **SPY** (and IVV/VOO) track the S&P 500 → large‑cap only, under‑representing the small/mid‑cap part of the FF market portfolio.
- **MSCI USA** ETFs (e.g. EUSA/IUSA) are also broad, but VTI is typically more liquid, widely used, and aligned with CRSP (same data family as used in the FF construction).

**Conclusion:** Use **VTI** as the preferred market proxy. SPY or an MSCI USA ETF are acceptable alternatives if explicitly described as large‑cap or MSCI-based proxies.

---

## 2. Size factor \(SMB\) – “small minus big” → **IJR** (iShares Core S&P Small‑Cap ETF)

**Rationale**

- **IJR** tracks the **S&P Small‑Cap 600**: investable small‑cap universe with **profitability and liquidity screens**.
- This gives strong **positive SMB exposure** while avoiding some junk/micro‑cap names that can dominate the Russell 2000.
- **IWM** (Russell 2000) is a standard small‑cap ETF and a fine alternative, but has more micro‑caps and less screening → noisier, more extreme small‑cap risk.
- **VB/SCHA** are also viable small‑cap funds; IJR is very common in factor work and has a clear, well‑documented index.

**Conclusion:** Use **IJR** as the main SMB proxy; mention **IWM** as a natural alternative if a broader/more “aggressive” small‑cap exposure is desired.

---

## 3. Value factor \(HML\) – “high minus low” → **IWD** (iShares Russell 1000 Value ETF)

**Rationale**

- **IWD** holds the **value half of the Russell 1000**: large and mid‑cap value stocks.
- Broader than **IVE** (S&P 500 Value), which is limited to the S&P 500; broader coverage better mirrors the diversified value leg in FF.
- **IUSV** (Core S&P US Value) and **VTV** (Vanguard Value) are similar value ETFs; IWD is very liquid and widely referenced.
- **VBR** (small‑cap value) has strong value exposure but also a **strong SMB tilt**; for the HML “slot” we prefer a **purer value** proxy rather than a size+value bundle.
- **RPV** (S&P 500 Pure Value) has very strong value tilts but is more concentrated and volatile, which may be less realistic for an average portfolio manager.

**Conclusion:** Use **IWD** as the primary HML proxy. It gives a diversified large/mid‑cap value tilt with good liquidity and a long history.

---

## 4. Profitability factor \(RMW\) – “robust minus weak” → **QUAL** (iShares MSCI USA Quality Factor ETF)

**Rationale**

- There is no pure long‑short RMW ETF, but **quality/profitability** funds are standard stand‑ins.
- **QUAL** tracks an MSCI USA Quality index focusing on high ROE, stable earnings, and low leverage → closely aligned with **robust profitability**.
- Widely used in both practitioner and academic factor discussions; transparent methodology and strong positive RMW loading in factor regressions.
- **SPHQ** (S&P 500 Quality) is similar; QUAL is slightly more standard in research and has broad US coverage.
- **JQUA** and other quality ETFs are alternatives; all share the key property of tilting toward profitable, stable firms.

**Conclusion:** Use **QUAL** as the preferred RMW proxy, with the caveat that it mixes in broader “quality” dimensions as well as profitability.

---

## 5. Investment factor \(CMA\) – “conservative minus aggressive” → **USMV** (iShares MSCI USA Minimum Volatility ETF)

**Rationale**

- CMA targets firms with **conservative vs aggressive investment** (asset growth). There is no dedicated CMA ETF.
- **USMV** implements a **minimum‑volatility** portfolio over US equities. Such portfolios typically tilt toward **stable, mature, low‑growth, conservatively investing firms**, which correspond to the “conservative” side of CMA.
- **SPLV** (S&P 500 Low Volatility) is a viable alternative; USMV’s MSCI optimization tends to be more diversified across sectors.
- **VIG/HDV** (dividend ETFs) also lean toward conservative, mature firms, but mix in value and profitability tilts more strongly and are less directly interpretable as an investment‑intensity screen.
- All of these are long‑only, highly investable, low‑cost funds, appropriate for a typical portfolio manager.

**Conclusion:** Use **USMV** as the practical CMA proxy, clearly noting in the report that it is a conservative‑investment / low‑volatility stand‑in, not a precise replica of the long‑short CMA factor.

---

## Summary table

| FF factor | Proxy ETF | Main reasons vs alternatives |
|-----------|-----------|------------------------------|
| **Market (Rm−Rf)** | **VTI** | Broad US market (CRSP total market), closer to FF market than SPY; more comprehensive than S&P 500; highly liquid. |
| **SMB (size)** | **IJR** | Small‑cap (S&P 600) with profitability/liquidity screens; cleaner size exposure than IWM; standard in practice. |
| **HML (value)** | **IWD** | Russell 1000 Value (large/mid‑cap value); diversified, liquid; less mixed with size than small‑cap value funds like VBR. |
| **RMW (profitability)** | **QUAL** | MSCI USA Quality; tilts toward high‑profitability, stable firms; widely used quality/profitability proxy. |
| **CMA (investment)** | **USMV** | Minimum‑volatility US equity; tilts toward conservative, low‑growth firms; practical stand‑in for conservative investment. |

