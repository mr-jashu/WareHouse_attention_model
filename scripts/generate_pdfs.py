"""Generate the 5 formal project PDFs (reportlab platypus)."""
import os
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, Image, HRFlowable, ListFlowable,
                                ListItem, KeepTogether)
from reportlab.lib.colors import HexColor

ROOT = Path("/home/jaswanth/Documents/Ai_Prof/warehouse-attention")
OUT = ROOT / "documents_pdf"
OUT.mkdir(exist_ok=True)

NAVY = HexColor("#1a3a5c")
ACCENT = HexColor("#2a7ab5")
LIGHT = HexColor("#eaf2f9")
GREY = HexColor("#555555")

def styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("DocTitle", parent=ss["Title"], fontSize=22, leading=26,
                          textColor=NAVY, spaceAfter=6, alignment=TA_CENTER))
    ss.add(ParagraphStyle("DocSub", parent=ss["Normal"], fontSize=12, leading=16,
                          textColor=GREY, alignment=TA_CENTER, spaceAfter=2))
    ss.add(ParagraphStyle("H1", parent=ss["Heading1"], fontSize=14, leading=18,
                          textColor=NAVY, spaceBefore=14, spaceAfter=6,
                          borderPadding=(0, 0, 4, 0)))
    ss.add(ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, leading=15,
                          textColor=ACCENT, spaceBefore=10, spaceAfter=4))
    ss.add(ParagraphStyle("Body", parent=ss["Normal"], fontSize=9.5, leading=14,
                          alignment=TA_JUSTIFY, spaceAfter=5))
    ss.add(ParagraphStyle("Bul", parent=ss["Normal"], fontSize=9.5, leading=14,
                          leftIndent=18, bulletIndent=6, spaceAfter=3))
    ss.add(ParagraphStyle("CodeB", parent=ss["Code"], fontSize=8, leading=11,
                          backColor=HexColor("#f4f4f4"), borderPadding=6, spaceAfter=6))
    ss.add(ParagraphStyle("Cap", parent=ss["Normal"], fontSize=8, leading=11,
                          textColor=GREY, alignment=TA_CENTER, spaceAfter=8))
    ss.add(ParagraphStyle("Cell", parent=ss["Normal"], fontSize=8, leading=11))
    ss.add(ParagraphStyle("CellH", parent=ss["Normal"], fontSize=8, leading=11,
                          textColor=colors.white, fontName="Helvetica-Bold"))
    ss.add(ParagraphStyle("Foot", parent=ss["Normal"], fontSize=7.5, leading=10,
                          textColor=GREY, alignment=TA_CENTER))
    return ss

SS = styles()

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawCentredString(A4[0]/2, 1.5*cm,
        f"{doc.title}  |  Warehouse Demand Attention  |  PEDDADA  |  p. {doc.page}")
    canvas.restoreState()

def title_block(story, title, subtitle, meta_lines):
    story.append(Spacer(1, 2.2*cm))
    story.append(Paragraph(title, SS["DocTitle"]))
    story.append(Paragraph(subtitle, SS["DocSub"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="60%", thickness=1.2, color=ACCENT,
                            spaceAfter=10, spaceBefore=6, hAlign="CENTER"))
    for m in meta_lines:
        story.append(Paragraph(m, SS["DocSub"]))
    story.append(Spacer(1, 0.6*cm))

def h1(story, t): story.append(Paragraph(t, SS["H1"]))
def h2(story, t): story.append(Paragraph(t, SS["H2"]))
def p(story, t): story.append(Paragraph(t, SS["Body"]))
def bullets(story, items):
    for it in items:
        story.append(Paragraph(f"&bull;&nbsp;&nbsp;{it}", SS["Bul"]))
def code(story, t):
    story.append(Paragraph(f"<font face='Courier'>{t}</font>", SS["CodeB"]))

def table(story, header, rows, widths=None, caption=None):
    data = [[Paragraph(h, SS["CellH"]) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), SS["Cell"]) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("BACKGROUND", (0,1), (-1,-1), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, HexColor("#f5f8fc")]),
        ("GRID", (0,0), (-1,-1), 0.5, HexColor("#b9c9da")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(t)
    if caption:
        story.append(Paragraph(caption, SS["Cap"]))

def fig(story, relpath, width_cm, caption):
    fp = ROOT / relpath
    if fp.exists():
        story.append(Image(str(fp), width=width_cm*cm, height=width_cm*0.45*cm,
                           kind="proportional"))
        story.append(Paragraph(caption, SS["Cap"]))
    else:
        story.append(Paragraph(f"<i>[Figure missing: {relpath}]</i>", SS["Cap"]))

def build(path, title, story):
    doc = SimpleDocTemplate(str(path), pagesize=A4, topMargin=2*cm, bottomMargin=2.2*cm,
                            leftMargin=2.2*cm, rightMargin=2.2*cm, title=title)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", path, f"({path.stat().st_size/1024:.0f} KB)")

META = ["Author: PEDDADA", "Project: From First Principles: Warehouse Demand Attention Model",
        "Date: 01 October 2026 &nbsp;|&nbsp; Python 3.12, torch CPU, numpy, matplotlib, pytest",
        "Data seed 1234, shift seed 4321, model seeds 0-4 &nbsp;|&nbsp; Status: complete (18/18 items)"]

# ---------------------------------------------------------------- 1. Phase 0
def doc_phase0():
    s = []
    title_block(s, "Phase 0 Design Document",
                "Design Before Code &mdash; Locked Pre-registration",
                META + ["Locked: 30 Sep 2026, before any model/experiment code (git commit evidence)"])
    p(s, "This document is the frozen Phase 0 design for the warehouse-demand attention prototype. "
         "It was committed to git <b>before</b> <font face='Courier'>attention.py</font> or any experiment existed. "
         "Hypotheses H1&ndash;H5 and verification expectation V1 (Section 7) are never edited afterwards; "
         "later corrections go only in the Amendments Log (Section 9). The reviewer should be able to trace "
         "<b>Phase-0 prediction &rarr; implementation &rarr; result &rarr; explanation</b>.")
    h1(s, "1. Mathematical Problem Formulation")
    p(s, "Let <b>y<sub>t</sub> &ge; 0</b> be warehouse orders received in hour <b>t</b>. One-step-ahead forecasting:")
    code(s, "Input&nbsp;&nbsp;: W(t) = (y(t-23) ... y(t)) &nbsp; 24 hourly values + calendar features<br/>"
            "Output : y_hat(t+1) = f_theta(W(t)) &nbsp;&nbsp; next-hour demand<br/>"
            "Goal&nbsp;&nbsp;&nbsp; : theta* = argmin E[ L(f_theta(W), y(t+1)) ]")
    p(s, "Per-token features (T=24 tokens, one per hour i): <b>x<sub>i</sub> in R<sup>5</sup></b> = "
         "z-scored demand z<sub>i</sub>=(y<sub>i</sub>&minus;&mu;<sub>train</sub>)/&sigma;<sub>train</sub> (&mu;,&sigma; from "
         "training split only) + sin/cos(2&pi;&middot;hour/24) + sin/cos(2&pi;&middot;dow/7). "
         "Calendar is known in advance so it is not leakage; without a position signal attention is "
         "permutation-invariant and cannot separate &ldquo;yesterday same hour&rdquo; from &ldquo;one hour ago&rdquo;.")
    code(s, "E = X W_e + b_e + P &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; W_e in R(5x16), P in R(24x16) learned<br/>"
            "Q = E W_Q, K = E W_K, V = E W_V &nbsp; d_model = d_k = d_v = 16<br/>"
            "S = Q K^T / sqrt(d_k) &nbsp; (B,24,24) &rarr; A = softmax(S) &rarr; Y = A V &nbsp; (B,24,16)<br/>"
            "h = concat(Y[:,-1,:], E[:,-1,:]) (B,32) &rarr; MLP(32&rarr;32&rarr;1, ReLU) &rarr; z-score")
    p(s, "Single head, single layer, no causal mask (all 24 tokens precede the target). The last-row "
         "distribution <b>A[:,&minus;1,:]</b> is the &ldquo;which past hours matter&rdquo; map. "
         "Training loss: MSE on z-score. Reported metric: MAE in orders/hour (MSE targets the conditional mean, "
         "MAE the conditional median; with right-skewed spikes they differ slightly &mdash; accepted mismatch).")
    h1(s, "2. Synthetic Warehouse Environment")
    code(s, "L(t) = mu &middot; Trend(t) &middot; Daily(hour) &middot; Weekly(dow) &nbsp; deterministic level<br/>"
            "y(t) = round(max(0, L(t) &middot; (1+eps(t)) &middot; E(t))) &nbsp; observed demand<br/>"
            "eps(t) = 0.5 eps(t-1) + eta(t), eta ~ N(0, sigma_eta^2) &nbsp; AR(1) relative noise<br/>"
            "E(t) = product of active spike / drop multipliers (multiplicative: busy hours fluctuate proportionally more)")
    table(s, ["Component", "Regime A (train)", "Notes"],
        [["Base &mu;", "200 orders/hour", "Level scale"],
         ["Trend 1+0.0005&middot;day", "+9% over 180 days", "Mild; invisible in 24h window"],
         ["Daily: 0.35+0.9&middot;g(h;10,2)+1.0&middot;g(h;15,2.5), mean-normalised", "Night trough ~0.44, plateau 09&ndash;16h ~1.4&ndash;1.8", "Circular Gaussian bumps; broad plateau"],
         ["Weekly Mon..Sun", "1.05,1.05,1.00,1.00,0.95,0.70,0.50", "Weekday/weekend pattern"],
         ["Noise AR(1) &rho;=0.5, &sigma;<sub>&eta;</sub>=0.06", "Stationary rel. std ~0.069", "Heteroscedastic in absolute terms"],
         ["Spikes p=1/72, mult 1+m&middot;e^(&minus;k/2), m~U(0.5,1), &tau;=2h", "~1 per 3 days, ~6h tail", "Flash-sale burst"],
         ["Drops p=1/144, mult (1&minus;d) len 1&ndash;3h, d~U(0.5,0.9)", "~1 per 6 days", "Outage / carrier hold"]],
        caption="Table 1. Regime A generator (src/data_generator.py, REGIME_A).")
    h2(s, "2.1 Splits and reproducibility")
    bullets(s, [
        "Total 180 days = 4320 h, one continuous series, <b>data_seed=1234</b>. Chronological splits: "
        "train days 0&ndash;119, val 120&ndash;149, test 150&ndash;179. All 25 h (24 inputs + target) must lie inside one split &mdash; no straddle, no leakage.",
        "Overlapping stride-1 windows (2856/696/696) are heavily autocorrelated: effective N is far below window count; confidence intervals must say so.",
        "Shifted 30-day series from Regime B with <b>shift_seed=4321</b>, normalised with <i>training</i> &mu;,&sigma;. Current data: &mu;<sub>train</sub>=185.85, &sigma;<sub>train</sub>=116.27; train spikes 33, drops 20.",
        "<b>Common random numbers:</b> each random quantity has its own RNG stream drawn full-length, so changing &sigma;<sub>&eta;</sub> changes only noise scale; changing spike rate changes only spike positions. This makes the B-noise / B-spikes decomposition clean.",
        "<b>Pure function:</b> generate_series(config, seed). Model results: mean&plusmn;std over 5 model_seeds 0&ndash;4. Same seed &rarr; byte-identical (tested). Day 0 = Monday."])
    h1(s, "3. Statistical Assumptions")
    bullets(s, [
        "Daily/weekly structure is deterministic given calendar; history adds three channels: (a) local AR(1) deviation (&rho;=0.5 &rarr; half of deviation persists one hour); (b) event tails (spike decay &tau;=2h; drops 1&ndash;3h then full recovery); (c) same-hour-yesterday (window position 1 = t&minus;23, exactly 24h before target) &mdash; noisy, weekday-biased.",
        "Spike/drop <b>onsets are Bernoulli and memoryless &mdash; unpredictable from history</b>. Best possible onset error is the unconditional expectation: an irreducible floor, not a bug.",
        "Rough irreducible error: with known L(t+1) and &rho;, one-step std is 0.06&middot;L; expected MAE ~0.8&middot;0.06&middot;200 &asymp; <b>10 orders/h</b> at mean level; events raise it.",
        "Gaussian AR(1) relative noise is lighter-tailed than real demand (known simplification). One synthetic world only; conclusions are about this generator."])
    h1(s, "4. Evaluation Methodology")
    table(s, ["Metric", "Role", "Rationale"],
        [["MAE (orders/h)", "Primary", "Interpretable, robust to spikes"],
         ["RMSE", "Secondary", "Penalises large misses; reveals spikes"],
         ["MAE event vs non-event", "Diagnostic", "Event = generator multiplier != 1 at target hour"],
         ["Skill = 1 &minus; MAE_model/MAE_baseline", "Comparison", "Scale-free advantage"]],
        caption="Table 2. Metrics fixed before any result.")
    p(s, "Protocol: hyperparameters on validation only; test touched once per frozen configuration; 5 model seeds; "
         "report mean&plusmn;std plus per-seed values.")
    h1(s, "5. Baselines and Expected Behaviour")
    table(s, ["Baseline", "Definition", "Expectation"],
        [["Last value", "y_hat = y(t)", "Error from hour-to-hour slope of daily curve + noise; band 25&ndash;35"],
         ["Moving average 24h", "mean of window", "Ignores time-of-day; very poor at peaks/troughs; band 45&ndash;70"],
         ["Seasonal naive", "y_hat = y(t&minus;23)", "Captures daily shape; doubles noise; fails Fri&rarr;Sat, Sat&rarr;Sun, Sun&rarr;Mon; band 30&ndash;45"],
         ["No-attention control", "A replaced by uniform 1/24", "Mean-pooling of V; isolates whether attention helps"]],
        caption="Table 3. Baselines (bands are soft predictions except ordering).")
    h1(s, "6. Planned Generalisation Experiment (Distribution Shift)")
    table(s, ["Parameter", "Regime A", "Regime B (shift)"],
        [["&sigma;_eta", "0.06", "0.12 (2x)"],
         ["Spike p", "1/72", "1/36 (2x frequent)"],
         ["Spike m", "U(0.5,1.0)", "U(1.0,2.0) (2x larger)"],
         ["Drops / seasonality / calendar", "as above", "unchanged (isolates noise+events)"]])
    p(s, "Decomposition runs: <b>B-noise</b> (only &sigma; changed), <b>B-spikes</b> (only spikes changed), <b>B-full</b>. "
         "Motivation: peak-season / promo period &mdash; volatility the deployed model never trained on. Document: "
         "original distribution, modified distribution, predicted impact (H4), observed impact, mathematical interpretation.")
    h1(s, "7. Pre-registered Hypotheses (with numbers, so they can be wrong)")
    h2(s, "H1 &mdash; Scaling controls softmax saturation (toy associative recall, T=8 pairs, d_k in {4,16,64,256}, 5 seeds)")
    p(s, "Reasoning: if q,k components are ~zero-mean unit-variance independent, Var(q&middot;k)=d<sub>k</sub>, so unscaled "
         "logits have std sqrt(d<sub>k</sub>) (std 8 at d<sub>k</sub>=64). Softmax over such logits is near one-hot; Jacobian "
         "diag(a)&minus;aa<sup>T</sup>&rarr;0 kills gradients to W<sub>Q</sub>,W<sub>K</sub>. Prediction: at init with d<sub>k</sub>=64, "
         "mean max-weight ~0.2&ndash;0.35 scaled vs &gt;0.8 unscaled; gradient-norm gap grows with d<sub>k</sub>; no difference at "
         "d<sub>k</sub>=4 (within 0.1). Unscaled needs &ge;1.5x steps to 90% at d<sub>k</sub>=64 (or fails on &ge;1 seed). "
         "<b>Falsified if:</b> unscaled reaches 90% within &le;1.2x steps of scaled at d<sub>k</sub>=64 on &ge;4/5 seeds.")
    h2(s, "H2 &mdash; Baseline ordering (test MAE)")
    p(s, "Prediction: <b>attention &asymp; control &lt; last &lt; seasonal &lt; MA-24</b>, bands model 11&ndash;16, last 25&ndash;35, "
         "seasonal 30&ndash;45, MA-24 45&ndash;70. <b>Falsified if:</b> model does not beat every baseline or baseline order differs. Bands soft; ordering + &ldquo;model best&rdquo; hard.")
    h2(s, "H3 &mdash; Attention adds little over pooling in-distribution")
    p(s, "Reasoning: seasonal level is a calendar function and both models see the last-token embedding directly; "
         "attention&rsquo;s extra peek at yesterday-same-hour is weak. Prediction: MAE gap <b>&lt;5%</b>, control wins some seeds. "
         "<b>Falsified if:</b> attention beats control by &gt;5% on &ge;4/5 seeds.")
    h2(s, "H4 &mdash; Shift erodes the advantage")
    p(s, "All methods&rsquo; MAE rises 1.8&ndash;3x on B-full; skill vs seasonal <b>shrinks by &ge;half</b>; model still beats MA-24; "
         "B-spikes hurts relative standing more than B-noise. <b>Falsified if:</b> B-full skill stays above half of in-distribution skill.")
    h2(s, "H5 &mdash; Where attention looks (visualisation, not causation)")
    p(s, "Averaged A[last,:] concentrates on position 24 then position 1; combined mass <b>&gt;0.30</b> vs 0.083 uniform. "
         "Known tension: if H3 holds, H5 may fail &mdash; both outcomes informative. Weights describe computation, not causation.")
    h2(s, "V1 &mdash; Verification (not about data)")
    p(s, "float64 central differences (&epsilon;=1e-5) vs autograd: max relative error <b>&lt;1e-6</b> for W<sub>Q</sub>,W<sub>K</sub>,W<sub>V</sub>,X. "
         "float32 expected ~1e-3&ndash;1e-2 (roundoff). Test tolerance 1e-6 in float64.")
    h1(s, "8. System-Specific Failure Modes (predicted)")
    table(s, ["#", "Failure mode", "Mechanism / diagnostic"],
        [["F1", "Spike onset missed", "Onset memoryless; error ~m&middot;L up to 100% level. Split onset vs non-event error"],
         ["F2", "Spike contaminates window", "Spike at pos 1 inflates yesterday-hour; at pos 24 momentum overshoots decay. Split by event position"],
         ["F3", "Drop misread as trend", "1&ndash;3h drops then rebound; last-token reliance under-forecasts rebound"],
         ["F4", "Out-of-range inputs under shift", "Regime-B spikes push |z| beyond train range; linear/ReLU extrapolation overshoots"],
         ["F5", "Softmax saturation / collapse", "Large ||QK^T|| &rarr; one-hot rows &rarr; dead W_Q,W_K grads. Track entropy, max-weight, grad norms"],
         ["F6", "Weekday-boundary / thin cells", "~17 examples per (dow,hour); Sat&rarr;Sun/Sun&rarr;Mon steps (Sun&rarr;Mon +110%). Error by dow/hour"],
         ["F7", "Unseen trend", "Test ~6&ndash;9% above train mean; train-&mu; z-scoring under-forecasts. Signed bias over time"]],
        caption="Table 4. Predicted failures. Formal investigation: F1/F2 (or F4 if clearer); choice depends on observation, not story-fitting.")
    h1(s, "9. Amendments Log")
    table(s, ["Date", "Change", "Reason / results seen"],
        [["30 Sep 2026", "Sec.2.1 prose: bumps at 10/15 (&sigma; 2,2.5) overlap into one broad 09&ndash;16h plateau (~1.7x mean, 1.4&ndash;1.8) with 0.44 night trough; formula unchanged.",
          "Unit test test_daily_shape_is_night_trough_and_daytime_plateau. No model/baseline results seen; H1&ndash;H5 unaffected."]])
    h1(s, "10. Traceability")
    table(s, ["Item", "Phase 0", "Implementation", "Results"],
        [["Data generator", "Sec.2", "src/data_generator.py", "results/data/"],
         ["Attention core", "Sec.1 + derivation", "src/attention.py", "results/gradcheck.txt"],
         ["Toy + ablation", "H1, Sec.10&ndash;11", "experiments/toy_ablation.py", "results/ablation/"],
         ["Baselines + model", "Sec.1,5; H2,H3,H5", "experiments/warehouse.py + src/models.py, src/baselines.py", "results/warehouse/"],
         ["Distribution shift", "Sec.6; H4", "experiments/shift.py", "results/shift/"],
         ["Failure investigation", "Sec.8", "docs/failure_investigation.md", "warehouse_results.json"]])
    p(s, "Assumptions where the PRD was ambiguous: learned positional embedding + calendar; d_model=16 projection of 5 features "
         "(d_k=1 would make scaling meaningless); single-head single-layer last-query; manual backprop only if time; seasonal-naive + "
         "no-attention control added to answer &ldquo;does attention add value?&rdquo;; orders/hour synthetic units. "
         "Limitations known in advance: Gaussian AR(1) light tails; one synthetic world; correlated windows; one data seed.")
    build(OUT / "Phase0_Design_Document.pdf", "Phase 0 Design Document", s)

# ---------------------------------------------------------------- 2. Math derivation
def doc_math():
    s = []
    title_block(s, "Mathematical Derivation",
                "Scaled Dot-Product Self-Attention &mdash; From First Principles",
                META + ["Implements: src/attention.py (ScaledDotProductAttention, stable_softmax)"])
    p(s, "Notation: batch B, window T=24, input dim 5 &rarr; d<sub>model</sub>=16, d<sub>k</sub>=d<sub>v</sub>=16. "
         "This document derives every operation a reviewer can identify in the source: Q/K/V, QK<sup>T</sup>, 1/sqrt(d<sub>k</sub>), "
         "softmax, AV, and the tensor shapes. The toy recall task and the warehouse forecaster are the same mathematics with different inputs.")
    h1(s, "1. Query, Key, Value &mdash; Why Three Projections?")
    code(s, "Q = X W_Q &nbsp; W_Q in R(d_in x d_k) &nbsp; 'what does position i ask for?'<br/>"
            "K = X W_K &nbsp; W_K in R(d_in x d_k) &nbsp; 'what does position j offer?'<br/>"
            "V = X W_V &nbsp; W_V in R(d_in x d_v) &nbsp; 'what does position j contribute if selected?'")
    p(s, "Input tokens X mix &ldquo;what I seek&rdquo; with &ldquo;what I offer&rdquo;. One projection cannot express asymmetric "
         "retrieval (&ldquo;hour 23 queries yesterday&rsquo;s hour, but not vice versa&rdquo;). Two projections (K,V) cannot separate "
         "<i>match score</i> from <i>payload</i>: a token could match strongly yet carry little content. Three maps decouple "
         "addressing (Q,K) from content (V) &mdash; the key&ndash;value memory view used by the toy task (pair keys&rarr;K, pair values&rarr;V, query&rarr;Q).")
    h1(s, "2. Dot-Product Similarity &mdash; Why QK<sup>T</sup>?")
    p(s, "For query q<sub>i</sub> and key k<sub>j</sub>, score S<sub>ij</sub>=q<sub>i</sub>&middot;k<sub>j</sub>=||q||&middot;||k||&middot;cos(&theta;) "
         "is the unnormalised log-odds that &ldquo;i should read j&rdquo;: large when request matches offer. The Gram-like matrix "
         "S=QK<sup>T</sup> in R<sup>B&times;T&times;T</sup> scores every past position against every query; the warehouse model uses only "
         "the last row S[:,&minus;1,:] (&ldquo;which past hours matter for next hour?&rdquo;) but computes all rows identically.")
    h1(s, "3. Scaling &mdash; Why 1/sqrt(d<sub>k</sub>)? (H1 mechanism)")
    p(s, "Assume components of q,k i.i.d. zero-mean unit-variance. Then Var(q&middot;k)=Var(&Sigma;<sub>l</sub> q<sub>l</sub>k<sub>l</sub>) "
         "= &Sigma;<sub>l</sub> Var(q<sub>l</sub>)Var(k<sub>l</sub>) = <b>d<sub>k</sub></b>, so std(q&middot;k)=sqrt(d<sub>k</sub>). "
         "Unscaled logits at d<sub>k</sub>=64 have std ~8, at d<sub>k</sub>=256 std ~16: softmax is near one-hot. Dividing by sqrt(d<sub>k</sub>) "
         "restores unit-variance logits at init regardless of width, keeping softmax diffuse and gradients alive. At d<sub>k</sub>=4 "
         "(std 2) scaling barely matters &mdash; the H1 falsification boundary.")
    p(s, "Softmax Jacobian: d<b>a</b>/d<b>s</b> = diag(a)&minus;aa<sup>T</sup>. At one-hot a&asymp;e<sub>j</sub> this matrix &rarr; 0, so "
         "gradients to W<sub>Q</sub>,W<sub>K</sub> vanish while the W<sub>V</sub> path (linear Y=AV) survives. The toy ablation measures exactly "
         "this: ||grad W<sub>Q</sub>|| scaled vs unscaled across d<sub>k</sub> in {4,16,64,256}. Observed: unscaled needs 6x steps at d<sub>k</sub>=64 "
         "and never learns at d<sub>k</sub>=256 (acc 0.80 vs 1.00).")
    h1(s, "4. Softmax &mdash; Logits to Weights (numerical stability)")
    code(s, "A_ij = exp(S_ij &minus; m_i) / sum_j exp(S_ij &minus; m_i), &nbsp; m_i = max_j S_ij")
    p(s, "Each row of A is a distribution (A&ge;0, rows sum to 1; asserted in tests). Subtracting m<sub>i</sub> is the identity "
         "softmax(x)=softmax(x&minus;c) since exp(x&minus;c)/&Sigma;exp(x&minus;c)=exp(x)/&Sigma;exp(x), but bounds the largest exponent at "
         "exp(0)=1. Naive exp overflows fp32 for x&#8819;89 (&#8819;710 fp64) &rarr; inf/inf=nan. The demo in experiments/gradcheck.py shows "
         "naive&rarr;nan at logits &ge;89 while max-subtraction returns the exact uniform row; threshold scan 80/89/90/100 documented in results/gradcheck.txt.")
    h1(s, "5. Weighted Aggregation &mdash; What Is AV?")
    p(s, "Y=AV, i.e. y<sub>i</sub>=&Sigma;<sub>j</sub> A<sub>ij</sub>v<sub>j</sub>: each output is a convex combination of value vectors gated by "
         "the learned addressing pattern. In the warehouse model Y[:,&minus;1,:] is the retrieved context for the last hour, concatenated with the "
         "last-token embedding E[:,&minus;1,:] (current demand/calendar direct) before the MLP head. The no-attention control replaces A by uniform "
         "1/24 (mean-pooling of V) &mdash; the H3 comparison, which matches attention within 1.5%.")
    h1(s, "6. Tensor Shapes (B=batch, T=24, d_in=5, d_model=d_k=d_v=16)")
    table(s, ["Tensor", "Shape", "Meaning"],
        [["X", "(B,24,5)", "z-demand + sin/cos hour + sin/cos dow"],
         ["W_e / P", "(5,16) / (24,16)", "input projection / learned positions"],
         ["E", "(B,24,16)", "embedded tokens"],
         ["W_Q,W_K,W_V", "(16,16)", "QKV projections"],
         ["Q,K,V", "(B,24,16)", "queries / keys / values"],
         ["S", "(B,24,24)", "scaled logits"],
         ["A", "(B,24,24)", "weights, rows sum to 1"],
         ["Y", "(B,24,16)", "aggregated context"],
         ["h", "(B,32)", "concat(Y[:,-1], E[:,-1])"],
         ["y_hat", "(B,)", "next-hour z-score (de-normalised for reporting)"]])
    h1(s, "7. Gradients Verified (V1)")
    p(s, "float64 central differences (&epsilon;=1e-5) vs autograd: max relative error ~1e-9 (W<sub>Q</sub> 1.7e-9, W<sub>K</sub> 7.9e-10, "
         "W<sub>V</sub> 8.3e-11, X 2.6e-9) &lt; 1e-6 &rarr; PASS. float32 shows ~1e-1: central differences divide ~1e-7 roundoff by "
         "&epsilon;=1e-5, so this is expected roundoff, not a bug (see Experiment Report Sec.2). Row-permutation sanity: without positions, "
         "shuffling tokens permutes outputs identically (permutation equivariance &mdash; why positions/calendar are required). Hand T=3/d=2 "
         "example and plain-NumPy reference in tests/test_attention.py (8 tests) lock the computation independently of torch.")
    h1(s, "8. Connection to the Experiments")
    bullets(s, [
        "Toy H1: init max-weight ~0.40 flat scaled vs 0.59/0.81/0.90/0.95 unscaled at d_k=4/16/64/256; steps-to-90% 200/220, 100/190, 50/300, 50/never.",
        "Warehouse H3/H5: uniform control isolates attention's value (1.5% gap); A[last,:] inspected at pos24+pos1 (0.168 mean vs 0.30 predicted &rarr; H5 falsified).",
        "Shift H4: B-spike |z| values leave the training range; linear embedding + ReLU MLP extrapolate linearly &rarr; overshoot (F4)."])
    build(OUT / "Mathematical_Derivation.pdf", "Mathematical Derivation", s)

# ---------------------------------------------------------------- 3. AI tools
def doc_ai():
    s = []
    title_block(s, "AI Tools Document",
                "AI Assistance Log (PRD s25) &mdash; Tool Use, Verification, Understanding",
                META + ["Tool: OpenCode with Muse Spark (Meta muse-spark-1.3), agentic code + bash, 30 Sep&ndash;01 Oct 2026"])
    p(s, "No other AI tools were used. All AI output was reviewed, tested, corrected and re-run by the candidate; "
         "the mathematics, hypotheses and verdicts are the candidate&rsquo;s own. A candidate who uses AI but cannot explain the "
         "mathematics receives no credit &mdash; Section 5 is the understanding check. Two real bugs occurred during the build, "
         "both found by <i>running code</i>; neither was hidden (PRD s24).")
    h1(s, "1. Tool Declaration")
    table(s, ["Item", "Detail"],
        [["Assistant", "OpenCode agent powered by Muse Spark (muse-spark-1.3, Meta)"],
         ["Mode", "Agentic: scaffold code + run pytest / experiment CLIs in the repo"],
         ["Other tools", "None (no ChatGPT / Copilot / Cursor)"],
         ["Responsibility", "Candidate owns correctness; every AI artefact verified by tests or re-runs"]])
    h1(s, "2. Task-by-Task Log")
    table(s, ["Task requested", "Generated output", "Candidate modifications / verification"],
        [["src/attention.py scaffold", "stable/naive softmax, QKV module", "Accepted; added numpy_reference_attention; fixed use_scale API for ablation"],
         ["tests/test_attention.py", "8 tests incl. hand T=3/d=2 example", "Fixed dtype bug found by pytest (float64 X into float32 module &rarr; RuntimeError); corrected to float32; 8/8 pass"],
         ["experiments/gradcheck.py", "central-difference gradcheck", "Fixed torch.ndindex (does not exist) &rarr; itertools.product; re-ran: float64 PASS 1e-9"],
         ["experiments/toy_ablation.py", "recall task + sweep", "Rewrote Q/K init to N(0,1) to match H1 unit-variance premise; removed dead placeholders; piloted d_k=64 before full sweep"],
         ["src/models.py, src/baselines.py, experiments/warehouse.py, experiments/shift.py", "model/control/baselines/training/eval", "Accepted structure; rewrote warehouse.py stub fully; piloted 1 seed/20 epochs before 5-seed run"],
         ["Docs (derivation, failure, reflection, demo)", "first drafts", "Rewrote with actual numbers; corrected float32-roundoff explanation to observed ~1e-1"]],
        caption="Table 1. Full AI-assistance log. Every row was executed and checked.")
    h1(s, "3. Debugging Evidence (PRD s24) &mdash; Two Real Bugs")
    table(s, ["#", "Symptom", "Hypothesis &rarr; diagnostic &rarr; fix &rarr; verification"],
        [["Bug 1", "pytest RuntimeError: expected float32 module, got float64 X in attention test",
          "Hypothesis: default torch module is float32 while NumPy fixtures default float64. Diagnostic: reran failing test. Fix: cast fixtures to float32. Verified: 8/8 attention tests pass"],
         ["Bug 2", "AttributeError: torch.ndindex does not exist (AI-hallucinated API) in gradcheck",
          "Hypothesis: API invented by generator. Diagnostic: python -c import check. Fix: itertools.product over index ranges. Verified: gradcheck PASS, results/gradcheck.txt written"]])
    p(s, "Both bugs were found by running code (pytest / python -m experiments.gradcheck), not by inspection. The log was not sanitised: "
         "soft-band misses (seasonal 48.65 vs 30&ndash;45, MA-24 106.15 vs 45&ndash;70) were kept visible instead of re-fitting bands to results.")
    h1(s, "4. Where AI Helped vs Where It Was Corrected")
    h2(s, "Helped")
    bullets(s, ["Scaffolding (attention module, test skeletons, experiment runners, doc drafts) and fast API-error turnaround.",
                "Repetitive evaluation plumbing (per-seed loops, JSON summaries, bar/curve plots)."])
    h2(s, "Corrected / rejected")
    bullets(s, ["Toy init scale: AI default uniform init (norms O(0.1)) would have made unscaled logits never saturate and H1 untestable &mdash; rewritten to N(0,1) after reasoning about the H1 premise.",
                "torch.ndindex hallucination (Bug 2 above).",
                "float32-roundoff explanation rewritten to match the observed ~1e-1, not textbook 1e-3.",
                "Soft-band expectations kept as misses (intellectual honesty) rather than adjusted post hoc."])
    h1(s, "5. Understanding Check (candidate's own words)")
    p(s, "Scaling keeps Var(logit)=1 so the softmax Jacobian diag(a)&minus;aa<sup>T</sup> stays away from zero; without it, one-hot rows kill "
         "gradW<sub>Q</sub>, gradW<sub>K</sub> while the AV path keeps gradW<sub>V</sub> alive &mdash; which is why unscaled toy runs learn values slowly "
         "rather than not at all at d<sub>k</sub>=64, and stall at d<sub>k</sub>=256. The warehouse control matching attention (1.5%) shows the win "
         "is features+MLP, not retrieval; diffuse weights <i>describe</i> a seasonal smoother, they do not causally explain each forecast.")
    build(OUT / "AI_Tools_Document.pdf", "AI Tools Document", s)

# ---------------------------------------------------------------- 4. Experiment report
def doc_exp():
    s = []
    title_block(s, "Experiment Report",
                "Toy Recall &middot; Ablation &middot; Warehouse Forecasting &middot; Shift &middot; Failure &middot; Verification",
                META + ["Commands: data regen &rarr; pytest &rarr; gradcheck &rarr; toy_ablation &rarr; warehouse &rarr; shift &rarr; plots"])
    h1(s, "1. Datasets and Reproducibility")
    p(s, "Generator src/data_generator.py (pure function of config+seed; CRN streams per quantity). Regime A train "
         "(data_seed 1234): &mu;<sub>train</sub>=185.85, &sigma;<sub>train</sub>=116.27; windows 2856/696/696; train spikes 33, drops 20. "
         "Regime B (shift_seed 4321): &sigma;<sub>&eta;</sub> 0.06&rarr;0.12, spike p 1/72&rarr;1/36, m U(0.5,1)&rarr;U(1,2); drops unchanged. "
         "Shift sets (30-day, train-&mu;/&sigma;): shift_noise / shift_spikes / shift_full + fresh Regime-A set, 696 windows each. "
         "Chronological splits; no window straddles a boundary; leakage tests corrupt y[t+1] and recount boundaries (22/22 data tests pass). "
         "Fresh-clone run 01 Oct 2026: data regen byte-identical, pytest 32/32, gradcheck identical, shift identical, warehouse smoke OK.")
    fig(s, "results/data/sample_data.png", 16, "Figure 1. Sample synthetic demand: daily plateau, weekly modulation, AR(1) noise, spike/drop events.")
    h1(s, "2. Gradient Verification + Numerical Stability (V1 &mdash; confirmed)")
    p(s, "experiments/gradcheck.py: float64 central differences (&epsilon;=1e-5) vs autograd. Max relative errors: W<sub>Q</sub> 1.7e-9, "
         "W<sub>K</sub> 7.9e-10, W<sub>V</sub> 8.3e-11, X 2.6e-9 &mdash; all &lt; 1e-6: PASS (results/gradcheck.txt). float32 shows ~1e-1: "
         "dividing ~1e-7 roundoff by &epsilon;=1e-5 inflates error six orders above the float64 1e-9 level &mdash; expected roundoff, not a bug. "
         "Stability: naive exp overflows fp32 at logits &#8819;89 (&#8819;710 fp64) &rarr; inf/inf=nan; max-subtracted softmax "
         "(identity softmax(x)=softmax(x&minus;c)) returns the exact uniform row; threshold scan 80 OK / 89,90,100 naive FAIL, stable OK.")
    h1(s, "3. Toy Associative Recall + Ablation (H1 &mdash; confirmed, 6x over)")
    p(s, "Task: N=8 (key,value) pairs (16 keys/16 values) + query token carrying a key from the sequence; target = paired value; chance 6.25%. "
         "Model: one-hot key+value (32-dim) &rarr; single-head attention (d<sub>k</sub> in {4,16,64,256}, Q/K init N(0,1) per H1 premise) &rarr; linear head on last position. "
         "Adam lr 1e-2, batch 128, 3000-step budget, 5 seeds; scaled softmax(QK<sup>T</sup>/sqrt(d<sub>k</sub>)) vs unscaled; logs: loss/acc curves, query-row entropy+max-weight init/end, ||grad W<sub>Q</sub>||/||grad W<sub>K</sub>||, steps-to-90%.")
    table(s, ["d_k", "Scaled acc / steps90 / init_max", "Unscaled acc / steps90 / init_max", "Reading"],
        [["4", "1.000 / 200 / 0.395", "1.000 / 220 / 0.588", "No meaningful gap (H1 boundary)"],
         ["16", "1.000 / 100 / 0.420", "1.000 / 190 / 0.806", "Gap opens"],
         ["64", "1.000 / 50 / 0.415", "0.990 / 300 / 0.902", "6x slowdown (>=1.5x predicted)"],
         ["256", "1.000 / 50 / 0.422", "0.796 / never (0/5) / 0.950", "Phase change: never learns"]],
        caption="Table 1. Toy ablation (5 seeds; results/ablation/toy_summary.txt). Scaled init_max ~0.40 flat; unscaled 0.59/0.81/0.90/0.95.")
    p(s, "Mechanism (training dynamics): Var(q&middot;k)=d<sub>k</sub> &rarr; one-hot rows &rarr; Jacobian diag(a)&minus;aa<sup>T</sup>&rarr;0 &rarr; dead "
         "W<sub>Q</sub>/W<sub>K</sub> grads while AV keeps W<sub>V</sub> alive: slow (not zero) learning at d<sub>k</sub>=64, stall at 256. "
         "Hardest implementation point: default uniform init (norms O(0.1)) never saturates &mdash; recognising the init-scale mismatch with H1 and "
         "switching the toy to N(0,1) Q/K init made the effect testable. (Scaled init_max 0.42 marginally above the 0.2&ndash;0.35 pre-registered band; H1 hard clauses still hold.)")
    fig(s, "results/ablation/toy_ablation.png", 16, "Figure 2. Toy ablation: accuracy/loss dynamics and attention saturation by d_k (scaled vs unscaled).")
    h1(s, "4. Warehouse Forecasting (H2 ordering confirmed; H3 confirmed; H5 falsified)")
    p(s, "Model (src/models.py, reuses Problem-1 attention): E=XW<sub>e</sub>+b<sub>e</sub>+P &rarr; QKV (16-dim) &rarr; S=QK<sup>T</sup>/sqrt(d<sub>k</sub>) "
         "&rarr; A &rarr; Y &rarr; h=concat(Y[:,-1],E[:,-1]) &rarr; MLP(32&rarr;32&rarr;1). Control: identical with A=uniform 1/24. "
         "Train: MSE on z-score, Adam lr 1e-2, 120 epochs, batch 256, best-val checkpoint; 5 seeds 0&ndash;4; test evaluated once (MAE/RMSE orders/h, event splits, skill vs seasonal, mean A[last,:]).")
    table(s, ["Model / baseline", "Test MAE", "RMSE", "Note"],
        [["Attention", "14.78 &plusmn; 0.44", "~27.3", "skill vs seasonal 0.696; best all 5 seeds"],
         ["No-attention control", "14.99 &plusmn; 0.36", "~27.5", "within 1.5% -> H3 holds; wins seeds 0,2"],
         ["Last value", "30.43", "44.85", "band 25-35 OK"],
         ["Seasonal naive", "48.65", "77.40", "band 30-45 marginal miss"],
         ["MA-24", "106.15", "120.79", "band 45-70 miss: ignoring time-of-day worse than guessed"]],
        caption="Table 2. Warehouse test results (results/warehouse/warehouse_results.json). Ordering attention ~ control < last < seasonal < MA-24 holds every seed (H2).")
    fig(s, "results/warehouse/attention_attention.png", 16, "Figure 3. Mean last-row attention on test (diffuse; pos24+pos1 mean 0.168 vs 0.30 predicted).")
    fig(s, "results/warehouse/attention_curves.png", 12, "Figure 4. Training curves (train MSE-z, val MSE-z; best-val checkpointing).")
    p(s, "Attention weights stay near-diffuse: pos24+pos1 mean <b>0.168</b> (best seed 0.247) vs 0.30 predicted; entropy ~3.0 vs uniform 3.18 "
         "&rarr; <b>H5 falsified</b>. Resolves the Phase-0 H3/H5 tension toward H3: the model barely attends at all, behaving like a seasonal smoother + AR(1) correction.")
    h1(s, "5. Distribution Shift (H4 &mdash; mixed)")
    table(s, ["Set", "Attention MAE", "Seasonal MAE", "Skill", "Ratio vs test"],
        [["Test (A)", "14.78", "48.65", "0.696", "1.00x"],
         ["B-noise", "21.64", "57.29", "0.622", "1.46x"],
         ["B-spikes", "21.33", "64.42", "0.669", "1.44x"],
         ["B-full", "29.13", "73.65", "0.604", "1.97x (band 1.8-3x OK)"]],
        caption="Table 3. Shift results, frozen checkpoints (results/shift/shift_results.json). Control beats attention on every shift set (slight Regime-A overfit).")
    fig(s, "results/shift/shift_bars.png", 16, "Figure 5. MAE by test set: attention vs last/seasonal baselines.")
    bullets(s, ["Magnitude 1.97x on B-full: within predicted 1.8&ndash;3x.",
                "Skill 0.696&rarr;0.604: NOT halved &rarr; H4 falsified clause. Model degrades gracefully.",
                "B-noise hurts relatively more than B-spikes (skill 0.622 vs 0.669): opposite of prediction &rarr; H4 falsified clause. The learned AR(1)-style correction is more brittle than the spike response."])
    h1(s, "6. Failure Investigation (F1/F2: spike-onset miss &mdash; 11x gap)")
    table(s, ["Target subset (test)", "n", "Attention MAE", "vs non-event"],
        [["Spike onset", "10", "~131-133", "~11x"],
         ["Drop onset", "6", "~94-100", "~8x"],
         ["Non-event", "~580", "~11-12", "baseline"],
         ["All spike-active / drop-active", "111 / 13", "~24 / ~64-78", "tail contamination"]],
        caption="Table 4. Event-split MAE (5 seeds; control identical). 10 onset hours contribute more squared error than all ~580 non-event hours.")
    bullets(s, ["Scenario: test window whose target t+1 is a spike onset (Bernoulli p=1/72, memoryless); window shows an ordinary ramp, no precursor.",
                "Expected (F1): best history-only forecast is the unconditional expectation; onset error ~m*L (+50-100% in Regime A).",
                "Actual: systematic under-forecast (signed error < 0); post-onset windows overshoot the tau=2h decay (F2 tail contamination, positive bias 1-3h after onset); baselines suffer identically -> floor is in the data, not the architecture.",
                "Why attention cannot help: mean A[last,:] near-diffuse (H5) -> seasonal smoother + AR(1) correction cannot foresee a memoryless jump.",
                "Fix (not implemented, out of scope): exogenous promo-calendar feature or event-conditional head; asymmetric/quantile loss if stockouts cost more than overstock. Non-findings checked: F5 no collapse (entropy ~3.0), F7 small bias, F6 weekend elevation without boundary blow-up."])
    h1(s, "7. Test Suite and Reproduction")
    bullets(s, ["pytest 32/32: 22 data-generator + 8 attention + 2 gradcheck/stability (pytest.ini: pythonpath=., testpaths=tests).",
                "Regenerate: python -m src.data_generator --out results/data; python -m pytest -q; python -m experiments.gradcheck; "
                "python -m experiments.toy_ablation --steps 3000 (~25 min CPU); python -m experiments.warehouse --epochs 120 (~20 min); "
                "python -m experiments.shift (~1 min); python -m experiments.plot_data; python -m experiments.plot_results.",
                "Cross-problem integration: Problem 2 reuses the Problem-1 ScaledDotProductAttention verbatim; whether it helped is answered quantitatively (H3: it adds ~nothing here)."])
    build(OUT / "Experiment_Report.pdf", "Experiment Report", s)

# ---------------------------------------------------------------- 5. Reflection
def doc_reflection():
    s = []
    title_block(s, "Reflection Report",
                "What I Expected, What Happened, What the Evidence Shows (PRD s26)",
                META + ["Covers all 11 reflection questions; hypothesis verdicts frozen (amendments only)"])
    qs = [
        ("1. What did you initially expect?",
         "That scaling would matter only at large widths (H1); the attention model would clearly beat all baselines with attention "
         "visibly focused on the last hour and yesterday's same hour (H2/H5); attention would add little over pooling (H3 \u2014 in tension "
         "with H5, as flagged in Phase 0); and the shift would halve the model's skill edge (H4)."),
        ("2. Which hypotheses were correct?",
         "H1 (scaling\u2194saturation): yes \u2014 unscaled init max-weight 0.59/0.81/0.90/0.95 at d<sub>k</sub>=4/16/64/256 vs scaled ~0.40 flat; "
         "steps-to-90% 200/220, 100/190, 50/300, 50/never (unscaled d<sub>k</sub>=256: 0/5 reach 90%, acc 0.80); the \u22651.5x clause at "
         "d<sub>k</sub>=64 held 6x over. H2 ordering: yes \u2014 14.78\u224814.99 < 30.43 < 48.65 < 106.15, model best all 5 seeds. "
         "H3 (<5% attention vs pooling): yes \u2014 1.5% mean gap; control wins seeds 0 and 2. V1 (gradcheck <1e-6 float64): yes \u2014 max rel err ~2.6e-9."),
        ("3. Which hypotheses were wrong?",
         "H5 (pos24+pos1 > 0.30): falsified \u2014 mean 0.168, best seed 0.247; attention near-diffuse (entropy ~3.0 vs uniform 3.18). "
         "The Phase-0 tension resolved toward H3: concentrated attention with no accuracy gain was not even needed. "
         "H4 sub-clauses: magnitude held (B-full 1.97x) but skill fell only 0.696\u21920.604 (no halving), and B-noise hurt relatively "
         "<i>more</i> than B-spikes (0.622 vs 0.669) \u2014 opposite of prediction. Soft bands: seasonal 48.65 (30\u201345) and MA-24 106.15 "
         "(45\u201370) both overshot \u2014 MA-24 far more than guessed, since ignoring time-of-day is worse than estimated."),
        ("4. What surprised you?",
         "Three things. (a) How <i>small</i> the attention contribution is \u2014 the uniform control matches it everywhere, yet the model still "
         "beats last-value 2x, so the win is features+MLP, not retrieval. (b) Unscaled d<sub>k</sub>=256 never learning while d<sub>k</sub>=64 only slows "
         "6x \u2014 a sharp phase change, not gradual. (c) Noise hurting skill more than spikes: the learned AR(1)-style correction is more brittle "
         "than the spike response."),
        ("5. What was the hardest implementation problem?",
         "Toy initialisation: default uniform init gives Q/K norms O(0.1), so unscaled logits never saturate and H1 would be untestable. "
         "Recognising the init-scale mismatch with H1's unit-variance premise and switching the toy to N(0,1) Q/K init took real thought. "
         "The two code bugs (dtype mismatch, torch.ndindex) were trivial by comparison \u2014 both caught by running tests."),
        ("6. What failure did you investigate?",
         "Spike-onset miss (F1+F2): onset MAE ~133 vs non-event ~12 (11x, all seeds, both models). Memoryless onsets are an "
         "information-theoretic floor; post-onset windows overshoot the \u03c4=2h decay. See Experiment Report Sec.6 and docs/failure_investigation.md."),
        ("7. What do you now understand better?",
         "Softmax saturation as a <i>mechanism</i>: Var(q\u00b7k)=d<sub>k</sub> \u2192 one-hot rows \u2192 Jacobian diag(a)\u2212aa<sup>T</sup>\u21920 \u2192 dead "
         "W<sub>Q</sub>/W<sub>K</sub> while AV keeps W<sub>V</sub> alive (hence slow, not zero, learning at d<sub>k</sub>=64). And the difference between "
         "attention weights as <i>description</i> vs causal explanation \u2014 diffuse weights describe a model behaving like a seasonal smoother."),
        ("8. What remains uncertain?",
         "Whether H5 diffuseness is a training artefact (120 epochs, lr 1e-2 \u2014 sharper patterns might emerge with longer training or entropy "
         "penalties) or a true optimum (calendar already carries the seasonal signal). Dataset-to-dataset variance was never measured (one data seed). "
         "Test has only 10 spike onsets and 6 drop onsets \u2014 the failure study is suggestive, not tight."),
        ("9. Where did AI assistance help?",
         "Scaffolding (attention module, test skeletons, experiment runners, doc drafts) and catching API errors fast. Full log in AI Tools Document."),
        ("10. Where did you need to correct or reject AI-generated suggestions?",
         "Toy init scale (uniform default would have made H1 untestable); torch.ndindex hallucination; float32-roundoff explanation rewritten to the "
         "observed ~1e-1; soft-band expectations kept as visible misses instead of re-fitting bands to results."),
        ("11. If given one additional day, what would you investigate?",
         "(a) Extra Regime-A/B series with new seeds for the failure study (per PROGRESS.md integrity note) to tighten onset statistics; "
         "(b) attention-entropy regularisation / longer training to test whether H5 can be recovered; (c) manual backprop bonus (PRD s10); "
         "(d) dataset-to-dataset variance with 3 data seeds."),
    ]
    for i, (q, a) in enumerate(qs):
        h1(s, q)
        p(s, a)
        if i == 1:
            table(s, ["Hypothesis", "Verdict", "Key number"],
                [["H1 scaling/saturation", "Confirmed", "6x slowdown d_k=64; d_k=256 never learns"],
                 ["H2 ordering", "Confirmed (2 soft bands missed)", "14.78~14.99 < 30.43 < 48.65 < 106.15"],
                 ["H3 attention vs pooling", "Confirmed", "1.5% gap; control wins seeds 0,2"],
                 ["H4 shift", "Mixed", "1.97x OK; no skill-halving; noise>spikes"],
                 ["H5 attention focus", "Falsified", "pos24+pos1 0.168 < 0.30"],
                 ["V1 gradcheck", "Confirmed", "float64 ~1e-9 < 1e-6"]])
    h1(s, "Closing")
    p(s, "A technically imperfect but well-investigated result is preferable to unsupported claims of correctness. "
         "The honest summary: scaling theory confirmed decisively; the forecaster works (2x over last-value) but its win is not attention; "
         "the shift degrades gracefully rather than catastrophically; and the dominant error \u2014 memoryless spike onsets \u2014 is a property of "
         "the world, not a defect of the model.")
    build(OUT / "Reflection_Report.pdf", "Reflection Report", s)

if __name__ == "__main__":
    doc_phase0()
    doc_math()
    doc_ai()
    doc_exp()
    doc_reflection()
