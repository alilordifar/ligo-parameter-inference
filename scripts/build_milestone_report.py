"""Generate the DSC 207 milestone PDF.

Usage:
    python scripts/build_milestone_report.py
    python scripts/build_milestone_report.py --output path/to/report.pdf

Dependencies are NumPy, Matplotlib, and ReportLab. The project already uses
NumPy and Matplotlib; install the PDF dependency with:
    python -m pip install reportlab

The numerical summaries below are a milestone snapshot sourced from the saved
outputs in notebooks/main.ipynb. Update them when the executed results change.
"""

from pathlib import Path
import argparse
import html
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "MILESTONE_REPORT.pdf"
PLOT = None


def make_plot():
    o3 = np.array([3.419, 3.434, 1.626, 3.429, 3.434, 3.446, 1.001, 3.416])
    o4 = np.array([10.84, 10.89, 11.22, 13.77, 11.35, 16.32, 12.15, 14.46])
    params = ["Chirp mass", "Mass ratio", "Effective spin", "Distance"]
    o3ks = np.array([0.152, 0.163, 0.123, 0.286])
    o4mean = np.array([0.193, 0.174, 0.138, 0.295])
    o4lo = np.array([0.105, 0.095, 0.080, 0.160])
    o4hi = np.array([0.270, 0.230, 0.245, 0.410])

    plt.rcParams.update({"font.size": 8.2, "axes.titlesize": 9.6, "axes.labelsize": 8.6})
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.05), constrained_layout=True)

    ax = axes[0]
    for x, vals, color, label in [(0, o3, "#2463a9", "O3a"), (1, o4, "#d95f3d", "O4a")]:
        jitter = np.linspace(-0.09, 0.09, len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals, s=25, color=color, alpha=.88, label=label)
        ax.hlines(np.median(vals), x-.20, x+.20, color="black", lw=1.6)
    ax.set_xticks([0, 1], ["O3a", "O4a"])
    ax.set_ylabel(r"Segment strain std. ($10^{-19}$)")
    ax.set_title("A. Detector-noise shift")
    ax.grid(axis="y", alpha=.25)

    ax = axes[1]
    attempted = np.array([1375, 847])
    accepted = np.array([800, 800])
    rates = accepted / attempted * 100
    bars = ax.bar(["O3a", "O4a"], rates, color=["#2463a9", "#d95f3d"], width=.62)
    for b, r, a in zip(bars, rates, attempted):
        ax.text(b.get_x()+b.get_width()/2, r+2, f"{r:.1f}%\n800/{a}", ha="center", va="bottom", fontsize=7.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accepted injections (%)")
    ax.set_title("B. SNR-gate acceptance")
    ax.grid(axis="y", alpha=.25)

    ax = axes[2]
    y = np.arange(4)
    ax.errorbar(o4mean, y, xerr=[o4mean-o4lo, o4hi-o4mean], fmt="o", color="#d95f3d",
                ecolor="#d95f3d", capsize=3, label="O4a: 50-example resamples")
    ax.scatter(o3ks, y, marker="D", s=25, color="#2463a9", label="O3a held-out")
    ax.set_yticks(y, params)
    ax.invert_yaxis()
    ax.set_xlabel("SBC KS statistic (lower is better)")
    ax.set_title("C. Cross-run calibration")
    ax.grid(axis="x", alpha=.25)
    ax.legend(fontsize=6.7, loc="lower right")

    fig.suptitle("Exploratory and preliminary evidence from the executed pipeline", fontsize=11.5, weight="bold")
    fig.savefig(PLOT, dpi=210, facecolor="white")
    plt.close(fig)


NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#2463A9")
PALE = colors.HexColor("#EAF1F8")
LIGHT = colors.HexColor("#F5F7FA")
RED = colors.HexColor("#A23B2A")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D6DEE8"))
    canvas.line(0.55*inch, 0.43*inch, 7.95*inch, 0.43*inch)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#52606D"))
    canvas.drawString(0.58*inch, 0.25*inch, "DSC 207 Milestone | LIGO NF-PE Calibration Pipeline")
    canvas.drawRightString(7.92*inch, 0.25*inch, f"Page {doc.page}")
    canvas.restoreState()


def build():
    make_plot()
    doc = BaseDocTemplate(
        str(OUT), pagesize=letter,
        leftMargin=.58*inch, rightMargin=.58*inch,
        topMargin=.48*inch, bottomMargin=.52*inch,
        title="LIGO NF-PE Calibration Pipeline - Project Milestone",
        author="Ali Lordifar, Kennedy Goliman, Noah Fonck, Aaron Rassiq",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates(PageTemplate(id="milestone", frames=[frame], onPage=footer))

    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleX", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18,
                           leading=20, textColor=NAVY, alignment=TA_LEFT, spaceAfter=5)
    subtitle = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=9.2, leading=11,
                              textColor=colors.HexColor("#40566F"), spaceAfter=7)
    h1 = ParagraphStyle("H1X", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=12.2,
                        leading=14, textColor=NAVY, spaceBefore=5, spaceAfter=3)
    h2 = ParagraphStyle("H2X", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=9.6,
                        leading=11.2, textColor=BLUE, spaceBefore=3, spaceAfter=2)
    body = ParagraphStyle("BodyX", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.25,
                          leading=10.2, textColor=colors.HexColor("#263442"), spaceAfter=3)
    small = ParagraphStyle("Small", parent=body, fontSize=7.3, leading=8.8)
    callout = ParagraphStyle("Callout", parent=body, backColor=PALE, borderColor=BLUE, borderWidth=.6,
                             borderPadding=5, leading=10.4, spaceBefore=3, spaceAfter=5)
    caption = ParagraphStyle("Caption", parent=small, fontName="Helvetica-Oblique", textColor=colors.HexColor("#40566F"),
                             leading=8.6, spaceBefore=2, spaceAfter=4)
    table_head = ParagraphStyle("TableHead", parent=small, fontName="Helvetica-Bold", fontSize=6.8,
                                leading=7.5, textColor=colors.white)
    table_cell = ParagraphStyle("TableCell", parent=small, fontSize=6.45, leading=7.4,
                                textColor=colors.black)

    def wrapped(rows):
        return [[Paragraph(html.escape(str(value)), table_head if i == 0 else table_cell)
                 for value in row] for i, row in enumerate(rows)]

    S = []
    S += [Paragraph("LIGO NF-PE Calibration Pipeline", title),
          Paragraph("Project milestone: calibration drift in gravitational-wave parameter estimation across observing runs", subtitle)]

    team = wrapped([
        ["Team member", "Berkeley email", "Team member", "Berkeley email"],
        ["Ali Lordifar", "ali_lordifar@berkeley.edu", "Kennedy Goliman", "kennedyg@berkeley.edu"],
        ["Noah Fonck", "noahfonck@berkeley.edu", "Aaron Rassiq", "aaronrassiq@berkeley.edu"],
    ])
    t = Table(team, colWidths=[1.12*inch, 2.07*inch, 1.12*inch, 2.07*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 7.6),
        ("BACKGROUND", (0,1), (-1,-1), LIGHT), ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    S += [t, Paragraph("1. Motivation and progress", h1)]
    S += [Paragraph(
        "<b>Question.</b> Does a neural posterior estimator trained on simulated binary-black-hole (BBH) signals in LIGO O3a noise remain calibrated when deployed on O4a noise after a hardware/sensitivity upgrade? This matters because a fast model can retain plausible point estimates while becoming silently overconfident; incorrect credible intervals can propagate into astrophysical conclusions.", body),
        Paragraph("<b>Progress.</b> The team has implemented acquisition, quality vetting, waveform injection, whitening/bandpass/cropping, an SNPE model with a masked autoregressive flow (MAF), checkpointing, and Simulation-Based Calibration (SBC). The executed notebook contains preliminary O3a/O4a results; unit tests cover validation, injection alignment, PSD aggregation, checkpoint integrity, and calibration statistics.", body),
        Spacer(1, 3), Paragraph("Preliminary finding: O4a noise differs strongly from O3a, yet matched-size SBC comparisons show no strong evidence of additional cross-run calibration degradation. Distance is miscalibrated in both runs, so the result is not that the model is fully calibrated.", callout),
        Paragraph("2. Data and experimental split", h1)]

    data_rows = wrapped([
        ["Stage", "Representation and size", "Allocation / role"],
        ["Raw noise", "H1 GWOSC strain; 8 segments/run x 1,024 s x 4,096 Hz = 33,554,432 samples/run (536.87 MB checkpoint/run)", "O3a backgrounds for training; O4a backgrounds for distribution-shift testing"],
        ["Accepted injections", "800/run after SNR >= 8; O3a: 800/1,375 attempts; O4a: 800/847 attempts", "Known BBH parameters injected into event-free real noise"],
        ["Model-ready", "Each example: 16,384 float samples (4 s at 4,096 Hz); target: 4 continuous parameters; processed checkpoint 209.77 MB", "Implemented: 750 O3a train, 0 explicit validation, 50 O3a test, 800 O4a test"],
        ["Planned revision", "Same fixed test sets", "Split the 750-development pool into 600 train / 150 validation; never tune on either test set"],
    ])
    dt = Table(data_rows, colWidths=[.84*inch, 3.70*inch, 2.80*inch], repeatRows=1)
    dt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,0), 7.3),
        ("FONTSIZE", (0,1), (-1,-1), 7.0), ("LEADING", (0,1), (-1,-1), 8.2),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0,0), (-1,-1), "TOP"), ("TOPPADDING", (0,0), (-1,-1), 3.2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3.2),
    ]))
    S += [dt, Spacer(1, 2), Paragraph(
        "<b>Inputs and targets.</b> Input <i>x</i> is a whitened, 35-350 Hz bandpassed, merger-centered H1 strain window. Target <i>theta</i> = (chirp mass, mass ratio, effective spin, luminosity distance). Source: public GWOSC H1 data; labels come from PyCBC IMRPhenomD simulations injected into that noise.", body)]

    # Let the EDA section flow naturally after the data summary. A forced break
    # here left nearly half of page 1 empty and made the report look unfinished.
    S += [Paragraph("3. Exploratory data analysis", h1), Image(str(PLOT), width=7.15*inch, height=2.02*inch)]
    interp = wrapped([
        ["Panel", "Interpretation"],
        ["A - noise distribution", "All eight O4a segment standard deviations exceed the O3a median. In 35-350 Hz, the ASD comparison gives KS=0.902 (p approximately 0) and a 65.1% median relative difference: a clear time/run-dependent covariate shift."],
        ["B - SNR selection", "O3a rejects 41.8% of attempted injections versus 5.5% for O4a. The gate therefore changes the accepted target/SNR population across runs and must be audited as a potential selection effect."],
        ["C - calibration", "For every target, the O3a held-out KS statistic lies inside the range from ten matched-size O4a resamples. This does not prove equivalence, but it provides no strong evidence of extra O4a degradation at n=50."],
    ])
    it = Table(interp, colWidths=[1.20*inch, 6.14*inch])
    it.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 7.3),
        ("LEADING", (0,1), (-1,-1), 8.8), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#CBD5E1")), ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    S += [it, Paragraph("EDA coverage and next analyses", h2), Paragraph(
        "The executed notebook also inspects raw-strain time traces, strain histograms/statistics, merger-centered spectrograms, PSD/ASD curves, injection waveforms, SNR, and target ranges. Because the input is a 16,384-dimensional time series rather than a conventional feature table, raw Pearson feature-target heatmaps are not directly interpretable. The next EDA pass will compute correlations between physical targets and compact summaries (SNR, band powers, peak amplitude, duration), show target histograms/boxplots by run, and use permutation/ablation importance on the baseline and CNN. Derived chirp mass and mass ratio are expected to correlate because both are functions of component masses; this must be measured on accepted injections rather than assumed from priors.", body),
        Paragraph("4. Preprocessing and data challenges", h1)]
    challenge_rows = wrapped([
        ["Requirement", "Implemented treatment / limitation"],
        ["Missing values", "No imputation: reject empty, wrong-length, NaN/Inf, gaps, or failed-data-quality segments; validation occurs at acquisition and checkpoint load."],
        ["Outliers", "Exclude cataloged events (+/-1 h), require CBC data-quality bits, plausible strain std (1e-19 to 1e-17), and SNR >= 8. Training warned of localized >10x-IQR samples; compare robust clipping/no clipping only on validation data."],
        ["Scaling", "Estimate run-specific Welch PSD; whiten strain; zero-phase Butterworth bandpass (35-350 Hz); crop 4 s around merger. SBI z-scores inputs/targets during training."],
        ["Categoricals", "None are model inputs. Detector is fixed to H1; observing run defines train/test domain and is not encoded, preventing leakage."],
        ["Feature selection", "Physics-guided time/frequency selection: merger-centered window and analysis band. The raw samples remain model input; engineered summaries are reserved for baseline/EDA."],
        ["Challenges", "Large downloads/checkpoints, nonstationary noise, few real events, injection-selection shift, expensive rejection sampling, and one failed SBC example per run. Atomic checkpoints, bounded retries, explicit failure indices, and matched-size comparisons address these risks."],
    ])
    ct = Table(challenge_rows, colWidths=[1.15*inch, 6.19*inch], repeatRows=1)
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 7.1),
        ("LEADING", (0,1), (-1,-1), 8.45), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#CBD5E1")), ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TOPPADDING", (0,0), (-1,-1), 3.5), ("BOTTOMPADDING", (0,0), (-1,-1), 3.5),
    ]))
    S += [ct]

    # Continue naturally so the report uses the available page space instead of
    # producing a mostly empty intermediate page.
    S += [Paragraph("5. Methodology and planned experiments", h1)]
    methods = wrapped([
        ["Model", "Why it is appropriate", "Tuning and evaluation"],
        ["Baseline: ridge / elastic-net on engineered summaries", "Transparent, fast sanity check for whether band powers, SNR, amplitude, and duration contain predictive signal.", "Tune regularization and L1 ratio on validation. Report standardized MAE/RMSE and R2 per target."],
        ["Improvement 1: 1D CNN point/heteroscedastic regressor", "Learns local chirp morphology directly from strain; convolution adds translation-tolerant time-frequency feature extraction.", "Tune channels, kernel sizes, dropout, learning rate, and weight decay. Compare point error plus interval coverage/NLL if heteroscedastic."],
        ["Improvement 2: SNPE + MAF posterior (implemented)", "Returns the full conditional posterior needed for calibrated uncertainty, including non-Gaussian parameter degeneracies.", "Tune flow depth/width, transforms, batch size, learning rate, and training simulations. Evaluate SBC KS, rank histograms, empirical 50/90% coverage, posterior width, and sampling-failure rate."],
    ])
    mt = Table(methods, colWidths=[1.34*inch, 2.94*inch, 3.06*inch], repeatRows=1)
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 7.1),
        ("LEADING", (0,1), (-1,-1), 8.5), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#CBD5E1")), ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    S += [mt, Paragraph(
        "<b>Experimental protocol.</b> Use group-aware splits so windows sharing a noise segment never cross train/validation boundaries. Tune only on O3a validation data; freeze preprocessing and hyperparameters before evaluating the O3a and O4a test sets. Repeat training across seeds and report mean plus confidence intervals. Select the best model using validation performance with calibration as a first-class criterion, not test-set point error. Compare O3a versus matched-size O4a resamples to separate distribution shift from sample-size effects.", body),
        Spacer(1, 3), Paragraph("<b>Preliminary results.</b> SNPE+MAF converged after 39 epochs on 750 O3a examples. Valid SBC rows: 49/50 O3a and 799/800 O4a. O3a KS = 0.152/0.163/0.123/0.286 and matched-size O4a means = 0.193/0.174/0.138/0.295 for chirp mass, mass ratio, effective spin, and distance. Distance shows the clearest shared calibration problem. The PSD-drift monitor correctly flags the run shift, but the matched-size SBC evidence does not yet show additional O4a degradation.", callout),
        Paragraph("6. Contributions and notebook ownership", h1)]

    contrib = wrapped([
        ["Member", "Specific contribution / notebook ownership status"],
        ["Ali Lordifar", "Repository integration, executable pipeline, documentation, proposal/PDF, tests, and current Git history. Declared owner: notebooks/main.ipynb."],
        ["Kennedy Goliman", "Proposal feedback reflected in commit 4e4ab51 and learning-material review. Proposed sole owner: notebooks/learn_by_building.ipynb - team must confirm before submission."],
        ["Noah Fonck", "Owner of notebooks/baseline_eda.ipynb: checkpoint audit, explicit O3a validation split, target/feature EDA, engineered summaries, NumPy ridge baseline, regression metrics, residual analysis, and permutation importance."],
        ["Aaron Rassiq", "No notebook ownership or committed contribution is evidenced in the current Git history. Assign a distinct numbered notebook and document completed analysis before submission."],
    ])
    tt = Table(contrib, colWidths=[1.15*inch, 6.19*inch], repeatRows=1)
    tt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 7.05),
        ("LEADING", (0,1), (-1,-1), 8.35), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#CBD5E1")), ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TOPPADDING", (0,0), (-1,-1), 3.5), ("BOTTOMPADDING", (0,0), (-1,-1), 3.5),
    ]))
    S += [tt, Paragraph(
        "<b>Required pre-submission action:</b> Git currently attributes existing commits to Ali; baseline_eda.ipynb is a new uncommitted contribution. The team must confirm Kennedy's proposed notebook ownership, add Aaron's distinct notebook contribution, execute notebooks so code-cell numbering starts at 1, and confirm the final ownership record. Repository: <link href='https://github.com/alilordifar/ligo-parameter-inference' color='#2463A9'>github.com/alilordifar/ligo-parameter-inference</link>", ParagraphStyle("Warn", parent=small, textColor=RED, spaceBefore=4, spaceAfter=3)),
        Paragraph("Sources: repository README, config.py, TUTORIAL.md, PROPOSAL.md, notebooks/main.ipynb saved outputs, and Git history after pull on 2026-09-24; GWOSC data source: https://gwosc.org/data/.", small)]

    doc.build(S)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "MILESTONE_REPORT.pdf",
        help="PDF destination (default: repository root/MILESTONE_REPORT.pdf)",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ligo-milestone-") as temp_dir:
        OUT = args.output.resolve()
        PLOT = Path(temp_dir) / "milestone_eda.png"
        build()
    print(OUT)
