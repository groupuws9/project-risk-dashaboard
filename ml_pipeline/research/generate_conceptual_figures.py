"""Generate professional matplotlib versions of conceptual diagrams."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
FIG_DIR = BASE / "research" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Professional styling
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "blue": "#2c3e50",
    "light_blue": "#3498db",
    "green": "#27ae60",
    "orange": "#e67e22",
    "purple": "#9b59b6",
    "red": "#e74c3c",
    "gray": "#95a5a6",
    "white": "#ffffff"
}


def draw_box(ax, x, y, width, height, text, color, fontsize=9, text_color="white"):
    """Draw a styled box with text."""
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor=COLORS["blue"], linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color, wrap=True)


def draw_arrow(ax, x1, y1, x2, y2):
    """Draw a styled arrow."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                            mutation_scale=20, color=COLORS["blue"], linewidth=2)
    ax.add_patch(arrow)


def fig_3_1_research_workflow():
    """Figure 3.1: Research and evaluation workflow."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Figure 3.1: Research and Evaluation Workflow", 
                 fontsize=16, fontweight="bold", pad=20)

    # Data processing column
    draw_box(ax, 1, 8.5, 2.5, 0.8, "Raw WERS Data", COLORS["light_blue"], 10)
    draw_box(ax, 1, 7.5, 2.5, 0.8, "01_build_dataset.py", COLORS["orange"], 9)
    draw_box(ax, 1, 6.5, 2.5, 0.8, "02_exploratory_analysis.py", COLORS["orange"], 9)
    draw_box(ax, 1, 5.5, 2.5, 0.8, "03_feature_exploration.py", COLORS["orange"], 9)
    
    # Model training column
    draw_box(ax, 4.5, 7.5, 2.5, 0.8, "04_model_training.py", COLORS["red"], 9)
    draw_box(ax, 4.5, 6.5, 2.5, 0.8, "05_imbalance_tuning.py", COLORS["red"], 9)
    draw_box(ax, 4.5, 5.5, 2.5, 0.8, "06_model_evaluation.py", COLORS["red"], 9)
    draw_box(ax, 4.5, 4.5, 2.5, 0.8, "07_shap_explanation.py", COLORS["red"], 9)
    
    # API column
    draw_box(ax, 8, 6.5, 2.5, 0.8, "dashboard/app.py", COLORS["green"], 9)
    draw_box(ax, 8, 5.5, 2.5, 0.8, "tests/test_api.py", COLORS["green"], 9)
    
    # Results
    draw_box(ax, 11, 6, 2.5, 1.5, "Results\n(Tables & Figures)", COLORS["purple"], 10)
    
    # Arrows
    for i in [8.5, 7.6, 6.7, 5.8]:
        draw_arrow(ax, 2.25, i, 2.25, i - 0.9)
    draw_arrow(ax, 3.5, 7.9, 4.5, 7.9)
    draw_arrow(ax, 3.5, 5.9, 4.5, 5.9)
    draw_arrow(ax, 5.75, 7.4, 5.75, 6.4)
    draw_arrow(ax, 5.75, 6.4, 5.75, 5.4)
    draw_arrow(ax, 5.75, 5.4, 5.75, 4.4)
    draw_arrow(ax, 7, 7, 8, 6.9)
    draw_arrow(ax, 7, 5, 8, 5.9)
    draw_arrow(ax, 10.5, 6.9, 11, 6.75)
    draw_arrow(ax, 10.5, 5.9, 11, 6.25)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_3_1_research_workflow.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: fig_3_1_research_workflow.png")


def fig_3_2_artefact_architecture():
    """Figure 3.2: Logical artefact architecture."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Figure 3.2: Logical Artefact Architecture", 
                 fontsize=16, fontweight="bold", pad=20)
    
    # Layer backgrounds
    rect1 = Rectangle((0.5, 8), 11, 1.5, facecolor=COLORS["light_blue"], alpha=0.2, edgecolor=COLORS["blue"])
    rect2 = Rectangle((0.5, 6), 11, 1.5, facecolor=COLORS["orange"], alpha=0.2, edgecolor=COLORS["blue"])
    rect3 = Rectangle((0.5, 4), 11, 1.5, facecolor=COLORS["red"], alpha=0.2, edgecolor=COLORS["blue"])
    rect4 = Rectangle((0.5, 2), 11, 1.5, facecolor=COLORS["green"], alpha=0.2, edgecolor=COLORS["blue"])
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.add_patch(rect4)
    
    # Layer labels
    ax.text(0.7, 9.5, "Data Layer", fontsize=11, fontweight="bold", color=COLORS["blue"])
    ax.text(0.7, 7.5, "Model Layer", fontsize=11, fontweight="bold", color=COLORS["blue"])
    ax.text(0.7, 5.5, "API Layer", fontsize=11, fontweight="bold", color=COLORS["blue"])
    ax.text(0.7, 3.5, "UI Layer", fontsize=11, fontweight="bold", color=COLORS["blue"])
    
    # Data layer boxes
    draw_box(ax, 1.5, 8.3, 2, 0.4, "Raw Data", COLORS["light_blue"], 8)
    draw_box(ax, 4, 8.3, 2, 0.4, "Processed CSV", COLORS["light_blue"], 8)
    draw_box(ax, 6.5, 8.3, 2, 0.4, "Metadata", COLORS["light_blue"], 8)
    
    # Model layer boxes
    draw_box(ax, 1.5, 6.8, 2, 0.4, "Preprocessing", COLORS["orange"], 8)
    draw_box(ax, 4, 6.8, 2, 0.4, "EFA Scoring", COLORS["orange"], 8)
    draw_box(ax, 6.5, 6.8, 2, 0.4, "5 Models", COLORS["orange"], 8)
    draw_box(ax, 9, 6.8, 2, 0.4, "Best Model", COLORS["orange"], 8)
    
    # API layer boxes
    draw_box(ax, 1.5, 4.8, 2, 0.4, "Validation", COLORS["red"], 8)
    draw_box(ax, 4, 4.8, 2, 0.4, "Predict API", COLORS["red"], 8)
    draw_box(ax, 6.5, 4.8, 2, 0.4, "What-If Lab", COLORS["red"], 8)
    draw_box(ax, 9, 4.8, 2, 0.4, "Insights API", COLORS["red"], 8)
    
    # UI layer boxes
    draw_box(ax, 2, 2.8, 3, 0.4, "Flask Dashboard", COLORS["green"], 8)
    draw_box(ax, 6, 2.8, 3, 0.4, "Prediction Forms", COLORS["green"], 8)
    
    # Vertical arrows
    for x in [2.5, 5, 7.5, 10]:
        draw_arrow(ax, x, 8.2, x, 7.1)
    for x in [2.5, 5, 7.5, 10]:
        draw_arrow(ax, x, 6.7, x, 5.1)
    for x in [3.5, 7.5]:
        draw_arrow(ax, x, 4.7, x, 3.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_3_2_artefact_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: fig_3_2_artefact_architecture.png")


def fig_4_1_training_pipeline():
    """Figure 4.1: Implemented training pipeline."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Figure 4.1: Implemented Training Pipeline", 
                 fontsize=16, fontweight="bold", pad=20)
    
    # Input
    draw_box(ax, 0.5, 6.5, 2, 0.6, "Input\nCSV", COLORS["light_blue"], 9)
    
    # Preprocessing
    draw_box(ax, 3, 6.5, 1.5, 0.6, "Clean", COLORS["orange"], 8)
    draw_box(ax, 4.8, 6.5, 1.5, 0.6, "Split", COLORS["orange"], 8)
    draw_box(ax, 6.6, 6.5, 1.5, 0.6, "Impute", COLORS["orange"], 8)
    draw_box(ax, 8.4, 6.5, 1.5, 0.6, "Scale", COLORS["orange"], 8)
    draw_box(ax, 10.2, 6.5, 1.5, 0.6, "Encode", COLORS["orange"], 8)
    draw_box(ax, 12, 6.5, 1.5, 0.6, "EFA", COLORS["orange"], 8)
    
    # CV
    draw_box(ax, 3, 4.5, 3, 0.8, "Cross-Validation\n5-Fold Stratified", COLORS["purple"], 9)
    
    # Models
    draw_box(ax, 7, 4.7, 1.2, 0.5, "LogReg", COLORS["red"], 8)
    draw_box(ax, 8.5, 4.7, 1.2, 0.5, "RF", COLORS["red"], 8)
    draw_box(ax, 10, 4.7, 1.2, 0.5, "XGB", COLORS["red"], 8)
    draw_box(ax, 11.5, 4.7, 1.2, 0.5, "LGBM", COLORS["red"], 8)
    draw_box(ax, 13, 4.7, 1.2, 0.5, "MLP", COLORS["red"], 8)
    
    # Selection
    draw_box(ax, 8, 3, 3, 0.8, "Model Selection\nMax CV Balanced Acc", COLORS["green"], 9)
    
    # Output
    draw_box(ax, 4, 1.5, 2, 0.6, "best_model.pkl", COLORS["gray"], 8)
    draw_box(ax, 6.5, 1.5, 2, 0.6, "metadata.json", COLORS["gray"], 8)
    draw_box(ax, 9, 1.5, 2, 0.6, "tables/", COLORS["gray"], 8)
    
    # Arrows
    draw_arrow(ax, 2.5, 6.8, 3, 6.8)
    for i in range(3, 12):
        draw_arrow(ax, i + 1.5, 6.8, i + 3, 6.8)
    draw_arrow(ax, 4.5, 6.4, 4.5, 5.4)
    draw_arrow(ax, 12.75, 6.4, 12.75, 5.4)
    draw_arrow(ax, 4.5, 4.4, 7.6, 4.95)
    draw_arrow(ax, 12.75, 4.4, 13.6, 4.95)
    for i in range(7, 13):
        draw_arrow(ax, i + 0.6, 4.7, i + 1.2, 4.7)
    draw_arrow(ax, 9.5, 4.4, 9.5, 3.9)
    draw_arrow(ax, 9.5, 2.9, 5, 2.1)
    draw_arrow(ax, 9.5, 2.9, 7.5, 2.1)
    draw_arrow(ax, 9.5, 2.9, 10, 2.1)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_4_1_training_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: fig_4_1_training_pipeline.png")


def fig_4_3_api_integration():
    """Figure 4.3: Flask API integration."""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("Figure 4.3: Flask API Integration", 
                 fontsize=16, fontweight="bold", pad=20)
    
    # Client
    draw_box(ax, 1, 7.5, 2.5, 0.8, "Client\nWeb/API", COLORS["light_blue"], 10)
    
    # Flask layers
    rect = Rectangle((4, 6), 9, 2.5, facecolor=COLORS["orange"], alpha=0.15, edgecolor=COLORS["blue"])
    ax.add_patch(rect)
    ax.text(4.2, 8.3, "Flask Application", fontsize=11, fontweight="bold", color=COLORS["blue"])
    
    draw_box(ax, 4.5, 7.5, 2, 0.5, "CORS", COLORS["orange"], 8)
    draw_box(ax, 7, 7.5, 2, 0.5, "Rate Limit", COLORS["orange"], 8)
    draw_box(ax, 9.5, 7.5, 2, 0.5, "Logger", COLORS["orange"], 8)
    
    draw_box(ax, 4.5, 6.5, 2, 0.5, "/predict", COLORS["red"], 8)
    draw_box(ax, 7, 6.5, 2, 0.5, "/whatif", COLORS["red"], 8)
    draw_box(ax, 9.5, 6.5, 2, 0.5, "/insights", COLORS["red"], 8)
    
    draw_box(ax, 5.5, 6, 3, 0.5, "Pydantic Validation", COLORS["purple"], 8)
    
    # Storage
    draw_box(ax, 4.5, 4, 2, 0.6, "model.pkl", COLORS["green"], 8)
    draw_box(ax, 7, 4, 2, 0.6, "metadata.json", COLORS["green"], 8)
    draw_box(ax, 9.5, 4, 2, 0.6, "dataset.csv", COLORS["green"], 8)
    
    # Arrows
    draw_arrow(ax, 3.5, 7.9, 4.5, 7.75)
    draw_arrow(ax, 6.5, 7.5, 7, 7.75)
    draw_arrow(ax, 9, 7.5, 9.5, 7.75)
    draw_arrow(ax, 5.5, 7.4, 5.5, 6.6)
    draw_arrow(ax, 8, 7.4, 8, 6.6)
    draw_arrow(ax, 10.5, 7.4, 10.5, 6.6)
    draw_arrow(ax, 7, 6.4, 7, 6.1)
    draw_arrow(ax, 7, 5.9, 5.5, 4.7)
    draw_arrow(ax, 7, 5.9, 8, 4.7)
    draw_arrow(ax, 7, 5.9, 10.5, 4.7)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_4_3_api_integration.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: fig_4_3_api_integration.png")


def fig_2_1_research_problem():
    """Figure 2.1: Research problem to artefact flow."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Figure 2.1: Research Problem to Artefact Flow", 
                 fontsize=16, fontweight="bold", pad=20)
    
    # Research problem
    draw_box(ax, 0.5, 6, 3, 1, "Research Problem\nClassify Operational Risk", COLORS["light_blue"], 10)
    
    # RQs
    draw_box(ax, 1, 4.5, 2, 0.6, "RQ1", COLORS["orange"], 9)
    draw_box(ax, 3.5, 4.5, 2, 0.6, "RQ2", COLORS["orange"], 9)
    draw_box(ax, 6, 4.5, 2, 0.6, "RQ3", COLORS["orange"], 9)
    
    # Requirements
    draw_box(ax, 1, 3, 2, 0.6, "DR1", COLORS["purple"], 9)
    draw_box(ax, 3.5, 3, 2, 0.6, "DR2", COLORS["purple"], 9)
    draw_box(ax, 6, 3, 2, 0.6, "DR3", COLORS["purple"], 9)
    draw_box(ax, 8.5, 3, 2, 0.6, "DR4", COLORS["purple"], 9)
    
    # Artefacts
    draw_box(ax, 1, 1.5, 2, 0.6, "Dataset", COLORS["green"], 9)
    draw_box(ax, 3.5, 1.5, 2, 0.6, "Models", COLORS["green"], 9)
    draw_box(ax, 6, 1.5, 2, 0.6, "SHAP", COLORS["green"], 9)
    draw_box(ax, 8.5, 1.5, 2, 0.6, "API", COLORS["green"], 9)
    
    # Arrows
    draw_arrow(ax, 2, 5.9, 2, 5.2)
    draw_arrow(ax, 4.5, 5.9, 4.5, 5.2)
    draw_arrow(ax, 7, 5.9, 7, 5.2)
    draw_arrow(ax, 2, 4.4, 2, 3.7)
    draw_arrow(ax, 4.5, 4.4, 4.5, 3.7)
    draw_arrow(ax, 7, 4.4, 7, 3.7)
    draw_arrow(ax, 7, 4.4, 9.5, 3.7)
    draw_arrow(ax, 2, 2.9, 2, 2.2)
    draw_arrow(ax, 4.5, 2.9, 4.5, 2.2)
    draw_arrow(ax, 7, 2.9, 7, 2.2)
    draw_arrow(ax, 9.5, 2.9, 9.5, 2.2)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_2_1_research_problem.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: fig_2_1_research_problem.png")


if __name__ == "__main__":
    print("Generating professional conceptual diagrams...")
    fig_2_1_research_problem()
    fig_3_1_research_workflow()
    fig_3_2_artefact_architecture()
    fig_4_1_training_pipeline()
    fig_4_3_api_integration()
    print("\nAll diagrams generated successfully!")
