# Ayna_Assignment_PolygonColourization
Here's an updated `README.md` for your Conditional U-Net polygon color-filling project, with Weights & Biases and command-line training instructions removed, and clear guidance for using the project with a Jupyter/Colab notebook.

#  Polygon Color Filling using Conditional U-Net

This repository provides a PyTorch-based implementation of a **Conditional U-Net** for automatic polygon color filling. The model receives a polygon outline and a target color, then outputs the color-filled result.

##  Project Overview

**Goal:**  
Given an outline (grayscale) image of a polygon and a color name, generate a filled version with the chosen color.

**Architecture:**  
Conditional U-Net supporting various conditioning techniques:
- `film`: Feature-wise Linear Modulation
- `concat_rgb`: Concatenate target RGB values to input image
- `concat_idx`: Concatenate embedding of color index

##  Dataset Structure

```
dataset/
├── training/
│   ├── inputs/         # Polygon outline PNGs
│   ├── outputs/        # Color-filled PNGs
│   └── data.json       # Maps input/output filenames and color names
├── validation/
│   └── ... (same as training)
```
A typical `data.json` entry:
```json
{
  "input_polygon": "hexagon.png",
  "colour": "magenta",
  "output_image": "magenta_hexagon.png"
}
```

##  Setup

1. **Clone this repository:**
   ```bash
   !git clone https://github.com/SreejaBommagani/Ayna_Assignment_PolygonColourization.git
   %cd Ayna_Assignment_PolygonColourization

   ```
2.  Open the provided notebook in Google Colab for an interactive workflow.

##  Using the Project in Colab or Jupyter

- **Simply upload or open the included notebook** (`.ipynb` file) in Colab or Jupyter.
- The notebook guides you through:
  - Importing dependencies
  - Loading polygon data
  - Choosing the conditioning method
  - Training the Conditional U-Net model
  - Generating and visualizing color-filled polygons
  
- **No manual script running is required**—all steps are self-contained in the notebook with explanations and output visualizations.

##  Results

| Metric   | Sample Value |
|----------|--------------|
| PSNR     | ~28.5 dB     |
| MSE      | ~0.002       |
| L1 Loss  | ~0.03        |

##  Report

For a detailed discussion of:
- Hyperparameter tuning
- Model design choices
- Experiment results
- Insights and learning

See: `Ayna_Assignment_Report.pdf`

## Author
Sreeja Bommagani
