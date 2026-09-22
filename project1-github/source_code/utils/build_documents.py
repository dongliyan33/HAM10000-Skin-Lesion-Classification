"""Generate the Final Project report and presentation from saved real results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches as PInches, Pt


REFERENCES = [
    "[1] P. Tschandl, C. Rosendahl, and H. Kittler, “The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions,” Scientific Data, vol. 5, Art. no. 180161, 2018, doi: 10.1038/sdata.2018.161.",
    "[2] A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in Advances in Neural Information Processing Systems 32, 2019, pp. 8024–8035.",
    "[3] M. Tan and Q. V. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in Proc. ICML, 2019, pp. 6105–6114.",
    "[4] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE CVPR, 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.",
    "[5] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” in Proc. IEEE CVPR, 2017, pp. 4700–4708, doi: 10.1109/CVPR.2017.243.",
    "[6] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in Proc. ICLR, 2015, arXiv:1409.1556.",
    "[7] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in MICCAI, 2015, pp. 234–241, doi: 10.1007/978-3-319-24574-4_28.",
    "[8] Z. Abou El Houda, A. S. Hafid, L. Khoukhi, and B. Brik, “When collaborative federated learning meets blockchain to preserve privacy in healthcare,” IEEE Trans. Network Science and Engineering, vol. 10, no. 5, pp. 2455–2465, 2023, doi: 10.1109/TNSE.2022.3211192.",
]


def create_dataset_figure(data_dir: Path, output: Path):
    metadata_path = data_dir / "HAM10000_metadata"
    if not metadata_path.exists(): metadata_path = data_dir / "HAM10000_metadata.csv"
    metadata = pd.read_csv(metadata_path)
    image_map = {}
    for folder in data_dir.glob("HAM10000_images_part_*"):
        image_map.update({path.stem: path for path in folder.glob("*.jpg")})
    mask_dir = data_dir / "HAM10000_segmentations_lesion_tschandl"
    samples = [metadata[metadata.dx == name].iloc[0] for name in sorted(metadata.dx.unique())]
    fig, axes = plt.subplots(2, 7, figsize=(14, 4.5))
    for column, row in enumerate(samples):
        image = Image.open(image_map[row.image_id]).convert("RGB")
        mask = Image.open(mask_dir / f"{row.image_id}_segmentation.png").convert("L")
        axes[0, column].imshow(image); axes[0, column].set_title(row.dx)
        axes[1, column].imshow(mask, cmap="gray"); axes[1, column].set_title("mask")
        axes[0, column].axis("off"); axes[1, column].axis("off")
    fig.suptitle("HAM10000 examples and corresponding real lesion masks")
    fig.tight_layout(); fig.savefig(output, dpi=250, bbox_inches="tight"); plt.close(fig)


def add_doc_table(document, frame, formats=None):
    table = document.add_table(rows=1, cols=len(frame.columns)); table.style = "Table Grid"
    for index, column in enumerate(frame.columns): table.rows[0].cells[index].text = str(column)
    for _, row in frame.iterrows():
        cells = table.add_row().cells
        for index, column in enumerate(frame.columns):
            value = row[column]
            cells[index].text = formats.get(column, lambda x: str(x))(value) if formats else str(value)
    return table


def build_report(project: Path):
    comparison = pd.read_csv(project / "results/classification_model_comparison.csv")
    segmentation = json.load((project / "results/segmentation_metrics.json").open())
    distribution = pd.read_csv(project / "dataset_manifest/class_distribution.csv")
    best = comparison.sort_values("Accuracy", ascending=False).iloc[0]
    document = Document(); section = document.sections[0]; section.top_margin=Inches(.7); section.bottom_margin=Inches(.7)
    title=document.add_heading("HAM10000 Skin Lesion Classification and Segmentation",0); title.alignment=WD_ALIGN_PARAGRAPH.CENTER
    subtitle=document.add_paragraph("Final Project Report"); subtitle.alignment=WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph("All reported measurements were loaded from saved experiment outputs. Existing classification checkpoints were not retrained for this report.")

    sections = [
        ("Abstract", f"This project investigates automated skin-lesion analysis using 10,015 HAM10000 dermoscopic images across seven diagnostic classes. Five established convolutional architectures and a custom CNN were evaluated on one canonical lesion-level validation split using Apple MPS. {best.Model} achieved the highest classification accuracy ({best.Accuracy:.2%}) and macro F1 ({best['Macro F1']:.4f}). A compact U-Net trained with 10,015 genuine lesion masks achieved Dice {segmentation['dice']:.4f} and IoU {segmentation['iou']:.4f}. The results show that efficient transfer-learning architectures provide strong classification performance while pixel-level segmentation can accurately localize lesion regions."),
        ("1. Introduction", "Skin cancer screening benefits from consistent analysis of dermoscopic images, but visual similarity among lesion types and severe class imbalance make automated diagnosis challenging. Deep convolutional networks can learn discriminative features, while segmentation separates the lesion from surrounding skin. This project combines both tasks in one reproducible PyTorch pipeline."),
        ("2. Related Work", "HAM10000 was introduced as a multi-source dermatoscopic benchmark [1]. Residual connections [4], dense connectivity [5], very deep VGG networks [6], and compound EfficientNet scaling [3] provide complementary approaches to image classification. U-Net introduced an encoder–decoder architecture with skip connections for biomedical segmentation [7]. PyTorch supplies the training and MPS execution framework [2]. The course-referenced HealthFed study illustrates broader privacy and collaboration concerns for healthcare AI [8]."),
        ("3. Dataset", "The project uses all 10,015 HAM10000 RGB images and all 10,015 corresponding genuine lesion masks. The seven labels are akiec, bcc, bkl, df, mel, nv, and vasc. The data are highly imbalanced, with melanocytic nevi forming the majority. No synthetic ground-truth mask was created."),
        ("4. Methods", "Classification images were resized to 224×224 and normalized using ImageNet statistics. Training augmentation included flips, rotations, and mild color jitter. Final evaluation used deterministic preprocessing. The canonical 80/20 split was grouped by lesion_id with seed 42, producing 7,967 training and 2,048 validation images. Grouping prevents different images of the same lesion from crossing the split."),
        ("5. Classification", "The evaluated models were ResNet50, VGG16, VGG19, DenseNet121, EfficientNet-B0, and a four-block custom CNN. Saved checkpoints were loaded without retraining and inferred on the exact same validation image IDs. Accuracy, per-class precision/recall/F1, macro averages, weighted averages, confusion matrices, and parameter counts were recorded."),
        ("6. Segmentation", "A compact U-Net used four encoder and decoder levels with skip connections. Images and masks were jointly resized to 128×128 and augmented with synchronized flips and rotations. The objective combined binary cross-entropy and soft Dice loss. Training used five epochs, batch size 32, Adam optimization, lesion-level splitting, and Apple MPS."),
        ("7. Experiments", "All completed final classifier inference ran on Apple MPS. The six checkpoints were evaluated on 2,048 canonical validation images. Segmentation used 7,967 training and 2,048 validation pairs. Existing classifier histories were retained for training-curve analysis; the custom CNN lacks a genuine epoch-level history, so no curve was fabricated."),
    ]
    for heading, body in sections: document.add_heading(heading,1); document.add_paragraph(body)
    document.add_picture(str(project / "figures/dataset_examples.png"), width=Inches(6.8)); document.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER

    document.add_heading("8. Results",1)
    display = comparison[["Model","Accuracy","Macro Precision","Macro Recall","Macro F1","Weighted F1","Parameters"]]
    add_doc_table(document, display, {"Model":str,"Accuracy":lambda x:f"{x:.4f}","Macro Precision":lambda x:f"{x:.4f}","Macro Recall":lambda x:f"{x:.4f}","Macro F1":lambda x:f"{x:.4f}","Weighted F1":lambda x:f"{x:.4f}","Parameters":lambda x:f"{int(x):,}"})
    document.add_picture(str(project / "figures/canonical_accuracy_macro_f1.png"), width=Inches(6.5))
    document.add_paragraph(f"Segmentation results: Dice {segmentation['dice']:.4f}, IoU {segmentation['iou']:.4f}, precision {segmentation['precision']:.4f}, recall {segmentation['recall']:.4f}, and combined loss {segmentation['loss']:.4f}.")
    document.add_picture(str(project / "figures/segmentation_examples.png"), width=Inches(6.5))

    remaining = [
        ("9. Discussion", f"{best.Model} provided the strongest balance of accuracy and macro F1 with only {int(best.Parameters):,} parameters. Its accuracy exceeds macro F1, showing that performance on the dominant nv class is stronger than balanced performance across diagnoses. VGG19 has lower accuracy but high macro recall, consistent with more minority-class predictions. The compact CNN is computationally small but underfits the classification task. Segmentation scores indicate strong region overlap, although the experiment uses an internal validation split rather than an external cohort."),
        ("10. Limitations", "The original classifiers were trained under different historical split strategies and hyperparameters. Canonical inference makes final evaluation samples identical, but it cannot remove differences in training exposure. There is no independent external test set, clinical calibration analysis, or prospective validation. HAM10000 is imbalanced, and diagnostic labels do not replace histopathological confirmation. Segmentation was trained at 128×128 for efficiency. The custom CNN training history is unavailable."),
        ("11. Future Work", "Future work should define one lesion-level train/validation/test split before training, use imbalance-aware losses or sampling, assess calibration and uncertainty, evaluate on external datasets, increase segmentation resolution, explore joint classification–segmentation learning, and investigate privacy-aware multi-institutional training."),
        ("12. Conclusion", f"A complete classification and segmentation pipeline was developed for HAM10000. On the shared validation split, {best.Model} achieved {best.Accuracy:.2%} accuracy and macro F1 {best['Macro F1']:.4f}. The U-Net achieved Dice {segmentation['dice']:.4f}. Reporting macro metrics, confusion matrices, and methodological limitations provides a more reliable interpretation than accuracy alone."),
    ]
    for heading, body in remaining: document.add_heading(heading,1); document.add_paragraph(body)
    document.add_heading("References",1)
    for reference in REFERENCES: document.add_paragraph(reference)
    document.save(project / "Project_Report.docx")


def add_slide(prs, title, bullets, picture=None, picture_width=5.6):
    slide=prs.slides.add_slide(prs.slide_layouts[5]); slide.background.fill.solid(); slide.background.fill.fore_color.rgb=RGBColor(248,250,252)
    title_box=slide.shapes.add_textbox(PInches(.65),PInches(.3),PInches(12),PInches(.7)); p=title_box.text_frame.paragraphs[0]; p.text=title; p.font.size=Pt(28); p.font.bold=True; p.font.color.rgb=RGBColor(15,23,42)
    width=6.0 if picture else 11.6; body=slide.shapes.add_textbox(PInches(.8),PInches(1.2),PInches(width),PInches(5.7)).text_frame; body.clear()
    for index,item in enumerate(bullets):
        p=body.paragraphs[0] if index==0 else body.add_paragraph(); p.text=item; p.font.size=Pt(19); p.space_after=Pt(12); p.font.color.rgb=RGBColor(30,41,59)
    if picture and Path(picture).exists(): slide.shapes.add_picture(str(picture),PInches(7.0),PInches(1.25),width=PInches(picture_width))
    return slide


def build_presentation(project: Path):
    comparison=pd.read_csv(project/"results/classification_model_comparison.csv"); seg=json.load((project/"results/segmentation_metrics.json").open()); best=comparison.iloc[0]
    prs=Presentation(); prs.slide_width=PInches(13.333); prs.slide_height=PInches(7.5)
    add_slide(prs,"HAM10000 Skin Lesion Classification and Segmentation",["Final Project","Six classification checkpoints • lesion segmentation • Apple MPS","All figures and metrics come from saved experiments"],project/"figures/dataset_examples.png")
    add_slide(prs,"Motivation",["Early and consistent lesion assessment is clinically valuable","Dermoscopic classes have overlapping appearance","Class imbalance can hide minority-class failures","Segmentation adds spatial lesion localization"])
    add_slide(prs,"Background",["Transfer learning reuses ImageNet feature representations","Residual, dense, VGG and compound-scaled CNNs capture features differently","U-Net combines encoder context with decoder localization","Accuracy is reported together with macro and weighted F1"])
    add_slide(prs,"Dataset",["HAM10000: 10,015 dermoscopic RGB images","Seven diagnoses; nv is the majority class","10,015 genuine lesion segmentation masks","Official source: Harvard Dataverse (doi:10.7910/DVN/DBW86T)"],project/"figures/dataset_examples.png")
    add_slide(prs,"Data Split and Preprocessing",["Canonical lesion-level split with seed 42","7,967 training / 2,048 validation images","Classification: 224×224 and ImageNet normalization","Segmentation: paired 128×128 image/mask transforms","No independent external test set"])
    add_slide(prs,"Classification Models",["ResNet50: residual connections","VGG16/VGG19: deep sequential convolution blocks","DenseNet121: dense feature reuse","EfficientNet-B0: compound depth/width/resolution scaling","All existing checkpoints evaluated without retraining"],project/"figures/five_architectures_comparison.png")
    add_slide(prs,"Custom CNN",["Four convolutional blocks with batch normalization","Adaptive pooling and a compact classification head","423,175 parameters — smallest classifier","Canonical accuracy 48.29%; macro F1 0.3376","No epoch history was fabricated"])
    add_slide(prs,"Segmentation Method",["Compact U-Net with four encoder/decoder levels","Skip connections preserve spatial information","BCE + soft Dice objective","Adam, batch size 32, five epochs, MPS","Ground truth comes from real HAM10000 masks"],project/"figures/segmentation_examples.png")
    add_slide(prs,"Experimental Setup",["macOS / Apple Silicon / Python 3.13","PyTorch 2.13.0 and torchvision 0.28.0","All final classification inference device: mps","Same 2,048 image IDs for every classifier","Metrics: accuracy, precision, recall, F1 and confusion matrices"])
    slide=add_slide(prs,"Classification Results",[f"Best: {best.Model} — {best.Accuracy:.2%} accuracy",f"Macro F1 {best['Macro F1']:.4f}; weighted F1 {best['Weighted F1']:.4f}","Accuracy–macro F1 gaps reveal class imbalance"],project/"figures/canonical_accuracy_macro_f1.png")
    add_slide(prs,"Best Model Confusion Matrix",["EfficientNet-B0 performs strongly on nv","Melanoma remains a difficult minority class","Major confusion includes mel → nv","Per-class analysis is required for safe interpretation"],project/"results/confusion_matrices/EfficientNetB0_confusion_matrix.png",5.3)
    add_slide(prs,"Segmentation Results",[f"Dice: {seg['dice']:.4f}",f"IoU: {seg['iou']:.4f}",f"Precision: {seg['precision']:.4f}",f"Recall: {seg['recall']:.4f}","Strong overlap on the internal lesion-level validation set"],project/"figures/segmentation_examples.png")
    add_slide(prs,"Discussion and Limitations",["EfficientNet-B0 offers the best accuracy/parameter balance","VGG models are substantially larger","Historical training protocols were not identical","Canonical inference aligns samples but not training exposure","No external test cohort or clinical validation"])
    add_slide(prs,"Conclusion and Future Work",[f"EfficientNet-B0: {best.Accuracy:.2%} canonical accuracy",f"U-Net: Dice {seg['dice']:.4f}","Use one fixed split before future training","Improve imbalance handling and calibration","Evaluate externally and explore joint multi-task learning"])
    add_slide(prs,"References",["[1] Tschandl et al., HAM10000, Scientific Data, 2018","[2] Paszke et al., PyTorch, NeurIPS, 2019","[3] Tan and Le, EfficientNet, ICML, 2019","[4] He et al., ResNet, CVPR, 2016","[5] Huang et al., DenseNet, CVPR, 2017","[6] Simonyan and Zisserman, VGG, ICLR, 2015","[7] Ronneberger et al., U-Net, MICCAI, 2015","[8] Abou El Houda et al., IEEE TNSE, 2023"])
    prs.save(project/"Final_Project_Presentation.pptx")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--project",type=Path,required=True); parser.add_argument("--data-dir",type=Path,required=True); args=parser.parse_args()
    args.project.joinpath("figures").mkdir(exist_ok=True); create_dataset_figure(args.data_dir,args.project/"figures/dataset_examples.png")
    build_report(args.project); build_presentation(args.project); print("Report and 15-slide presentation generated.")


if __name__ == "__main__": main()
