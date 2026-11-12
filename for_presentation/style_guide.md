# Style guide (extracted from the sample deck)

Source: `reference/CSCI447_Final_Project_Sample_Presentation.pptx` (27 slides).
Rendering: Rendered 27 slides with LibreOffice + pdftoppm.

## Slide size
13.33 × 7.50 in (16:9). Margins: content starts 0.60 in from the left and spans 12.10 in.

## Theme colours (ppt/theme/theme1.xml)
| Role | Hex |
|---|---|
| dk1 | #000000 |
| lt1 | #FFFFFF |
| dk2 | #14213D |
| lt2 | #F3F5F9 |
| accent1 | #24375F |
| accent2 | #B8913A |
| accent3 | #2E7D7A |
| accent4 | #B4432E |
| accent5 | #7A8699 |
| accent6 | #C9A55C |
| hlink | #1F5FA8 |
| folHlink | #6B4E9B |

Roles used on the slides: dk2 #14213D = titles, big numbers, table header fill; accent1 #24375F =
numbered circles (first half); accent2 #B8913A = section labels, numbered circles (second half),
big-number accent; accent4 #B4432E = warnings/limitations circle; lt2 #F3F5F9 = cards and table
banding; #F3E9D2 = takeaway and callout boxes; body text #1F2937; secondary text #5B6472;
grid/borders #D5DAE3. Closing slide ("DARK" layout) background #14213D with white title.

## Fonts
- Headings: **Cambria** (titles 30 pt bold, colour #14213D; card titles 19–20 pt; big numbers 30–40 pt).
- Body: **Calibri** (body 15–16 pt; secondary 13–14 pt; tables 15 pt).
- Equations: Cambria Math 18 pt; equation numbers "(n)" 15 pt in #5B6472 at the right edge.
- Captions: Times New Roman 12 pt, "Table N:" placed **above** tables, "Fig. N:" placed **below** figures.

## Recurring elements
- **Section label** ("SECTION N · NAME", all caps): 12 pt bold, colour #B8913A, at (0.60 in, 0.30 in).
- **Title**: Cambria 30 pt, #14213D, at (0.60 in, 0.62 in).
- **Footer**: 3 parts separated by "   |   " — course name, "Project ID: <ID>", project title (our footer text is given on slide 1 of `slides.md`); bottom left (0.60 in, 7.05 in; 11 pt). **Slide number** bottom right (12.20 in, 7.03 in).
- **Tables**: header row fill #14213D with white bold 15 pt text; banded rows #F3F5F9 / white; centred numbers; best value per column in bold.
- **Cards**: rounded rectangles #F3F5F9 with a numbered circle (0.50 in, white bold number) and a Cambria 19–20 pt card title.
- **Big-number callouts**: rounded #F3F5F9 boxes 3.85 × 1.0–1.5 in; value in Cambria 30–34 pt bold (#14213D, or #B4432E for a cost/negative); label 14 pt #5B6472 below.
- **Takeaway box**: full-width rounded #F3E9D2 box (12.10 × 1.10 in) at the bottom; "Takeaway:" in bold 15–16 pt.
- **Roadmap**: two columns of numbered circles (navy 1–5, gold 6–9) and a #F3E9D2 presenter callout.
- **Algorithm slide**: pseudocode card on the left (Cambria 12 pt, line numbers in #5B6472), grouped explanation cards on the right alternating #F3E9D2 / #F3F5F9.

## Model colours used in our figures
Logistic Regression #1F5FA8, Random Forest #A8822C, XGBoost #109A92, LightGBM #B4432E, MLP #6B4E9B
(deck hues adjusted for colour-blind separation); cycles from light to dark navy (#9FB0CC, #4F6A99, #14213D).
