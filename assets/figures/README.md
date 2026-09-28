# Paper figure assets

These are the authors' original figure sources for
[POLAR-Bench, arXiv:2605.19127v1](https://arxiv.org/abs/2605.19127v1), exported for
the project README and dataset documentation. They are **not screenshots,
new experiments, or redrawn approximations**.

| Published asset | Paper figure | Original LaTeX source filename |
| --- | --- | --- |
| [concept-overview.jpg](concept-overview.jpg) | Figure 1: conceptual overview | `outline_more_wide.jpg` |
| [pipeline.png](pipeline.png) / [PDF](pipeline.pdf) | Figure 9: benchmark pipeline | `simple_benchmark_pipeline.pdf` |
| [privacy-utility.png](privacy-utility.png) / [PDF](privacy-utility.pdf) | Figure 3: Privacy–Utility scatter | `pareto_scatter_multilambda.pdf` |
| [diagnostic-surface.png](diagnostic-surface.png) / [PDF](diagnostic-surface.pdf) | Figure 4: policy–attack surfaces | `average_privacy_utility_surfaces_high_contrast.pdf` |
| [word-counts.png](word-counts.png) / [PDF](word-counts.pdf) | Figure 6: text lengths | `word_count_raincloud_camera_ready.pdf` |
| [attribute-counts.png](attribute-counts.png) / [PDF](attribute-counts.pdf) | Figure 7: attribute composition | `attribute_distribution_panel.pdf` |

The concept illustration is an unchanged **2,155 × 753 pixel JPEG** copied from
the paper's original figure source, not an enlarged screenshot. Its original
conceptual wording is retained; the README clarifies that the implemented
Attribute Utility metric is information coverage, not end-to-end task success.

PDFs are unchanged copies of the LaTeX figure files. PNG previews are rendered
directly from those PDFs with Poppler at **2,800 pixels on the longest edge**,
on a white background. The PDFs retain the original scalable content for zooming
and reuse. No figures were generated with an image-generation model.

To reproduce the previews from the PDFs in this directory:

```bash
for figure in pipeline privacy-utility diagnostic-surface word-counts attribute-counts; do
  pdftoppm -png -singlefile -scale-to 2800 "${figure}.pdf" "${figure}"
done
```

The plots retain the paper's original terminology and numbers. In accompanying
documentation, **Attribute Utility** clarifies the original plots' “Utility”:
coverage of required information, not end-to-end task success. Figure 7's
“private” category means the protected-attribute set.

License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/),
as for the paper. Credit Qiaoyuan Zheng, Yiqu Yang, Qi Gao, and Imanol Schlag and
cite the paper when reusing these figures. The repository's MIT license applies
to code; it does not replace the figure license.
