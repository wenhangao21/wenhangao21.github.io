---
layout: archive
title: "First or Co-first Author Publications"
permalink: /publications/
author_profile: true
---
\* indicates equal contributors.

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}

# First or Co-first Author Preprints
* One paper on the resolution-invariance and discretization mismatch errors in neural operators; submitted to **ICLR 25**; **W.Gao**, R.Xu, Y.Deng, Y.Liu
* One paper addressing the misalignment between optimized masks and actual explanatory subgraphs in 3D GNNs; ubmitted to **ICLR 25**; X.Liu\*, **W.Gao\***, Y.Liu
* One paper on designing enhanced kernels for localized effects and interactions in FNO; **Preprint**; **W.Gao**, J.Luo, R.Xu, Y.Liu

# Other Publications

*　Xufeng Liu, Dongsheng Luo, **Wenhan Gao**, Yi Liu, 3DGraphX: Explaining 3D Molecular Graph Models via Incorporating Chemical Priors, The 31st SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2025.